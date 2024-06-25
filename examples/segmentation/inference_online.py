"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
"""
import math
import copy
import project_inst
import map_utils
import open3d as o3d
#import info_proc
import expand_polygon
import get_info
import get_instances
import conversion_utils
from natsort import natsorted
import time
import glob
import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port, load_checkpoint_inv
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.dataset.semantic_kitti.semantickitti import load_label_kitti, load_pc_kitti, remap_lut_read, remap_lut_write, get_semantickitti_file_list
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def generate_test_list(cfg):
    test_list = sorted(os.listdir(cfg.path_test))
    test_list = [os.path.join(cfg.path_test, item) for item in test_list]
    return test_list


def load_data(file_path, cfg):
    data = np.load(file_path)  # xyzrgbl, N*7
    if cfg.points != 0:
        sub_idx = np.linspace(0, data.shape[0]-1, cfg.points, dtype=int)
        sub_data = data[sub_idx]
    else:
        sub_data = data
    coord, feat = sub_data[:, :3], sub_data[:, 3:6]
    
    feat = np.clip(feat / 255., 0, 1).astype(np.float32)

    idx_points = []
    voxel_idx, reverse_idx_part,reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)

    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max()+1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle] # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0) # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(coord.shape[0]))

    return coord, feat, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


def main(gpu, cfg):

    online = 0

    col_inst = {
    0: [255, 255, 0],
    1: [255, 0, 255],
    2: [0, 255, 255],
    3: [0, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 0],
    6: [0, 255, 0],
    7: [0, 0, 255],
    8: [255, 0, 0],
    9: [0, 100, 0],
    10: [0, 0, 100],
    11: [100, 0, 0],
    12: [100, 0, 255],
    13: [0, 255, 100],
    13: [255, 100, 0]
    }

    # init get_instances parameters
    rad_p = 0.04               # max distance for pipe growing                             //PARAM
    rad_v = 0.04               # max distance for valve growing                            //PARAM
    dim_p = 3                  # compute 2D (2) or 3D (3) distance for pipe growing        //PARAM
    dim_v = 2                  # compute 2D (2) or 3D (3) distance for valve growing       //PARAM
    min_p_p = 50               # minimum number of points to consider a blob as a pipe     //PARAM
    min_p_v = 30 # 40 80 140   # minimum number of points to consider a blob as a valve    //PARAM

    targets_path = "/home/miguel/PointNeXt/valve_targets"

    n_pc = 0

    now = time.time()
    tzero = now-now

    T_read = tzero
    T_inferference = tzero

    T_instaces_valve = tzero
    T_instaces_pipe = tzero

    T_info_valve = tzero
    T_info_pipe = tzero

    T_ref_valve = tzero
    T_ref_pipe = tzero

    T_publish =  tzero
    T_total = tzero

    # get valve matching targets
    targets_list = list()
    for file_name in natsorted(os.listdir(targets_path)):
        target_path = os.path.join(targets_path, file_name)
        target = get_info.read_ply(target_path, "model")
        xyz_central = np.mean(target, axis=0)[0:3]
        target[:, 0:3] -= xyz_central  
        target_o3d = o3d.geometry.PointCloud()
        target_o3d.points = o3d.utility.Vector3dVector(target[:,0:3])
        target_o3d.colors = o3d.utility.Vector3dVector(target[:,3:6])
        target_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
        target_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))
        targets_list.append(target_o3d)


    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)


    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    cfg.cmap = np.array(cfg.cmap)

    # optionally resume from a checkpoint
    model_module = model.module if hasattr(model, 'module') else model
    best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)


    model.eval()  # set model to eval mode
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    set_random_seed(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    dataset_name = cfg.dataset.common.NAME.lower()

    cfg.save_path = cfg.get('save_path', f'results/{cfg.task_name}/{cfg.dataset.test.split}/{cfg.cfg_basename}')
    os.makedirs(cfg.save_path, exist_ok=True)

    gravity_dim = cfg.datatransforms.kwargs.gravity_dim
    nearest_neighbor = cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor'

    while 1:

        data_list = generate_test_list(cfg)

        for file_path in data_list:

            t0 = time.time()

            all_logits = []
            coord, feat, idx_points, voxel_idx, reverse_idx_part, reverse_idx  = load_data(file_path, cfg)

            name = os.path.basename(file_path)[:-4]
            
            if online==1:
                os.remove(file_path)
            
            t1 = time.time()

            len_part = len(idx_points)
            nearest_neighbor = len_part == 1
            pbar = tqdm(range(len(idx_points)))
            for idx_subcloud in pbar:
                pbar.set_description(f"Inference on {name}")
                if not (nearest_neighbor and idx_subcloud>0):
                    idx_part = idx_points[idx_subcloud]
                    coord_part = coord[idx_part]
                    coord_part -= coord_part.min(0)

                    feat_part =  feat[idx_part] if feat is not None else None
                    data = {'pos': coord_part}
                    if feat_part is not None:
                        data['x'] = feat_part
                    if pipe_transform is not None:
                        data = pipe_transform(data)
                    if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                        data['heights'] = torch.from_numpy(coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)
                    if not cfg.dataset.common.get('variable', False):
                        if 'x' in data.keys():
                            data['x'] = data['x'].unsqueeze(0)
                        data['pos'] = data['pos'].unsqueeze(0)
                    else:
                        data['o'] = torch.IntTensor([len(coord)])
                        data['batch'] = torch.LongTensor([0] * len(coord))

                    for key in data.keys():
                        data[key] = data[key].cuda(non_blocking=True)
                    data['x'] = get_features_by_keys(data, cfg.feature_keys)
                    logits = model(data)
                    
                all_logits.append(logits)
            all_logits = torch.cat(all_logits, dim=0)
            if not cfg.dataset.common.get('variable', False):
                all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

            if not nearest_neighbor:
                # average merge overlapped multi voxels logits to original point set
                idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
                all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
            else:
                # interpolate logits by nearest neighbor
                all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]
            pred = all_logits.argmax(dim=1)

            pred = pred.cpu().numpy().squeeze()
            pred = cfg.cmap[pred, :]

            t2 = time.time()


            # output pred labels
            write_obj(coord, feat, os.path.join(cfg.path_out, f'input-{name}.obj'))
            # output pred labels
            write_obj(coord, pred, os.path.join(cfg.path_out, f'{cfg.cfg_basename}-{name}.obj'))
            
            t3 = time.time()

            pred_sub = np.hstack((coord, feat)) 

            class_list = []
            for row in pred:
                if row[1] == 1:
                    class_list.append(1)
                elif row[2] == 1:
                    class_list.append(2)
                else:
                    class_list.append(0)

            class_np = np.array(class_list)

            pred_sub = np.c_[pred_sub,class_np]

            pred_sub_pipe = pred_sub[pred_sub[:,6] == 1]       # get points predicted as pipe
            pred_sub_valve = pred_sub[pred_sub[:,6] == 2]     # get points predicted as valve

            # get valve instances
            instances_ref_valve_list, pred_sub_pipe_ref, stolen_list  = get_instances.get_instances(pred_sub_valve, dim_v, rad_v, min_p_v, ref=True, ref_data = pred_sub_pipe, ref_rad = 0.1)
            #instances_ref_valve_list, pred_sub_pipe_ref, stolen_list  = get_instances.get_instances_o3d(pred_sub_valve, dim_v, rad_v, min_p_v, ref=True, ref_data = pred_sub_pipe, ref_rad = 0.1)
            t4 = time.time()

            # get valve information
            info_valves_list = list()
            for i, inst in enumerate(instances_ref_valve_list): # for each valve instance
                # transform instance to o3d pointcloud
                xyz_central = np.mean(inst, axis=0)[0:3]    # get isntance center
                inst[:, 0:3] -= xyz_central                 # move center to origin
                inst_o3d = o3d.geometry.PointCloud()
                inst_o3d.points = o3d.utility.Vector3dVector(inst[:,0:3])
                inst_o3d.colors = o3d.utility.Vector3dVector(inst[:,3:6])
                inst_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15)) # compute normal
                inst_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))                   # align normals
                inst[:, 0:3] += xyz_central                                                                          # recover original position

                info_valve = get_info.get_info(inst_o3d, targets_list, method="matching")        # get valve instance info list([fitness1, rotation1],[fitness2, rotation2], ...)  len = len(targets_list)
                max_info =  max(info_valve)                                                      # the max() function compares the first element of each info_list element, which is fitness)
                max_idx = info_valve.index(max_info)                                             # idx of best valve match
                
                rad = math.radians(max_info[1])
                vector = np.array([math.cos(rad), math.sin(rad), 0])                             # get valve unit vector at zero
                vector = vector*0.18                                                             # resize vector to valve size //PARAM
                info_valves_list.append([xyz_central, vector, max_idx, inst[:,0:3], max_info])   # append valve instance info


            # based on valve fitness, delete it and return stolen points to pipe prediction
            descart_valves_list = [i for i, x in enumerate(info_valves_list) if x[4][0] < 0.4]     # if max fitnes < thr  //PARAM
            for i in descart_valves_list:
                print("Valve descarted")
                descarted_points = np.vstack(instances_ref_valve_list[i])                           # notate points to discard
                if len(stolen_list[i])>0:                                                                  # if there were stolen points
                    stolen_idx = list(np.vstack(stolen_list[i])[:,0].astype(int))                       # get stolen idx
                    stolen_cls = np.vstack(stolen_list[i])[:,1].astype(int)                             # get stolen class
                    stolen_cls = stolen_cls.reshape(stolen_cls.shape[0],1)                              # reshape stolen class
                    stolen_points = descarted_points[stolen_idx, :-2]                               # recover stolen points
                    stolen_points = np.concatenate((stolen_points,stolen_cls),axis=1)               # concatenate stolen points and stolen class
                    pred_sub_pipe_ref = np.concatenate((pred_sub_pipe_ref,stolen_points),axis=0)    # add points and class pipe prediction points
            
            for index in sorted(descart_valves_list, reverse=True):                                 # delete discarted valve info                                                                             
                del info_valves_list[index]
                del instances_ref_valve_list[index]     # for print only  
            t5 = time.time()

            # get pipe instances
            instances_ref_pipe_list, _, _  = get_instances.get_instances(pred_sub_pipe_ref, dim_p, rad_p, min_p_p)
            #instances_ref_pipe_list, _, _  = get_instances.get_instances_o3d(pred_sub_pipe_ref, dim_p, rad_p, min_p_p)
            t6 = time.time()

            info_pipes_list = list()
            info_connexions_list = list()
            k_pipe = 0

            for i, inst in enumerate(instances_ref_pipe_list): # for each pipe instance

                info_pipe = get_info.get_info(inst, models=0, method="skeleton") # get pipe instance info list( list( list(chain1, start1, end1, elbow_list1, vector_chain_list1), ...), list(connexions_points)) 
                
                for j, pipe_info in enumerate(info_pipe[0]):                         # stack pipes info
                    inst_list = list()
                    inst_list.append(i)
                    pipe_info.append(inst_list)
                    info_pipes_list.append(pipe_info)

                for j, connexion_info in enumerate(info_pipe[1]):                    # stack conenexions info
                    connexion_info[1] = [x+k_pipe for x in connexion_info[1]]
                    info_connexions_list.append(connexion_info)

                k_pipe += len(info_pipe[0])                                          # update actual pipe idx
            t7 = time.time()

            info_pipes_list_copy = copy.deepcopy(info_pipes_list) 
            info_connexions_list_copy = copy.deepcopy(info_connexions_list)
            info_pipes_list2, info_connexions_list2 = get_info.unify_chains(info_pipes_list_copy, info_connexions_list_copy)
            t8 = time.time()

            info_valves_list_copy = copy.deepcopy(info_valves_list)
            info_valves_list2 = get_info.refine_valves(info_valves_list_copy, info_pipes_list2) 
            t9 = time.time()

            info1 = [info_pipes_list, info_connexions_list, info_valves_list, instances_ref_pipe_list]
            #info2 = [info_pipes_list2, info_connexions_list2, info_valves_list, instances_ref_pipe_list] 
            info3 = [info_pipes_list2, info_connexions_list2, info_valves_list2, instances_ref_pipe_list]

            empty1 = False
            if len(info_pipes_list)==0 and len(info_connexions_list)==0 and len(info_valves_list)==0 and len(instances_ref_pipe_list)==0:
                empty1 = True
            
            empty3 = False
            if len(info_pipes_list2)==0 and len(info_connexions_list2)==0 and len(info_valves_list2)==0 and len(instances_ref_pipe_list)==0:
                empty3 = True

            if empty1==False:
                path_out1 = os.path.join(cfg.path_out, name +'_info.ply')
                conversion_utils.info_to_ply(info1, path_out1)
                
            #path_out2 = os.path.join(cfg.path_out, name +'_info2.ply')
            #conversion_utils.info_to_ply(info2, path_out2)

            if empty3==False:
                path_out3 = os.path.join(cfg.path_out, name +'_info_ref.ply')
                conversion_utils.info_to_ply(info3, path_out3)

                info3_array = conversion_utils.info_to_array(info3)
                path_out3_array = os.path.join(cfg.path_out, name +'_info_ref.npy')
                np.save(path_out3_array, info3)

                fout_inst = open(os.path.join(cfg.path_out, name +'_info_ref.txt'), 'w')
                fout_inst.write('    x         y         z   type  info class inst    \n')
                for i in range(info3_array.shape[0]):
                    if info3_array[i,6]!= 0 and info3_array[i,6]!= 4 and info3_array[i,6]!= 6 and info3_array[i,6]!= 7:
                        fout_inst.write('%f %f %f  %d     %d     %d     %d\n' % (info3_array[i,0], info3_array[i,1], info3_array[i,2], info3_array[i,6], info3_array[i,7], info3_array[i,8], info3_array[i,9]))
            # print info

            print(" ")
            print("INFO VALVES:")
            for valve in info_valves_list:
                valve.pop(-2)
                print(valve)
            print(" ")

            print("INFO VALVES2:")
            for valve in info_valves_list2:
                valve.pop(-2)
                print(valve)
            print(" ")

            print("INFO PIPES:")
            for pipe1 in info_pipes_list:
                pipe1.pop(0)
                print(pipe1)
            print(" ")

            print("INFO PIPES2")
            for pipe2 in info_pipes_list2:
                pipe2.pop(0)
                print(pipe2)
            print(" ")

            print("INFO CONNEXIONS:")
            for connexion in info_connexions_list:
                print(connexion)
            print(" ")

            print("INFO CONNEXIONS2:")
            for connexion in info_connexions_list2:
                print(connexion)
            print(" ")

            # PRINTS

            i = len(instances_ref_valve_list)

            if len(instances_ref_valve_list)>0:
                instances_ref_valve = np.vstack(instances_ref_valve_list)
            if len(instances_ref_pipe_list)>0:
                instances_ref_pipe = np.vstack(instances_ref_pipe_list)
                instances_ref_pipe[:,7] = instances_ref_pipe[:,7]+i

            if len(instances_ref_valve_list)>0 and len(instances_ref_pipe_list)>0:
                instances_ref = np.concatenate((instances_ref_valve, instances_ref_pipe), axis=0)
            elif len(instances_ref_valve_list)==0 and len(instances_ref_pipe_list)>0:
                instances_ref = instances_ref_pipe
            elif len(instances_ref_valve_list)>0 and len(instances_ref_pipe_list)==0:
                instances_ref = instances_ref_valve
            else:
                instances_ref = None

            fout_sub = open(os.path.join(cfg.path_out, name +'_pred_sub.obj'), 'w')
            fout_sub_col = open(os.path.join(cfg.path_out, name +'_pred_sub_col.obj'), 'w')
            for i in range(pred_sub.shape[0]):
                fout_sub.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], int(pred_sub[i,3]*255), int(pred_sub[i,4]*255), int(pred_sub[i,5]*255), pred_sub[i,6]))
            for i in range(pred_sub.shape[0]):
                color = cfg.cmap[int(pred_sub[i,6])]
                fout_sub_col.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], color[0], color[1], color[2], pred_sub[i,6]))

            if instances_ref is not None: # if instances were found
                fout_inst = open(os.path.join(cfg.path_out, name +'_pred_inst_ref.obj'), 'w')
                fout_inst_col = open(os.path.join(cfg.path_out, name +'_pred_inst_ref_col.obj'), 'w')
                for i in range(instances_ref.shape[0]):
                    fout_inst.write('v %f %f %f %d %d %d %d %d\n' % (instances_ref[i,0], instances_ref[i,1], instances_ref[i,2], int(instances_ref[i,3]*255), int(instances_ref[i,4]*255), int(instances_ref[i,5]*255), instances_ref[i,6], instances_ref[i,7]))
                for i in range(instances_ref.shape[0]):
                    color = col_inst[instances_ref[i,7]]
                    fout_inst_col.write('v %f %f %f %d %d %d %d %d\n' % (instances_ref[i,0], instances_ref[i,1], instances_ref[i,2], color[0], color[1], color[2], instances_ref[i,6], instances_ref[i,7]))

            t10 = time.time()

            time_read = t1-t0
            time_inferference = t2-t1

            time_instaces_valve = t4-t3
            time_instaces_pipe = t6-t5
            time_instaces = time_instaces_valve + time_instaces_pipe

            time_info_valve = t5-t4
            time_info_pipe = t7-t6
            time_info = time_info_valve + time_info_pipe

            time_ref_valve = t9-t8
            time_ref_pipe = t8-t7
            time_ref = time_ref_valve + time_ref_pipe

            time_publish = t10-t9
            time_total = t10-t0
            
            # print time info
            print('INFO TIMES:')	
            print("")
            print('Pc processing took seconds. Split into:' + str(time_total))
            print('Reading -------- seconds ' + str(time_read) + '  percentaje: ' + str((time_read/time_total)*100))
            print('Inference ------ seconds ' + str(time_inferference) + '  percentaje: ' + str((time_inferference/time_total)*100))
            print('Instances ------ seconds ' + str(time_instaces) + '  percentaje: ' + str((time_instaces/time_total)*100))
            print(' - Valve - seconds ' + str(time_instaces_valve) + '  percentaje: ' + str((time_instaces_valve/time_total)*100))
            print(' - Pipe -- seconds ' + str(time_instaces_pipe) + '  percentaje: ' + str((time_instaces_pipe/time_total)*100))
            print('Info ----------- seconds ' + str(time_info) + '  percentaje: ' + str((time_info/time_total)*100))
            print(' - Valve - seconds ' + str(time_info_valve) + '  percentaje: ' + str((time_info_valve/time_total)*100))
            print(' - Pipe -- seconds ' + str(time_info_pipe) + '  percentaje: ' + str((time_info_pipe/time_total)*100))
            print('Refine --------- seconds ' + str(time_ref) + '  percentaje: ' + str((time_ref/time_total)*100))
            print(' - Valve - seconds ' + str(time_ref_valve) + '  percentaje: ' + str((time_ref_valve/time_total)*100))
            print(' - Pipe -- seconds ' + str(time_ref_pipe) + '  percentaje: ' + str((time_ref_pipe/time_total)*100))
            print('Publish -------- seconds ' + str(time_publish) + '  percentaje: ' + str((time_publish/time_total)*100))

            print(" ")
            print(" ")
            print("--------------------------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------------------------")
            print(" ")
            print(" ")



            T_read = T_read + time_read
            T_inferference = T_inferference + time_inferference

            T_instaces_valve = T_instaces_valve + time_instaces_valve
            T_instaces_pipe = T_instaces_pipe + time_instaces_pipe
            T_instaces = T_instaces_valve + T_instaces_pipe

            T_info_valve = T_info_valve + time_info_valve
            T_info_pipe = T_info_pipe + time_info_pipe
            T_info = T_info_valve + T_info_pipe

            T_ref_valve = T_ref_valve + time_ref_valve
            T_ref_pipe = T_ref_pipe + time_ref_pipe
            T_ref = T_ref_valve + T_ref_pipe

            T_publish =  T_publish + time_publish
            T_total = T_total + time_total

            n_pc = n_pc + 1


            # print time info mean
            print('INFO TIMES MEAN:')	
            print("")
            print('Pc processing took seconds. Split into:' + str((T_total)/n_pc))
            print('Reading -------- seconds ' + str((T_read)/n_pc) + '  percentaje: ' + str((T_read/T_total)*100))
            print('Inference ------ seconds ' + str((T_inferference)/n_pc) + '  percentaje: ' + str((T_inferference/T_total)*100))
            print('Instances ------ seconds ' + str((T_instaces)/n_pc) + '  percentaje: ' + str((T_instaces/T_total)*100))
            print(' - Valve - seconds ' + str((T_instaces_valve)/n_pc) + '  percentaje: ' + str((T_instaces_valve/T_total)*100))
            print(' - Pipe -- seconds ' + str((T_instaces_pipe)/n_pc) + '  percentaje: ' + str((T_instaces_pipe/T_total)*100))
            print('Info ----------- seconds ' + str((T_info)/n_pc) + '  percentaje: ' + str((T_info/T_total)*100))
            print(' - Valve - seconds ' + str((T_info_valve)/n_pc) + '  percentaje: ' + str((T_info_valve/T_total)*100))
            print(' - Pipe -- seconds ' + str((T_info_pipe)/n_pc) + '  percentaje: ' + str((T_info_pipe/T_total)*100))
            print('Refine --------- seconds ' + str((T_ref)/n_pc) + '  percentaje: ' + str((T_ref/T_total)*100))
            print(' - Valve - seconds ' + str((T_ref_valve)/n_pc) + '  percentaje: ' + str((T_ref_valve/T_total)*100))
            print(' - Pipe -- seconds ' + str((T_ref_pipe)/n_pc) + '  percentaje: ' + str((T_ref_pipe/T_total)*100))
            print('Publish -------- seconds ' + str((T_publish)/n_pc) + '  percentaje: ' + str((T_publish/T_total)*100))

            print(" ")
            print(" ")
            print("--------------------------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------------------------")
            print(" ")
            print(" ")

                    

        if online == 0:
            break
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--path_model', type=str, required=True, help='config file')
    parser.add_argument('--path_test', type=str, required=True, help='config file')
    parser.add_argument('--path_out', type=str, required=True, help='config file')
    args, opts = parser.parse_known_args()

    for file in os.listdir(os.path.join(args.path_model, 'checkpoint')):
        if "best" in file:
            break
    pretrained_path = os.path.join(args.path_model, 'checkpoint', file)


    cfg_file = os.listdir(os.path.join(args.path_model,'cfg/child'))

    path_cfg = os.path.join(args.path_model, 'cfg/child', cfg_file[0])

    cfg = EasyConfig()
    cfg.load(path_cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml

    cfg.visualize = True
    cfg.wandb.use_wandb = False
    cfg.path_test = args.path_test
    cfg.pretrained_path = pretrained_path
    cfg.mode =  "test"

    cfg.path_out = args.path_out
    os.makedirs(cfg.path_out, exist_ok=True)


    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = cfg.log_dir
    cfg.cfg_basename = path_cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg.cfg_path = path_cfg

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
