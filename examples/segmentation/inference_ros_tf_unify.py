"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
"""
import math
import copy
import tf
import project_inst
import map_utils
import open3d as o3d
import expand_polygon
import get_info
import get_instances
import conversion_utils
from natsort import natsorted
import time
import struct
import map_utils
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

from openpoints.dataset.vis3d import write_obj


import rospy
import ctypes
from sensor_msgs.msg import PointField
import message_filters
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

warnings.simplefilter(action='ignore', category=FutureWarning)

class Pointcloud_Seg:


    def __init__(self, name):


        self.col_inst = {
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
        self.rad_p = 0.04               # max distance for pipe growing                             //PARAM
        self.rad_v = 0.04               # max distance for valve growing                            //PARAM
        self.dim_p = 3                  # compute 2D (2) or 3D (3) distance for pipe growing        //PARAM
        self.dim_v = 2                  # compute 2D (2) or 3D (3) distance for valve growing       //PARAM
        self.min_p_p = 50               # minimum number of points to consider a blob as a pipe     //PARAM
        self.min_p_v = 30 # 40 80 140   # minimum number of points to consider a blob as a valve    //PARAM

        # listener
        self.listener = tf.TransformListener()
        
        # inits info map
        self.info_map_key = False
        self.info_pipes_list_map = list()
        self.info_connexions_list_map = list()
        self.info_valves_list_map = list()
        self.instances_ref_pipe_list_map = list()
        self.info_map = [self.info_pipes_list_map, self.info_connexions_list_map, self.info_valves_list_map, self.instances_ref_pipe_list_map]
        self.count = 0
        self.count_target = 5
        self.count_thr = 2

        # get valve matching targets
        targets_path = "/home/miguel/PointNeXt/valve_targets"
        self.targets_list = list()
        for file_name in natsorted(os.listdir(targets_path)):
            target_path = os.path.join(targets_path, file_name)
            target = get_info.read_ply(target_path, "model")
            xyz_central = np.mean(target, axis=0)[0:3]
            target[:, 0:3] -= xyz_central  
            target[:, 2] *= -1                                                  # flip Z axis
            target_o3d = o3d.geometry.PointCloud()
            target_o3d.points = o3d.utility.Vector3dVector(target[:,0:3])
            target_o3d.colors = o3d.utility.Vector3dVector(target[:,3:6])
            target_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))  # //PARAM
            target_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))                    # //PARAM
            self.targets_list.append(target_o3d)

        self.n_pc = 0

        self.name = name

        self.init = False
        self.new_pc = False

        path_model = "/home/miguel/PointNeXt/log/pipes/s-001-sub6k"
        path_out = "/home/miguel/PointNeXt/log/pipes/s-001-sub6k/visualization_ros_info"

        for file in os.listdir(os.path.join(path_model, 'checkpoint')):
            if "best" in file:
                break
        pretrained_path = os.path.join(path_model, 'checkpoint', file)


        cfg_file = os.listdir(os.path.join(path_model,'cfg/child'))

        path_cfg = os.path.join(path_model, 'cfg/child', cfg_file[0])

        cfg = EasyConfig()
        cfg.load(path_cfg, recursive=True)

        cfg.visualize = True
        cfg.wandb.use_wandb = False
        cfg.pretrained_path = pretrained_path
        cfg.mode =  "test"

        cfg.path_out = path_out
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

        self.cfg = cfg

        self.gpu = 0
        
        # set subscribers
        # pc_sub = message_filters.Subscriber('/stereo_down/scaled_x2/points2_filtered', PointCloud2)     # //PARAM
        pc_sub = message_filters.Subscriber('/stereo_down/scaled_x2/points2', PointCloud2)             # //PARAM
        pc_sub.registerCallback(self.cb_pc)

        # Set class image publishers
        self.pub_pc_base = rospy.Publisher("/stereo_down/scaled_x2/points2_base", PointCloud2, queue_size=4)
        self.pub_pc_seg = rospy.Publisher("/stereo_down/scaled_x2/points2_seg", PointCloud2, queue_size=4)
        self.pub_pc_inst = rospy.Publisher("/stereo_down/scaled_x2/points2_inst", PointCloud2, queue_size=4)
        self.pub_pc_info = rospy.Publisher("/stereo_down/scaled_x2/points2_info", PointCloud2, queue_size=4)
        self.pub_pc_info_world = rospy.Publisher("/stereo_down/scaled_x2/points2_info_world", PointCloud2, queue_size=4)
        self.pub_pc_info_map = rospy.Publisher("/stereo_down/scaled_x2/points2_info_map", PointCloud2, queue_size=4)

        # Set segmentation timer
        self.fps = 1.0                # target fps        //PARAM
        self.period = 1.0/self.fps    # target period     //PARAM
        rospy.Timer(rospy.Duration(self.period), self.run)


    def cb_pc(self, pc):

        self.pc = pc
        self.new_pc = True


    def set_model(self):

        if self.cfg.distributed:
            if self.cfg.mp:
                self.cfg.rank = self.gpu
            dist.init_process_group(backend=self.cfg.dist_backend,
                                    init_method=self.cfg.dist_url,
                                    world_size=self.cfg.world_size,
                                    rank=self.cfg.rank)
            dist.barrier()

        # logger
        setup_logger_dist(self.cfg.log_path, self.cfg.rank, name=self.cfg.dataset.common.NAME)
        if self.cfg.rank == 0:
            Wandb.launch(self.cfg, self.cfg.wandb.use_wandb)
            writer = SummaryWriter(log_dir=self.cfg.run_dir) if self.cfg.is_training else None
        else:
            writer = None
        set_random_seed(self.cfg.seed + self.cfg.rank, deterministic=self.cfg.deterministic)
        torch.backends.cudnn.enabled = True

        if self.cfg.model.get('in_channels', None) is None:
            self.cfg.model.in_channels = self.cfg.model.encoder_args.in_channels
        self.model = build_model_from_cfg(self.cfg.model).to(self.cfg.rank)
        model_size = cal_model_parm_nums(self.model)


        if self.cfg.sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('Using Synchronized BatchNorm ...')
        if self.cfg.distributed:
            torch.cuda.set_device(self.gpu)
            self.model = nn.parallel.DistributedDataParallel(self.model.cuda(), device_ids=[self.cfg.rank], output_device=self.cfg.rank)
            logging.info('Using Distributed Data parallel ...')

        self.cfg.cmap = np.array(self.cfg.cmap)

        # optionally resume from a checkpoint
        model_module = self.model.module if hasattr(self.model, 'module') else self.model
        best_epoch, best_val = load_checkpoint(self.model, pretrained_path=self.cfg.pretrained_path)


        self.model.eval()  # set model to eval mode
        all_cm = ConfusionMatrix(num_classes=self.cfg.num_classes, ignore_index=self.cfg.ignore_index)
        set_random_seed(0)
        self.cfg.visualize = self.cfg.get('visualize', False)
        if self.cfg.visualize:
            from openpoints.dataset.vis3d import write_obj
            self.cfg.cmap = self.cfg.cmap.astype(np.float32) / 255.

        trans_split = 'val' if self.cfg.datatransforms.get('test', None) is None else 'test'
        self.pipe_transform = build_transforms_from_cfg(trans_split, self.cfg.datatransforms)

        dataset_name = self.cfg.dataset.common.NAME.lower()

        self.cfg.save_path = self.cfg.get('save_path', f'results/{self.cfg.task_name}/{self.cfg.dataset.test.split}/{self.cfg.cfg_basename}')
        os.makedirs(self.cfg.save_path, exist_ok=True)

        self.gravity_dim = self.cfg.datatransforms.kwargs.gravity_dim
        self.nearest_neighbor = self.cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor'


    def run(self,_):

        rospy.loginfo('[%s]: Running', self.name)	
        t0 = rospy.Time.now()

        # New pc available
        if not self.new_pc:
            rospy.loginfo('[%s]: No new pointcloud', self.name)	
            return
        self.new_pc = False

        # Retrieve image
        try:
            pc = self.pc
            header = self.pc.header
            if not self.init:
                rospy.loginfo('[%s]: Start pc segmentation', self.name)	
        except:
            rospy.logwarn('[%s]: There is no input pc to run the segmentation', self.name)
            return

        # Set model
        if not self.init:
            self.set_model()
            self.init = True
            return

        pc_np = self.pc2array(pc)
        if pc_np.shape[0] < 2000:               # return if points < thr   //PARAM
            rospy.loginfo('[%s]: Not enough input points', self.name)
            return

        left_frame_id = "turbot/stereo_down/left_optical"
        world_frame_id = "world_ned" 
        left2worldned = self.get_transform(world_frame_id, left_frame_id, header.stamp)

        name_out = str(header.stamp)

        pc_np[:, 2] *= -1  # flip Z axis        # //PARAM

        all_logits = []
        coord, feat, idx_points, voxel_idx, reverse_idx_part, reverse_idx  = self.process_data(pc_np) 
        
        t1 = rospy.Time.now()

        len_part = len(idx_points)
        self.nearest_neighbor = len_part == 1
        print("Inference on: " + name_out)

        error = False

        for idx_subcloud in range(len(idx_points)):
            if not (self.nearest_neighbor and idx_subcloud>0):
                idx_part = idx_points[idx_subcloud]
                coord_part = coord[idx_part]
                coord_part -= coord_part.min(0)

                feat_part =  feat[idx_part] if feat is not None else None
                data = {'pos': coord_part}
                if feat_part is not None:
                    data['x'] = feat_part
                if self.pipe_transform is not None:
                    data = self.pipe_transform(data)
                if 'heights' in self.cfg.feature_keys and 'heights' not in data.keys():
                    data['heights'] = torch.from_numpy(coord_part[:, self.gravity_dim:self.gravity_dim + 1].astype(np.float32)).unsqueeze(0)
                if not self.cfg.dataset.common.get('variable', False):
                    if 'x' in data.keys():
                        data['x'] = data['x'].unsqueeze(0)
                    data['pos'] = data['pos'].unsqueeze(0)
                else:
                    data['o'] = torch.IntTensor([len(coord)])
                    data['batch'] = torch.LongTensor([0] * len(coord))

                for key in data.keys():
                    data[key] = data[key].cuda(non_blocking=True)
                data['x'] = get_features_by_keys(data, self.cfg.feature_keys)



                try:
                    logits = self.model(data)
                except RuntimeError as e:
                    error = True
                    logging.warning(f"Failed to process point cloud: {e}")
                    break
                
            all_logits.append(logits)
        

        if error == False:
            all_logits = torch.cat(all_logits, dim=0)
            if not self.cfg.dataset.common.get('variable', False):
                all_logits = all_logits.transpose(1, 2).reshape(-1, self.cfg.num_classes)

            if not self.nearest_neighbor:
                # average merge overlapped multi voxels logits to original point set
                idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
                all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
            else:
                # interpolate logits by nearest neighbor
                all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]
            pred = all_logits.argmax(dim=1)

            pred = pred.cpu().numpy().squeeze()
            pred = self.cfg.cmap[pred, :]

            t2 = rospy.Time.now()

            # output pred labels
            write_obj(coord, feat, os.path.join(self.cfg.path_out, f'{name_out}.obj'))
            # output pred labels
            write_obj(coord, pred, os.path.join(self.cfg.path_out, f'{name_out}_pred.obj'))

            t3 = rospy.Time.now()


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
            instances_ref_valve_list, pred_sub_pipe_ref, stolen_list  = get_instances.get_instances(pred_sub_valve, self.dim_v, self.rad_v, self.min_p_v, ref=True, ref_data = pred_sub_pipe, ref_rad = 0.1)
            #instances_ref_valve_list, pred_sub_pipe_ref, stolen_list  = get_instances.get_instances_o3d(pred_sub_valve, self.dim_v, self.rad_v, self.min_p_v, ref=True, ref_data = pred_sub_pipe, ref_rad = 0.1)
            
            t4 = rospy.Time.now()

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

                info_valve = get_info.get_info(inst_o3d, self.targets_list, method="matching")        # get valve instance info list([fitness1, rotation1],[fitness2, rotation2], ...)  len = len(targets_list)
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
            t5 = rospy.Time.now()

            # get pipe instances
            instances_ref_pipe_list, _, _  = get_instances.get_instances(pred_sub_pipe_ref, self.dim_p, self.rad_p, self.min_p_p)
            #instances_ref_pipe_list, _, _  = get_instances.get_instances_o3d(pred_sub_pipe_ref, self.dim_p, self.rad_p, self.min_p_p)
            t6 = rospy.Time.now()

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
            t7 = rospy.Time.now()

            info_pipes_list_copy = copy.deepcopy(info_pipes_list) 
            info_connexions_list_copy = copy.deepcopy(info_connexions_list)
            info_pipes_list2, info_connexions_list2 = get_info.unify_chains(info_pipes_list_copy, info_connexions_list_copy)
            t8 = rospy.Time.now()

            info_valves_list_copy = copy.deepcopy(info_valves_list)
            info_valves_list2 = get_info.refine_valves(info_valves_list_copy, info_pipes_list2) 
            t9 = rospy.Time.now()

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
                path_out1 = os.path.join(self.cfg.path_out, name_out +'_info.ply')
                conversion_utils.info_to_ply(info1, path_out1)
                
            #path_out2 = os.path.join(self.cfg.path_out, name_out +'_info2.ply')
            #conversion_utils.info_to_ply(info2, path_out2)

            if empty3==False:

                self.count +=1

                info_array = conversion_utils.info_to_array(info3)
                pc_info = self.array2pc_info(header, info_array)
                self.pub_pc_info.publish(pc_info)

                path_out3 = os.path.join(self.cfg.path_out, name_out +'_info_ref.ply')
                conversion_utils.info_to_ply(info3, path_out3)

                info3_array = conversion_utils.info_to_array(info3)
                path_out3_array = os.path.join(self.cfg.path_out, name_out +'_info_ref.npy')
                np.save(path_out3_array, info3)

                fout_inst = open(os.path.join(self.cfg.path_out, name_out +'_info_ref.txt'), 'w')
                fout_inst.write('    x         y         z   type  info class inst    \n')
                for i in range(info3_array.shape[0]):
                    if info3_array[i,6]!= 0 and info3_array[i,6]!= 4 and info3_array[i,6]!= 6 and info3_array[i,6]!= 7:
                        fout_inst.write('%f %f %f  %d     %d     %d     %d\n' % (info3_array[i,0], info3_array[i,1], info3_array[i,2], info3_array[i,6], info3_array[i,7], info3_array[i,8], info3_array[i,9]))
            
                if isinstance(left2worldned,int) == False:
                    info_array_world = info_array.copy()
                    for i in range(info_array.shape[0]):
                        xyz = np.array([[info_array[i,0]],
                                        [info_array[i,1]],
                                        [info_array[i,2]],
                                        [1]])
                        xyz_trans_rot = np.matmul(left2worldned, xyz)
                        info_array_world[i,0:3] = [xyz_trans_rot[0], xyz_trans_rot[1], xyz_trans_rot[2]]

                    header.frame_id = world_frame_id
                    pc_info_world = self.array2pc_info(header, info_array_world)
                    self.pub_pc_info_world.publish(pc_info_world)

                    out1 = False
                    if out1 == True:         
                        path_out_world_info = os.path.join(self.cfg.path_out, name_out +"_info_world.ply")
                        info_pipes_world_list, info_connexions_world_list, info_valves_world_list, info_inst_pipe_world_list = conversion_utils.array_to_info(info_array_world)
                        info_world = [info_pipes_world_list, info_connexions_world_list, info_valves_world_list, info_inst_pipe_world_list]
                        conversion_utils.info_to_ply(info_world, path_out_world_info)

                        pred_sub_world = pred_sub.copy()
                        for i in range(pred_sub.shape[0]):
                            xyz = np.array([[pred_sub[i,0]],
                                            [pred_sub[i,1]],
                                            [pred_sub[i,2]],
                                            [1]])
                            xyz_trans_rot = np.matmul(left2worldned, xyz)
                            pred_sub_world[i,0:3] = [xyz_trans_rot[0], xyz_trans_rot[1], xyz_trans_rot[2]]

                        path_out_world_base = os.path.join(self.cfg.path_out, name_out +"_base_world.obj")
                        path_out_world_pred = os.path.join(self.cfg.path_out, name_out +"_pred_world.obj")
                        fout_base = open(path_out_world_base, 'w')
                        fout_pred = open(path_out_world_pred, 'w')
                        for i in range(pred_sub_world.shape[0]):
                            fout_base.write('v %f %f %f %d %d %d\n' % (pred_sub_world[i,0], pred_sub_world[i,1], pred_sub_world[i,2], pred_sub_world[i,3], pred_sub_world[i,4], pred_sub_world[i,5]))
                        for i in range(pred_sub_world.shape[0]):
                            color = self.label2color[pred_sub_world[i,6]]
                            fout_pred.write('v %f %f %f %d %d %d\n' % (pred_sub_world[i,0], pred_sub_world[i,1], pred_sub_world[i,2], color[0], color[1], color[2]))

                    if self.info_map_key == True:

                        info_pipes_world_list, info_connexions_world_list, info_valves_world_list, info_inst_pipe_world_list = conversion_utils.array_to_info(info_array_world)
                        info_world = [info_pipes_world_list, info_connexions_world_list, info_valves_world_list, info_inst_pipe_world_list]

                        self.info_map = map_utils.get_info_map(self.info_map, info_world)

                        if self.count == self.count_target:
                            self.count = 0
                            self.info_map = map_utils.clean_map(self.info_map, self.count_thr)
                        
                        info_map_array = conversion_utils.info_to_array(self.info_map)
                        pc_info_map = self.array2pc_info(header, info_map_array)
                        self.pub_pc_info_map.publish(pc_info_map)

                        out2 = False
                        if out2 == True:
                            z = 1 # SAVE INFO MAP
                            path_out_world_info_np = os.path.join(self.cfg.path_out, name_out +"_info_world.npy") # save array of info3 to world used
                            np.save(path_out_world_info_np, info_array_world)

                    header.frame_id = "turbot/stereo_down/left_optical"            
                
            
            
            
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

            if instances_ref is None: # if instances were not found
                rospy.loginfo('[%s]: No instances found', self.name)	
                return

            if len(info_pipes_list2)>0 or len(info_valves_list2)>0:  # print here because instrances_ref is needed
                if isinstance(left2worldned,int) == False:
                    if out1 == True:
                        instances_ref_world = instances_ref.copy()
                        for i in range(instances_ref.shape[0]):
                            xyz = np.array([[instances_ref[i,0]],
                                            [instances_ref[i,1]],
                                            [instances_ref[i,2]],
                                            [1]])
                            xyz_trans_rot = np.matmul(left2worldned, xyz)
                            instances_ref_world[i,0:3] = [xyz_trans_rot[0], xyz_trans_rot[1], xyz_trans_rot[2]]

                        path_out_world_inst = os.path.join(self.cfg.path_out, name_out +"_inst_world.obj")
                        fout_pred = open(path_out_world_inst, 'w')
                        for i in range(instances_ref_world.shape[0]):
                            color = self.col_inst[instances_ref_world[i,7]]
                            fout_pred.write('v %f %f %f %d %d %d\n' % (instances_ref_world[i,0], instances_ref_world[i,1], instances_ref_world[i,2], color[0], color[1], color[2]))

            fout_sub = open(os.path.join(self.cfg.path_out, name_out +'_pred_sub.obj'), 'w')
            fout_sub_col = open(os.path.join(self.cfg.path_out, name_out +'_pred_sub_col.obj'), 'w')
            for i in range(pred_sub.shape[0]):
                fout_sub.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], int(pred_sub[i,3]*255), int(pred_sub[i,4]*255), int(pred_sub[i,5]*255), pred_sub[i,6]))
            for i in range(pred_sub.shape[0]):
                color = self.cfg.cmap[int(pred_sub[i,6])]
                fout_sub_col.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], color[0], color[1], color[2], pred_sub[i,6]))

            if instances_ref is not None: # if instances were found
                fout_inst = open(os.path.join(self.cfg.path_out, name_out +'_pred_inst_ref.obj'), 'w')
                fout_inst_col = open(os.path.join(self.cfg.path_out, name_out +'_pred_inst_ref_col.obj'), 'w')
                for i in range(instances_ref.shape[0]):
                    fout_inst.write('v %f %f %f %d %d %d %d %d\n' % (instances_ref[i,0], instances_ref[i,1], instances_ref[i,2], int(instances_ref[i,3]*255), int(instances_ref[i,4]*255), int(instances_ref[i,5]*255), instances_ref[i,6], instances_ref[i,7]))
                for i in range(instances_ref.shape[0]):
                    color = self.col_inst[instances_ref[i,7]]
                    fout_inst_col.write('v %f %f %f %d %d %d %d %d\n' % (instances_ref[i,0], instances_ref[i,1], instances_ref[i,2], color[0], color[1], color[2], instances_ref[i,6], instances_ref[i,7]))
            
            # publishers
            pred_np = np.hstack((coord,pred))
            pc_base = self.array2pc(header, pc_np) 
            pc_seg = self.array2pc(header, pred_np)
            pc_inst = self.array2pc(header, instances_ref)
            self.pub_pc_base.publish(pc_base)
            self.pub_pc_seg.publish(pc_seg)
            self.pub_pc_inst.publish(pc_inst)

            #publish info TODO + check all things published are ok TODO

            t10 = rospy.Time.now()

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


    def process_data(self, data):

        coord, feat = data[:, :3], data[:, 3:6]
        
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)

        idx_points = []
        voxel_idx, reverse_idx_part,reverse_idx_sort = None, None, None
        voxel_size = self.cfg.dataset.common.get('voxel_size', None)

        if voxel_size is not None:
            # idx_sort: original point indicies sorted by voxel NO.
            # voxel_idx: Voxel NO. for the sorted points
            idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
            if self.cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
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


    def pc2array(self, ros_pc):
        gen = pc2.read_points(ros_pc, skip_nans=True)   # ROS pointcloud into generator
        pc_np = np.array(list(gen))                     # generator to list to numpy

        if pc_np.size > 0:                              # if there are points

            if self.cfg.points != 0:                    # downsample
                if pc_np.shape[0] > self.cfg.points:
                    sub_idx = np.linspace(0, pc_np.shape[0]-1, self.cfg.points, dtype=int)
                    pc_np = pc_np[sub_idx]

            rgb_list = list()

            for rgb in pc_np[...,3]:
                # cast float32 to int so that bitwise operations are possible
                s = struct.pack('>f' ,rgb)
                i = struct.unpack('>l',s)[0]
                # get back the float value by the inverse operations
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000)>> 16
                g = (pack & 0x0000FF00)>> 8
                b = (pack & 0x000000FF)
                rgb_np = np.array([r,g,b])
                rgb_list.append(rgb_np)

            rgb = np.vstack(rgb_list)
            pc_np = np.delete(pc_np, 3, 1) 
            pc_np = np.concatenate((pc_np, rgb), axis=1)

        return pc_np


    def array2pc(self, header, array):

        fields =   [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgba', 12, PointField.UINT32, 1)]
        
        points = list()

        for i, p in enumerate(array):
            r = int(p[3])
            g = int(p[4])
            b = int(p[5])
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]

            p_rgb = [p[0], p[1], p[2], rgb]
            points.append(p_rgb)

        pc = pc2.create_cloud(header, fields, points)
        return pc


    def array2pc_info(self, header, array):

        fields =   [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgba', 12, PointField.UINT32, 1),
                    PointField('t', 16, PointField.FLOAT32, 1),
                    PointField('info', 20, PointField.FLOAT32, 1),
                    PointField('c', 24, PointField.FLOAT32, 1),
                    PointField('inst', 28, PointField.FLOAT32, 1)]
        
        points = list()

        for i, p in enumerate(array):
            r = int(p[3])
            g = int(p[4])
            b = int(p[5])
            a = 255

            if p[6] == 6:                   # if its type instance data, make it transparent, information still there
                a = 0

            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]

            p_rgb = [p[0], p[1], p[2], rgb, p[6], p[7], p[8], p[9]]
            points.append(p_rgb)

        pc = pc2.create_cloud(header, fields, points)
        return pc


    def get_transform(self, parent, child, stamp):
        try:
            rospy.logwarn("[%s]: waiting transform from %s to %s", self.name, parent, child)
            #self.listener.waitForTransform(parent, child, rospy.Time(), rospy.Duration(0.1))
            (trans, rot) = self.listener.lookupTransform(parent, child, stamp)
            rospy.loginfo("[%s]: transform for %s found", self.name, child)
            transform = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf.Exception):
            rospy.logerr('[%s]: define %s transform!', self.name, child)
            transform = 0
        return transform


if __name__ == "__main__":
    try:
        rospy.init_node('seg_pc')
        Pointcloud_Seg(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
