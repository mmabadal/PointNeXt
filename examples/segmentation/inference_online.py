"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
"""
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


def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True):
    ious_table = [f'{item:.2f}' for item in ious]
    header = ['method', 'OA', 'mACC', 'mIoU'] + cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.cfg_basename, f'{oa:.2f}', f'{macc:.2f}',
            f'{miou:.2f}'] + ious_table + [str(best_epoch), cfg.run_dir,
                                           wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


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
#     test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)


# @torch.no_grad()
# def test(model, data_list, cfg, num_votes=1):
#     """using a part of original point cloud as input to save memory.
#     Args:
#         model (_type_): _description_
#         test_loader (_type_): _description_
#         cfg (_type_): _description_
#         num_votes (int, optional): _description_. Defaults to 1.
#     Returns:
#         _type_: _description_
#     """


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

    # while 1
        # get folder list
        # for each file in list
            # infer file and delete
    while 1:

        data_list = generate_test_list(cfg)

        for file_path in data_list:

            cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
            all_logits = []
            coord, feat, idx_points, voxel_idx, reverse_idx_part, reverse_idx  = load_data(file_path, cfg)

            name = os.path.basename(file_path)
            os.remove(file_path)

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

            # output pred labels
            write_obj(coord, feat, os.path.join(cfg.path_out, f'input-{name}.obj'))
            # output pred labels
            write_obj(coord, pred, os.path.join(cfg.path_out, f'{cfg.cfg_basename}-{name}.obj'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--path_model', type=str, required=True, help='config file')
    parser.add_argument('--path_test', type=str, required=True, help='config file')
    parser.add_argument('--path_out', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
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
    #cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
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
