"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
"""

import struct
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

        self.name = name

        self.init = False
        self.new_pc = False

        path_model = "/home/miguel/PointNeXt/log/pipes/s-001-sub6k"
        path_out = "/home/miguel/PointNeXt/log/pipes/s-001-sub6k/visualization_ros"

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


        name_out = str(header.stamp)

        pc_np[:, 2] *= -1  # flip Z axis        # //PARAM

        all_logits = []
        coord, feat, idx_points, voxel_idx, reverse_idx_part, reverse_idx  = self.process_data(pc_np) 

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

            # output pred labels
            write_obj(coord, feat, os.path.join(self.cfg.path_out, f'{name_out}.obj'))
            # output pred labels
            write_obj(coord, pred, os.path.join(self.cfg.path_out, f'{name_out}_pred.obj'))
            
            pred_np = np.hstack((coord,pred))
            # publishers
            pc_base = self.array2pc(header, pc_np)   # TODO
            pc_seg = self.array2pc(header, pred_np)
            self.pub_pc_base.publish(pc_base)
            self.pub_pc_seg.publish(pc_seg)


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


if __name__ == "__main__":
    try:
        rospy.init_node('seg_pc')
        Pointcloud_Seg(rospy.get_name())
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
