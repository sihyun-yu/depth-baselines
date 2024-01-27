"""Dataloader class for training monocular videos."""
import math
import pickle as pkl
import os
import random
import glob
import multiprocessing

from torchvision.transforms.functional import resize, InterpolationMode
import cv2
import imageio.v3 as iio
import numpy as np
import pycolmap
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import natsort

from plyfile import PlyData, PlyElement
from PIL import Image
from torchvision import transforms as T
import scipy.io
import distributed_utils as du

class GmuKitDataset(Dataset):
    """
    This class loads data from monocular video.
    """
    def __init__(self, args, mode, scenes, random_crop=True, **kwargs):
        assert len(scenes) == 1
        self.folder_path = os.path.join(args.rootdir, "data/gmu-kit")
        self.args = args
        self.mode = mode
        
        self.scene = scene = scenes[0]
        
        scales = {}
        scales[1] = -6.238
        scales[2] = -2.6919
        scales[3] = -6.9887
        scales[4] = -6.1620
        scales[5] = -6.1817
        scales[6] = -7.7468
        scales[7] = -5.8562
        scales[8] = -6.1502
        scales[9] = -5.9528
        
        kinectParams=  {}
        kinectParams["fx_rgb"] = 1.0477637710998533e+03
        kinectParams["fy_rgb"] = 1.0511749325842486e+03
        kinectParams["cx_rgb"] = 9.5926120509632392e+02
        kinectParams["cy_rgb"] = 5.2911546499433564e+02
            
        self.z_scale = scales[int(scene)]
        self.kinectParams = kinectParams

        self.scene_info = scipy.io.loadmat(os.path.join(self.folder_path, f'gmu-kitchens_info/scene_pose_info/scene_{scene}_reconstruct_info_frame_sort.mat'))["frames"].reshape(-1)
        
        self.scene_path = os.path.join(self.folder_path, f"gmu_scene_00{scene}")
        du.print_if_master(f'scene_path: {self.scene_path}')
        
        self.image_paths = natsort.natsorted(glob.glob(os.path.join(self.scene_path, "Images/*")))
        self.image_filenames = self.image_paths 
        self.depth_paths = natsort.natsorted(glob.glob(os.path.join(self.scene_path, "Depths/*")))
        
        # Use kinect intrinsic as color image intrinsic
        kinect_intrinsic = [ 1.0477637710998533e+03, 0., 9.5926120509632392e+02, 0., 1.0511749325842486e+03, 5.2911546499433564e+02, 0., 0., 1. ]
        self.intrinsic_matrix = np.array(kinect_intrinsic).reshape(3, 3)
        
        #TODO: hard-coded, Set resolution 
        self.resize_factor = scale_w = scale_h = 0.5
        self.intrinsic_matrix[0] *= scale_w
        self.intrinsic_matrix[1] *= scale_h
        self.image_height = 540
        self.image_width = 960
        
        world_to_camera_matrix_list = []
        camera_to_world_matrix_list = []
        image_filenames = []
        depths = []
        intrinsic_matrix_all = []
        k1s = []
        k2s = []
        
        cur_id = -1
        if os.path.exists(f'./cache/cache_gmukit_{scene}.pkl'):
            du.print_if_master(f'Start reading cache ./cache/cache_gmukit_{scene}.pkl...')
            with open(f'./cache/cache_gmukit_{scene}.pkl', 'rb') as f:
                (world_to_camera_matrix_list, camera_to_world_matrix_list, intrinsic_matrix_all, k1s, k2s, depths) = pkl.load(f)
        
        else:
            points_xyz = self.load_init_points()
            self.near_depth = np.percentile(points_xyz[..., -1], 0.5)
            self.far_depth = np.percentile(points_xyz[..., -1], 99.5)
        
            du.print_if_master(f'Start processing scannet dataset...')
            for i in tqdm(range(len(self.image_paths))):
                
                depth_path = self.depth_paths[i]
                rgb_path = self.image_paths[i]
                
                depth_num = int(depth_path.split("depth_")[-1].split(".png")[0])
                rgb_num = int(rgb_path.split("rgb_")[-1].split(".png")[0])
                scene_idx = rgb_num
                
                focal, k1, k2 = self.scene_info[scene_idx][1].reshape(3) # 3, 3
                focal = focal * self.resize_factor
                intrinsic_matrix = [focal, 0., self.image_width/2, 0., focal, self.image_height/2, 0., 0., 1. ]
                intrinsic_matrix = np.array(intrinsic_matrix).reshape(3, 3)
                intrinsic_matrix_all.append(intrinsic_matrix)
                
                k1s.append(k1)
                k2s.append(k2)
                
                # Convert OpenGL to OpenCV
                transform_matrix = np.array([[1., 0, 0, 0],
                                 [0, -1., 0, 0],  # Invert Y-axis
                                 [0, 0, -1., 0], # Invert Z-axis
                                 [0, 0, 0, 1.]]) 
                Rw2c = self.scene_info[scene_idx][2] # 3, 3
                Tw2c = self.scene_info[scene_idx][3].reshape(-1, 1) # 3, 1
                img_name = int(self.scene_info[scene_idx][4][0].split("rgb_")[-1].split(".png")[0])
                
                assert img_name == depth_num and img_name == rgb_num

                bottom = np.array([0, 0, 0, 1]).reshape(1, -1)
                w2c = np.concatenate([Rw2c, Tw2c], axis=1)
                w2c = np.concatenate([w2c, bottom], axis=0)
                
                # Apply transformation
                w2c = transform_matrix @ w2c
                c2w = np.linalg.inv(w2c)
            
                depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
                depth_torch = torch.Tensor(depth).unsqueeze(0)
                depth = resize(depth_torch, [self.image_height, self.image_width], antialias=True).permute(1, 2, 0).squeeze(-1).numpy()
                depths.append(depth)
                
                camera_to_world_matrix_list.append(c2w[:3, :])
                world_to_camera_matrix_list.append(w2c[:3, :])

            depths = np.stack(depths, axis=0)
            
            if du.is_master_proc():
                with open(f'./cache/cache_gmukit_{scene}.pkl', 'wb') as f:
                    pkl.dump([world_to_camera_matrix_list, camera_to_world_matrix_list, intrinsic_matrix_all, k1s, k2s, depths], f)
        
        
        self.depths = depths
        self.world_to_camera_matrix_list = world_to_camera_matrix_list
        self.camera_to_world_matrix_list = camera_to_world_matrix_list
        self.intrinsic_matrix_all = intrinsic_matrix_all
        self.k1s = k1s
        self.k2s = k2s
        self.num_frames = len(self.image_filenames)
        
        ''' Set cur z_vals '''
        if os.path.exists(f'./cache/cache_gmukit_{scene}_zvals.pkl'):
            du.print_if_master(f'Start reading cache ./cache/cache_gmukit_{scene}_zvals.pkl...')
            with open(f'./cache/cache_gmukit_{scene}_zvals.pkl', 'rb') as f:
                self.near_depth, self.far_depth = pkl.load(f)
        else:
        
            if du.is_master_proc():
                with open(f'./cache/cache_gmukit_{scene}_zvals.pkl', 'wb') as f:
                    pkl.dump([self.near_depth, self.far_depth], f)
        
        # du.print_if_master(f'near_depth_before_calib: {np.min(points_xyz[..., -1])}, far_depth_before_calib: {np.max(points_xyz[..., -1])}')
        du.print_if_master(f'near_depth_final: {self.near_depth}, far_depth_final: {self.far_depth}')
        du.print_if_master(f'depth file near_depth: {self.depths.min()}, far_depth: {self.depths.max()}')
        du.print_if_master(f'total #: {len(image_filenames)}, height: {self.image_height}, width: {self.image_width}')
        du.print_if_master(self.num_frames, self.depths.shape[0])
        
        print(self.num_frames, self.depths.shape[0])
        assert self.num_frames == self.depths.shape[0]

        self.manager = multiprocessing.Manager()
        self.rgb_dict = self.manager.dict()
        self.rgb_mask_dict = self.manager.dict()
        self.depth_dict = self.manager.dict()
        
        self.exclude_from_src = []
        
    def depth2world(self, depth, cam):

        # Convert and scale depth
        depth = depth.astype(float) / 1000
        col_depth = np.zeros_like(depth)
        valid_depth_indices = depth != 0
        col_depth[valid_depth_indices] = depth[valid_depth_indices] * self.z_scale

        # Project pixel coordinates in the local frame to 3D using scaled depth
        center = [self.kinectParams['cx_rgb'], self.kinectParams['cy_rgb']]
        imh, imw = col_depth.shape
        xgrid, ygrid = np.meshgrid(np.arange(imw) - center[0], np.arange(imh) - center[1])
        pcloud = np.zeros((imh, imw, 3))
        pcloud[:, :, 0] = -xgrid * col_depth / self.kinectParams['fx_rgb']
        pcloud[:, :, 1] = ygrid * col_depth / self.kinectParams['fy_rgb']
        pcloud[:, :, 2] = col_depth

        # Transform local point cloud to world coordinates
        Rw2c = cam['Rw2c']
        Tw2c = cam['Tw2c']
        Rc2w = np.linalg.inv(Rw2c)
        Tc2w = -Rc2w @ Tw2c

        pcl = pcloud.reshape(-1, 3)
        valid = pcl[:, 2] != 0
        pcl = pcl[valid]
        
        w_pcl = (Rc2w @ pcl.T).T + Tc2w.reshape(-1, 3)

        return w_pcl
    
    
    def load_init_points(self):
        points_xyz = []
    
        for idx in range(len(self.depth_paths)):
            
            depth_path = self.depth_paths[idx]
            rgb_path = self.image_paths[idx]
            
            depth = cv2.imread(depth_path, -1).astype(np.float32)
            
            depth_num = int(depth_path.split("depth_")[-1].split(".png")[0])
            rgb_num = int(rgb_path.split("rgb_")[-1].split(".png")[0])
            scene_idx = rgb_num
            Rw2c = self.scene_info[scene_idx][2] # 3, 3
            Tw2c = self.scene_info[scene_idx][3].reshape(-1, 1) # 3, 1
            img_name = int(self.scene_info[scene_idx][4][0].split("rgb_")[-1].split(".png")[0])
            
            assert img_name == depth_num and img_name == rgb_num

            bottom = np.array([0, 0, 0, 1]).reshape(1, -1)
            w2c = np.concatenate([Rw2c, Tw2c], axis=1)
            w2c = np.concatenate([w2c, bottom], axis=0)

            cam = {}
            cam["Rw2c"] = Rw2c
            cam["Tw2c"] = Tw2c

            w_pcl = self.depth2world(depth, cam)
            
            points_xyz.append(w_pcl)
    
        points_xyz = np.concatenate(points_xyz, 0)
    
        return points_xyz
    
    
    
    def __len__(self):
        return self.num_frames

    def set_exclude_from_src(self, exclude_from_src):
        self.exclude_from_src = exclude_from_src

    def set_epoch(self, current_epoch):
        self.max_step = min(3, (current_epoch // self.args.init_decay_epoch) + 1)
        du.print_if_master(f"New max_step: {self.max_step}")

    def __getitem__(self, idx):
        idx, epoch = idx % len(self), idx // len(self)
        rng = np.random.default_rng(idx + epoch*len(self))
        
        if idx in self.rgb_dict:
            query_rgb = self.rgb_dict[idx]
            query_rgb_mask = self.rgb_mask_dict[idx]
            
        else:
            rgb_array = iio.imread(self.image_filenames[idx]).astype(np.float32) 
            
            # Undistortion
            rgb_array = cv2.imread(self.image_filenames[idx])
            rgb_array = cv2.undistort(rgb_array, self.intrinsic_matrix_all[idx], np.array([self.k1s[idx], self.k2s[idx], 0, 0]))
            rgb_array = (cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)).astype(np.float32) 

            rgb_mask = rgb_array.sum(-1) > 0
            
            rgb_torch = torch.from_numpy(rgb_array / 255.0).permute(2, 0, 1)
            query_rgb = resize(rgb_torch, [self.image_height, self.image_width], antialias=True).permute(1, 2, 0).numpy()
            
            rgb_mask_torch = torch.from_numpy(rgb_mask).unsqueeze(0)
            query_rgb_mask = resize(rgb_mask_torch, [self.image_height, self.image_width], InterpolationMode.NEAREST).squeeze(0).numpy()
            
            self.rgb_dict[idx] = query_rgb
            self.rgb_mask_dict[idx] = query_rgb_mask
        
        query_c2w_matrix = self.camera_to_world_matrix_list[idx]
        query_w2c_matrix = self.world_to_camera_matrix_list[idx]

        (H, W) = img_size = query_rgb.shape[:2]
        intrinsic = self.intrinsic_matrix_all[idx]
        query_camera = np.concatenate(
            (list(img_size), intrinsic.flatten(), query_c2w_matrix.flatten(), query_w2c_matrix.flatten())
        )
        query_depth = self.depths[idx]
        assert query_depth.shape[0] == H and query_depth.shape[1] == W
        
        num_source_views = self.args.num_source_views
        num_candidates = num_source_views * 2 if self.mode == 'train' else num_source_views
        offset = 1
        index_list = []
        while len(index_list) < num_candidates:
            left = idx - offset
            right = idx + offset
            
            if (left >= 0) and (left not in self.exclude_from_src):
                index_list.append(left)
            
            if len(index_list) == num_candidates:
                break
            
            if (right < self.num_frames) and (right not in self.exclude_from_src):
                index_list.append(right)
            
            offset += 1
        src_ids = rng.choice(index_list, size=num_source_views, replace=False)
        src_ids = sorted(src_ids)
        
        src_rgbs = []
        src_rgb_masks = []
        src_cameras = []
        
        for src_id in src_ids:
            if (src_id in self.rgb_dict):
                src_rgb = self.rgb_dict[src_id]
                src_rgb_mask = self.rgb_mask_dict[src_id]
                
            else:
                rgb_array = iio.imread(self.image_filenames[src_id]).astype(np.float32) 
                
                # Undistortion
                rgb_array = cv2.imread(self.image_filenames[src_id])
                rgb_array = cv2.undistort(rgb_array, self.intrinsic_matrix_all[src_id], np.array([self.k1s[src_id], self.k2s[src_id], 0, 0]))
                rgb_array = (cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)).astype(np.float32) 

                rgb_mask = rgb_array.sum(-1) > 0
                
                rgb_torch = torch.from_numpy(rgb_array / 255.0).permute(2, 0, 1)
                src_rgb = resize(rgb_torch, [self.image_height, self.image_width], antialias=True).permute(1, 2, 0).numpy()
                
                rgb_mask_torch = torch.from_numpy(rgb_mask).unsqueeze(0)
                src_rgb_mask = resize(rgb_mask_torch, [self.image_height, self.image_width], InterpolationMode.NEAREST).squeeze(0).numpy()
                
                self.rgb_dict[src_id] = src_rgb
                self.rgb_mask_dict[src_id] = src_rgb_mask
                 
            src_rgbs.append(src_rgb)
            src_rgb_masks.append(src_rgb_mask)
            
            src_c2w_matrix = self.camera_to_world_matrix_list[src_id]
            src_w2c_matrix = self.world_to_camera_matrix_list[src_id]
            
            src_cameras.append(np.concatenate((list(img_size), intrinsic.flatten(), src_c2w_matrix.flatten(), src_w2c_matrix.flatten())))
        
        src_rgbs = np.stack(src_rgbs, axis=0)
        src_rgb_masks = np.stack(src_rgb_masks, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        
        depth_range = torch.tensor(
            [self.near_depth * 0.9, self.far_depth * 1.5]
        ).float()
        
        ret = {'idx': idx,
                'data_name': 'scannet',
                'query_id': idx,
               'query_camera': torch.from_numpy(query_camera).float(),
               'query_rgb': torch.from_numpy(query_rgb).float(),
               'query_rgb_mask': torch.from_numpy(query_rgb_mask),
               'query_depth': torch.from_numpy(query_depth).float(),
               'depth_range': depth_range,
               'src_ids': torch.from_numpy(np.array(src_ids)),
               'src_cameras': torch.from_numpy(src_cameras).float(),
               'src_rgbs': torch.from_numpy(src_rgbs[..., :3]).float(),
               'src_rgb_masks': torch.from_numpy(src_rgb_masks),
            }

        return ret