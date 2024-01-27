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
import pickle

class ScanNetDataset(Dataset):
    """
    This class loads data from monocular video.
    """
    def __init__(self, args, mode, scenes, random_crop=True, **kwargs):
        assert len(scenes) == 1
        self.folder_path = os.path.join(args.rootdir, "data/scans")
        self.args = args
        self.mode = mode
                
        self.scene = scene = scenes[0]
        
        self.scene_path = os.path.join(self.folder_path, scene)
        print(f'scene_path: {self.scene_path}')
        
        colordir = os.path.join(self.scene_path, "exported/color")
        self.image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f))]
        self.image_paths = [os.path.join(self.scene_path, "exported/color/{}.jpg".format(i)) for i in range(len(self.image_paths))]
        self.all_id_list, self.train_id_list, self.test_id_list = self.filter_valid_id()
        
        print("all_id_list: ",len(self.all_id_list))
        
        # self.intrinsic_matrix = np.loadtxt(os.path.join(self.scene_path, "exported/intrinsic/intrinsic_color.txt")).astype(np.float32)[:3,:3]
        
        # Use depth intrinsic as color image intrinsic
        self.depth_intrinsic = np.loadtxt(
            os.path.join(self.scene_path, "exported/intrinsic/intrinsic_depth.txt")).astype(np.float32)[:3, :3]
        intrinsic_matrix = self.depth_intrinsic

        self.intrinsic_matrix = np.eye(4)
        self.intrinsic_matrix[:3, :3] = intrinsic_matrix
        
        print(self.intrinsic_matrix)

        
        # image shape is hard-coded following depth size
        self.image_width = 640 
        self.image_height = 480 
        
        world_to_camera_matrix_list = []
        camera_to_world_matrix_list = []
        image_filenames = []
        depths = []
        
        cur_id = -1
        if os.path.exists(f'./cache/cache_scannet_{scene}.pkl'):
            print(f'Start reading cache ./cache/cache_scannet_{scene}.pkl...')
            with open(f'./cache/cache_scannet_{scene}.pkl', 'rb') as f:
                (world_to_camera_matrix_list, camera_to_world_matrix_list, image_filenames, depths,
                 z_vals, near_depth, far_depth, points_xyz) = pkl.load(f)
        
        else:
            points_xyz = self.load_init_points()
            z_vals = points_xyz[..., -1]
            near_depth = max(np.min(z_vals), 1e-3)
            far_depth = np.max(z_vals)
        
            print(f'Start processing scannet dataset...')
            for i in tqdm(range(len(self.all_id_list))):
                
                id = self.all_id_list[i]
                
                assert id > cur_id
                cur_id = id
                
                image_path = os.path.join(self.scene_path, "exported/color/{}.jpg".format(id))
                image_filenames.append(image_path)
                
                depth_f = self.read_depth(os.path.join(self.scene_path, "exported/depth/{}.png".format(id)))
                depths.append(depth_f)
                
                c2w = np.loadtxt(os.path.join(self.scene_path, "exported/pose", "{}.txt".format(id))).astype(np.float32)
                w2c = np.linalg.inv(c2w)
                
                camera_to_world_matrix_list.append(c2w)
                world_to_camera_matrix_list.append(w2c)

            depths = np.stack(depths, axis=0)
            
            if args.local_rank == 0:
                with open(f'./cache/cache_scannet_{scene}.pkl', 'wb') as f:
                    pkl.dump([world_to_camera_matrix_list, camera_to_world_matrix_list, image_filenames, depths,
                 z_vals, near_depth, far_depth, points_xyz], f)
        
        
        self.depths = depths[:200]
        self.image_filenames = image_filenames[:200]
        self.world_to_camera_matrix_list = world_to_camera_matrix_list[:200]
        self.camera_to_world_matrix_list = camera_to_world_matrix_list[:200]
        self.num_frames = len(self.image_filenames)
        
        ''' Set cur z_vals '''
        if os.path.exists(f'./cache/cache_scannet_{scene}_zvals.pkl'):
            print(f'Start reading cache ./cache/cache_scannet_{scene}_zvals.pkl...')
            with open(f'./cache/cache_scannet_{scene}_zvals.pkl', 'rb') as f:
                self.near_depth, self.far_depth = pkl.load(f)
        else:
            points_xyzh = np.concatenate([points_xyz, np.ones_like((points_xyz[..., -1:]))], axis=-1)
            w2c_matrix = np.stack(self.world_to_camera_matrix_list, 0)
            
            projections = (w2c_matrix @ points_xyzh.transpose(-1, -2)).transpose(0, 2, 1)
            
            depth = projections[..., 2:]
            depth = np.clip(depth, near_depth, far_depth)
            
            self.near_depth = np.percentile(depth, 5)
            self.far_depth = np.percentile(depth, 95)
        
            if args.local_rank == 0:
                with open(f'./cache/cache_scannet_{scene}_zvals.pkl', 'wb') as f:
                    pkl.dump([self.near_depth, self.far_depth], f)
        
        print(f'near_depth_before_calib: {near_depth}, far_depth_before_calib: {far_depth}')
        print("points_z min: ", np.min(z_vals), "max: ", np.max(z_vals))
        print(f'near_depth_final: {self.near_depth}, far_depth_final: {self.far_depth}')
        print(f'depth file near_depth: {self.depths.min()}, far_depth: {self.depths.max()}')
        print(f'total #: {len(image_filenames)}, height: {self.image_height}, width: {self.image_width}')
        print(self.num_frames, self.depths.shape[0])
        
        assert self.num_frames == self.depths.shape[0]

        self.manager = multiprocessing.Manager()
        self.rgb_dict = self.manager.dict()
        self.rgb_mask_dict = self.manager.dict()
        self.depth_dict = self.manager.dict()
        
        self.exclude_from_src = []
    
    def filter_valid_id(self):
        
        # load
        with open(os.path.join(self.folder_path, 'scannet_split.pickle'), 'rb') as f:
            data = pickle.load(f)
            
        train_id = data[self.scene]["train"]
        test_id = data[self.scene]["test"]
        all_id = sorted(train_id + test_id)
        print(f"Train: {train_id}")
        print(f"Test: {test_id}")
        
        return all_id, train_id, test_id
    
    def parse_mesh(self):
        points_path = os.path.join(self.scene_path, "exported/pcd.ply")
        mesh_path = os.path.join(self.scene_path, self.scene + "_vh_clean.ply")
        plydata = PlyData.read(mesh_path)
        print("plydata 0", plydata.elements[0], plydata.elements[0].data["blue"].dtype)

        vertices = np.empty(len( plydata.elements[0].data["blue"]), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertices['x'] = plydata.elements[0].data["x"].astype('f4')
        vertices['y'] = plydata.elements[0].data["y"].astype('f4')
        vertices['z'] = plydata.elements[0].data["z"].astype('f4')
        vertices['red'] = plydata.elements[0].data["red"].astype('u1')
        vertices['green'] = plydata.elements[0].data["green"].astype('u1')
        vertices['blue'] = plydata.elements[0].data["blue"].astype('u1')

        # save as ply
        ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
        ply.write(points_path)
    
    def read_depth(self, filepath):
        depth_im = cv2.imread(filepath, -1).astype(np.float32)
        depth_im /= 1000
        # depth_im[depth_im > 8.0] = 0
        # depth_im[depth_im < 0.3] = 0
        return depth_im
     
    def load_init_points(self):
        points_path = os.path.join(self.scene_path, "exported/pcd.ply")
        
        if not os.path.exists(points_path):
            if not os.path.exists(points_path):
                self.parse_mesh()
        plydata = PlyData.read(points_path)
        # plydata (PlyProperty('x', 'double'), PlyProperty('y', 'double'), PlyProperty('z', 'double'), PlyProperty('nx', 'double'), PlyProperty('ny', 'double'), PlyProperty('nz', 'double'), PlyProperty('red', 'uchar'), PlyProperty('green', 'uchar'), PlyProperty('blue', 'uchar'))
        # x,y,z=torch.as_tensor(plydata.elements[0].data["x"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["y"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["z"].astype(np.float32), device="cuda", dtype=torch.float32)
        x = plydata.elements[0].data["x"].astype(np.float32)
        y = plydata.elements[0].data["y"].astype(np.float32)
        z = plydata.elements[0].data["z"].astype(np.float32)
        points_xyz = np.stack([x,y,z], axis=-1)
        # if self.opt.ranges[0] > -99.0:
        #     ranges = torch.as_tensor(self.opt.ranges, device=points_xyz.device, dtype=torch.float32)
        #     mask = torch.prod(torch.logical_and(points_xyz >= ranges[None, :3], points_xyz <= ranges[None, 3:]), dim=-1) > 0
        #     points_xyz = points_xyz[mask]
        # np.savetxt(os.path.join(self.data_dir, self.scan, "exported/pcd.txt"), points_xyz.cpu().numpy(), delimiter=";")

        return points_xyz
    
    
    
    def __len__(self):
        return self.num_frames

    def set_exclude_from_src(self, exclude_from_src):
        self.exclude_from_src = exclude_from_src

    def set_epoch(self, current_epoch):
        self.max_step = min(3, (current_epoch // self.args.init_decay_epoch) + 1)
        print(f"New max_step: {self.max_step}")

    def __getitem__(self, idx):
        idx, epoch = idx % len(self), idx // len(self)
        rng = np.random.default_rng(idx + epoch*len(self))
        
        if idx in self.rgb_dict:
            query_rgb = self.rgb_dict[idx]
            # query_rgb_mask = self.rgb_mask_dict[idx]
            
        else:
            rgb_array = iio.imread(self.image_filenames[idx]).astype(np.float32) 
            rgb_mask = rgb_array.sum(-1) > 0
            
            rgb_torch = torch.from_numpy(rgb_array / 255.0).permute(2, 0, 1)
            query_rgb = resize(rgb_torch, [self.image_height, self.image_width], antialias=True).permute(1, 2, 0).numpy()
            
            rgb_mask_torch = torch.from_numpy(rgb_mask).unsqueeze(0)
            # query_rgb_mask = resize(rgb_mask_torch, [self.image_height, self.image_width], InterpolationMode.NEAREST).squeeze(0).numpy()
            
            self.rgb_dict[idx] = query_rgb
            #self.rgb_mask_dict[idx] = query_rgb_mask
        
        query_rgb_path = self.image_filenames[idx]
        query_c2w_matrix = self.camera_to_world_matrix_list[idx]
        query_w2c_matrix = self.world_to_camera_matrix_list[idx]

        (H, W) = img_size = query_rgb.shape[:2]
        query_camera = np.concatenate(
            (list(img_size), self.intrinsic_matrix.flatten(), query_c2w_matrix.flatten())
        )
        query_depth = self.depths[idx]
        assert query_depth.shape[0] == H and query_depth.shape[1] == W
        
        num_source_views = self.args.num_source_views
        num_candidates = num_source_views * 3 if self.mode == 'train' else num_source_views
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
                # src_rgb_mask = self.rgb_mask_dict[src_id]
                
            else:
                rgb_array = iio.imread(self.image_filenames[src_id]).astype(np.float32) 
                rgb_mask = rgb_array.sum(-1) > 0
                
                rgb_torch = torch.from_numpy(rgb_array / 255.0).permute(2, 0, 1)
                src_rgb = resize(rgb_torch, [self.image_height, self.image_width], antialias=True).permute(1, 2, 0).numpy()
                
                rgb_mask_torch = torch.from_numpy(rgb_mask).unsqueeze(0)
                src_rgb_mask = resize(rgb_mask_torch, [self.image_height, self.image_width], InterpolationMode.NEAREST).squeeze(0).numpy()
                
                self.rgb_dict[src_id] = src_rgb
                # self.rgb_mask_dict[src_id] = src_rgb_mask
                 
            src_rgbs.append(src_rgb)
            #src_rgb_masks.append(src_rgb_mask)
            
            src_c2w_matrix = self.camera_to_world_matrix_list[src_id]
            src_w2c_matrix = self.world_to_camera_matrix_list[src_id]
            
            src_cameras.append(np.concatenate((list(img_size), self.intrinsic_matrix.flatten(), src_c2w_matrix.flatten())))
        
        src_rgbs = np.stack(src_rgbs, axis=0)
        #src_rgb_masks = np.stack(src_rgb_masks, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        
        depth_range = torch.tensor(
            [self.near_depth * 0.9, self.far_depth * 1.5]
        ).float()
        
        ret = {'idx': idx,
                'data_name': 'scannet',
                'rgb_path': query_rgb_path,

                'query_id': idx,
               'camera': torch.from_numpy(query_camera).float(),
               'rgb': torch.from_numpy(query_rgb).float(),
            #    'query_rgb_mask': torch.from_numpy(query_rgb_mask),
            #    'query_depth': torch.from_numpy(query_depth).float(),
               'depth_range': depth_range,
            #    'src_ids': torch.from_numpy(np.array(src_ids)),
               'src_cameras': torch.from_numpy(src_cameras).float(),
               'src_rgbs': torch.from_numpy(src_rgbs[..., :3]).float(),
            #   'src_rgb_masks': torch.from_numpy(src_rgb_masks),
            }


        return ret