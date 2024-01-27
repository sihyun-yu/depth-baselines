# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from . import dataset_dict
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
from typing import Optional
from operator import itemgetter
import torch
from torch.utils.data import SubsetRandomSampler
from .data_utils import *
from torchvision.transforms.functional import resize
import torch.nn.functional as F

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def create_training_dataset(args):
    # parse args.train_dataset, "+" indicates that multiple datasets are used, for example "ibrnet_collect+llff+spaces"
    # otherwise only one dataset is used
    # args.dataset_weights should be a list representing the resampling rate for each dataset, and should sum up to 1

    print('training dataset: {}'.format(args.train_dataset))
    mode = 'train'
    if '+' not in args.train_dataset:
        train_dataset = dataset_dict[args.train_dataset](args, mode,
                                                         scenes=args.train_scenes
                                                         )
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    else:
        train_dataset_names = args.train_dataset.split('+')
        weights = args.dataset_weights
        assert len(train_dataset_names) == len(weights)
        assert np.abs(np.sum(weights) - 1.) < 1e-6
        print('weights:{}'.format(weights))
        train_datasets = []
        train_weights_samples = []
        for training_dataset_name, weight in zip(train_dataset_names, weights):
            train_dataset = dataset_dict[training_dataset_name](args, mode,
                                                                scenes=args.train_scenes,
                                                                )
            train_datasets.append(train_dataset)
            num_samples = len(train_dataset)
            weight_each_sample = weight / num_samples
            train_weights_samples.extend([weight_each_sample]*num_samples)

        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_weights = torch.from_numpy(np.array(train_weights_samples))
        sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_sampler = DistributedSamplerWrapper(sampler) if args.distributed else sampler

    return train_dataset, train_sampler

def create_valid_dataset(args):
    valid_dataset = dataset_dict[f"{args.train_dataset}"](
        args, mode='valid', scenes=args.train_scenes
    )

    return valid_dataset

def create_eval_dataset(args):
    eval_dataset = dataset_dict[f"{args.eval_dataset}"](
        args, mode='eval', scenes=args.eval_scenes
    )

    return eval_dataset


class MySubsetRandomSampler(SubsetRandomSampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
            indices (sequence): a sequence of indices
            generator (Generator): Generator used in sampling.
    """
    def __init__(self, indices, mult, compensate, num_data, generator=None) -> None:
            super().__init__(indices, generator)
            self.my_mult = mult
            self.compensate = compensate
            self.num_data = num_data

    def __iter__(self):
        for m in range(self.my_mult):
            randperm = torch.cat([torch.randperm(len(self.indices), generator=self.generator), torch.randperm(len(self.indices), generator=self.generator)])
            # randperm = torch.cat([torch.arange(len(self.indices)), torch.arange(len(self.indices))])
            randperm = randperm[:len(self.indices)+self.compensate]
            
            # this_indices = [self.indices[i] + m*self.num_data for i in randperm]
            # print(this_indices)
            for i in randperm:
                output = self.indices[i] + m*self.num_data
                yield output

    def __len__(self):
            return (len(self.indices) + self.compensate) * self.my_mult


class collate_fn(object):
    def __init__(self, N_rand, sample_mode, N_samples, inv_uniform, det, seed=None, resize_factor=1):
        self.N_rand = N_rand
        self.sample_mode = sample_mode

        self.N_samples = N_samples
        self.inv_uniform = inv_uniform
        self.det = det

        self.rng = np.random.RandomState(seed)
        self.resize_factor = resize_factor
    
    def __call__(self, data):
        data = torch.utils.data.dataloader.default_collate(data)
        
        # query_time = data['query_time'].float()
        src_ids = data['src_ids'].squeeze()
        
        query_frame_idx = int(data['query_id'].item())
        
        query_rgb = data['query_rgb']
        query_camera = data['query_camera']

        src_rgbs = data['src_rgbs']
        src_cameras = data['src_cameras']

        depth_range = data['depth_range']
        
        image_hw, intrinsic_matrix, c2w_matrix, w2c_matrix = query_camera[0, :2], query_camera[0, 2:11].reshape(3, 3), query_camera[0, 11:23].reshape(3, 4), query_camera[0, 23:].reshape(3, 4)
        H, W = [int(x) for x in image_hw]
        
        assert (query_rgb.shape[-3] == H) and (query_rgb.shape[-2] == W)
        
        if self.resize_factor != 1:
            query_rgb = query_rgb.squeeze(0).permute(2, 0, 1)
            query_rgb = resize(query_rgb, min(H, W) // self.resize_factor, antialias=True)
            query_rgb = query_rgb.permute(1, 2, 0)
            new_W, new_H = query_rgb.shape[:2]
            
            intrinsic_matrix[0, :] = intrinsic_matrix[0, :] * (new_W / W)
            intrinsic_matrix[1, :] = intrinsic_matrix[1, :] * (new_H / H)
            image_hw = (new_H, new_W)
            H, W = new_H, new_W
        
        uv_grid = create_meshgrid(height=H, width=W, normalized_coordinates=False)
        pixels = torch.cat([uv_grid, torch.ones_like(uv_grid[..., -1:])], dim=-1)  # (H, W, 3)

        query_rgb = query_rgb.reshape(-1, 3)
        pixels = pixels.reshape(-1, 3)
        
        select_inds = None
        if self.sample_mode is not None:
            select_inds = sample_random_pixel(self.N_rand, self.sample_mode, H, W, self.rng)
            
            query_rgb = query_rgb[select_inds]
            pixels = pixels[select_inds]

        # sample rays
        c2w_rotation = c2w_matrix[:3, :3]
        pixel2ray_matrix = torch.linalg.solve(intrinsic_matrix, c2w_rotation, left=False) # c2w_rotation @ inv(intrinsic_matrix)
        
        rays_direction = (pixel2ray_matrix @ pixels.transpose(-1, -2)).transpose(-1, -2) # (H*W, 3)
        rays_origin = c2w_matrix[:3, 3].unsqueeze(0) # 1 x 3

        # compute plucker coordinates given ray directions and origins
        rays_dir_basis = F.normalize(rays_direction, dim=-1)
        moment = torch.cross(rays_origin, rays_dir_basis, dim=-1)
        ref_rays_coords = torch.cat([rays_dir_basis, moment], dim=-1)
        
        # draw (or deterministically select) depth values, and compute 3D points given ray directions and origins
        (ref_pts_3d, ref_z_vals, ref_s_vals) = sample_along_camera_ray(rays_origin, rays_direction, depth_range, self.N_samples, self.inv_uniform, self.det)
        
        # _ref_pts_3d_h = torch.cat([ref_pts_3d, torch.ones_like(ref_pts_3d[..., -1:])], dim=-1)
        # _ref_pts_3d_h = _ref_pts_3d_h.reshape(H*W*64, 4)

        batched_data = {
            # 'query_time': query_time,
            'query_frame_idx': query_frame_idx,
            'query_rgb': query_rgb,
            'query_camera': query_camera,
            'src_rgbs': src_rgbs,
            'src_cameras': src_cameras,
            'src_frame_index': src_ids,
            # 'src_frame_offset': src_frame_offset,
            # 'rays_o': rays_o,
            # 'rays_d': rays_d,
            'ref_rays_coords': ref_rays_coords,
            'ref_pts_3d': ref_pts_3d,
            'ref_z_vals': ref_z_vals,
            'ref_s_vals': ref_s_vals,
            'depth_range': depth_range,
            'select_inds': select_inds,
            'HW': (H, W)
            # 'uv_grid': uv_grid
        }

        return batched_data