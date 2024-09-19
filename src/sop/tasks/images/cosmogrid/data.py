from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import os
import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import numpy as np
from exlib.datasets.cosmogrid import CosmogridDataset
from exlib.evaluators.common import convert_idx_masks_to_bool

from collections import namedtuple


def mask_transform(mask, config, processor=None):
    seg_mask_cut_off = config.num_masks_max
    # Preprocess the mask using the ViTImageProcessor
    if len(mask.shape) == 2 and mask.dtype == torch.bool:
        mask_dim1, mask_dim2 = mask.shape
        mask = mask.unsqueeze(0).expand(3, 
                                        mask_dim1, 
                                        mask_dim2).float()
        if processor is not None:
            inputs = processor(mask, 
                            do_rescale=False, 
                            do_normalize=False,
                            return_tensors='pt')
            # (1, 3, 224, 224)
            return inputs['pixel_values'][0][0]
        else:
            return mask
    else: # len(mask.shape) == 3
        if mask.dtype != torch.bool:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            # import pdb; pdb.set_trace()
            # mask = F.one_hot(mask).permute(0,3,1,2).to(x.device) # only applies to index masks
            mask = convert_idx_masks_to_bool(mask)
        bsz, mask_dim1, mask_dim2 = mask.shape
        mask = mask.unsqueeze(1).expand(bsz, 
                                        3, 
                                        mask_dim1, 
                                        mask_dim2).float()

        if bsz < seg_mask_cut_off:
            repeat_count = seg_mask_cut_off // bsz + 1
            mask = torch.cat([mask] * repeat_count, dim=0)

        # add additional mask afterwards
        mask_sum = torch.sum(mask[:seg_mask_cut_off - 1], dim=0, keepdim=True).bool()
        if False in mask_sum:
            mask = mask[:seg_mask_cut_off - 1]
            compensation_mask = (1 - mask_sum.int()).bool()
            mask = torch.cat([mask, compensation_mask])
        else:
            mask = mask[:seg_mask_cut_off]

        if processor is not None:
            inputs = processor(mask, 
                            do_rescale=False, 
                            do_normalize=False,
                            return_tensors='pt')
            
            return inputs['pixel_values'][:,0]
        else:
            return mask[:,0]

DatasetOutput = namedtuple('DatasetOutput', ['dataset', 'dataloader'])


def get_dataset(dataset_name, split='val', num_data=-1, batch_size=16, shuffle=False,
                processor=None, attr_dir=None, debug=False, config=None):
    if dataset_name == 'cosmogrid':
        # TRAIN_DATA_DIR = '/shared_data0/weiqiuy/sop/data/cosmogrid'
        # MASK_PATH = '/shared_data0/weiqiuy/sop/data/cosmogrid/masks/X_maps_Cosmogrid_100k_watershed_diagonal.npy'

        TRAIN_DATA_DIR = '/scratch/weiqiuy/datasets/cosmogrid'
        MASK_PATH = '/scratch/weiqiuy/datasets/cosmogrid/masks/X_maps_Cosmogrid_100k_watershed_diagonal.npy'

        dataset = CosmogridDataset(data_dir=TRAIN_DATA_DIR, split=split, data_size=num_data,
                                 mask_path=MASK_PATH, mask_transform=lambda x: mask_transform(x, config=config, processor=processor), 
                                 mask_big_first=True, even_data_sample=True)
    else:
        raise ValueError(f'dataset {dataset_name} is not the dataset')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    return DatasetOutput(dataset, dataloader)
