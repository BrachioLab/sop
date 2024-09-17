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


def resize_binary_image(image, size, mode='bilinear'):
    # Resize the image
    resized_image = F.interpolate(image, size=size, mode=mode, align_corners=False)

    # Threshold the image to convert values back to binary (0 or 1)
    thresholded_image = (resized_image > 0.5).float()

    return thresholded_image


class ImageFolderSegSubDataset(Dataset):
    def __init__(self, data_dir, seg_dir, attr_dir=None, transform=None, num_data=-1, debug=False):
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.attr_dir = attr_dir
        self.transform = transform
        self.image_paths = []
        self.seg_paths = []
        self.labels = []
        self.all_labels = []
        
        self.image_paths_all = []
        self.seg_paths_all = []
        self.labels_all = []
        self.use_indices = []

        # print('debug', debug)
        
        for label in tqdm(sorted(os.listdir(data_dir))):
            if debug:
                if len(self.all_labels) >= 100:
                    break
            seg_dirname = os.path.join(seg_dir, label)
            img_dirname = os.path.join(data_dir, label)
            if os.path.isdir(img_dirname):
                self.all_labels.append(label)
            if os.path.isdir(seg_dirname):
                count = 0
                for i, seg_path in enumerate(sorted(os.listdir(seg_dirname))):
                    if num_data != -1 and count >= num_data:
                        break
                    image_path = seg_path.replace('.png', '.JPEG')
                    if not os.path.exists(os.path.join(data_dir, label, image_path)):
                        continue
                    if self.attr_dir is not None:
                        attr_path = os.path.join(self.attr_dir, label, 
                                os.path.basename(image_path) + '.pt')
                        if not os.path.exists(attr_path):
                            print('failed' + attr_path)
                            import pdb; pdb.set_trace()
                            continue

                    self.image_paths.append(os.path.join(data_dir, label, image_path))
                    self.seg_paths.append(os.path.join(seg_dir, label, seg_path))
                    self.labels.append(label)
                    count += 1
                    
            if os.path.isdir(img_dirname):
                for i, image_path in enumerate(sorted(os.listdir(img_dirname))):
                    if os.path.join(data_dir, label, image_path) in self.image_paths:
                        
                        if self.attr_dir is not None:
                            try:
                                attr_path = os.path.join(self.attr_dir, label, os.path.basename(image_path) + '.pt')
                                attr = torch.load(attr_path)
                                # self.use_indices.append(len(self.image_paths_all)) # add the index in
                            except:
                                print('failed' + attr_path)
                                import pdb; pdb.set_trace()
                                continue
                        self.use_indices.append(len(self.image_paths_all)) # add the index in
                    seg_path = image_path.replace('.JPEG', '.png')
                    self.image_paths_all.append(os.path.join(data_dir, label, image_path))
                    self.seg_paths_all.append(os.path.join(seg_dir, label, seg_path))
                    self.labels_all.append(label)
        
        print('Loaded {} images and {} classes'.format(len(self.use_indices), len(self.all_labels))) 
    
    def __len__(self):
        return len(self.use_indices)
    
    def __getitem__(self, i):
        idx = self.use_indices[i]
        
        image_path = self.image_paths_all[idx]
        label = self.all_labels.index(self.labels_all[idx])
        seg_path = self.seg_paths_all[idx]
    
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        seg = Image.open(seg_path)
        seg = torch.tensor(np.asarray(seg))
        seg = seg.sum(-1)[None,None].float() # (bsz, num_channels, img_dim1, img_dim2)
        seg = resize_binary_image(seg, size=(image.shape[-2], image.shape[-1]))[0]
        
        if self.attr_dir is not None:
            attr_path = os.path.join(self.attr_dir, self.labels_all[idx], os.path.basename(image_path) + '.pt')
            attr = torch.load(attr_path)
            return image, label, seg, attr, idx
        return image, label, seg, idx


class ImageFolderSubDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_data=-1, start_data=0, debug=False):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label in sorted(os.listdir(data_dir)):
            if debug:
                if len(self.all_labels) >= 100:
                    break
            dirname = os.path.join(data_dir, label)
            if not os.path.isdir(dirname):
                continue
            for i, image_path in enumerate(sorted(os.listdir(dirname))):
                if i < start_data:
                    continue
                if num_data != -1 and i >= num_data:
                    break
                self.image_paths.append(os.path.join(data_dir, label, image_path))
                self.labels.append(label)
        self.all_labels = sorted(list(set(self.labels)))
        
        print('Loaded {} images and {} classes'.format(len(self.image_paths), len(self.all_labels)))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.all_labels.index(self.labels[idx])
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label



aug = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5)
])

def transform(image, processor=None):
    # Preprocess the image using the ViTImageProcessor
    image = image.convert("RGB")
    if processor is None:
        inputs = image
    else:
        inputs = processor(image, return_tensors='pt')
    # import pdb; pdb.set_trace()
    return inputs['pixel_values'].squeeze(0)

def transform_aug(image, processor=None):
    # Preprocess the image using the ViTImageProcessor
    image = image.convert("RGB")
    inputs = processor(image, return_tensors='pt')
    inputs = inputs['pixel_values']
    inputs = aug(inputs)
    return inputs.squeeze(0)


from collections import namedtuple


DatasetOutput = namedtuple('DatasetOutput', ['dataset', 'dataloader'])


def get_dataset(dataset_name, split='val', num_data=-1, start_data=0, batch_size=16, shuffle=False,
                processor=None, attr_dir=None, debug=False):
    if dataset_name == 'imagenet':
        TRAIN_DATA_DIR = '/scratch/datasets/imagenet/train'
        VAL_DATA_DIR = '/scratch/datasets/imagenet/val'
        if split == 'train':
            dataset = ImageFolderSubDataset(TRAIN_DATA_DIR, 
                                            transform=lambda x: transform_aug(x, processor=processor), 
                                            num_data=num_data,
                                            start_data=start_data,
                                            debug=debug)
        elif split == 'val':
            dataset = ImageFolderSubDataset(VAL_DATA_DIR, 
                                            transform=lambda x: transform(x, processor=processor), 
                                            num_data=num_data,
                                            start_data=start_data,
                                            debug=debug)
        elif split == 'train_val':
            dataset = ImageFolderSubDataset(TRAIN_DATA_DIR, 
                                            transform=lambda x: transform(x, processor=processor), 
                                            num_data=num_data,
                                            start_data=start_data,
                                            debug=debug)
        else:
            raise ValueError(f'split {split} not recognized')
        
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif dataset_name == 'imagenet_s':
        TRAIN_DATA_DIR = '/scratch/datasets/imagenet/train'
        VAL_DATA_DIR = '/scratch/datasets/imagenet/val'
        TRAIN_SEG_DIR = '/shared_data0/weiqiuy/github/ImageNet-S/datapreparation/ImageNetS919/train-semi-segmentation'
        VAL_SEG_DIR = '/shared_data0/weiqiuy/github/ImageNet-S/datapreparation/ImageNetS919/validation-segmentation'
        if split == 'train':
            dataset = ImageFolderSegSubDataset(TRAIN_DATA_DIR, TRAIN_SEG_DIR, attr_dir,
                                            transform=lambda x: transform_aug(x, processor=processor), 
                                            num_data=num_data,
                                            debug=debug)
        elif split == 'val':
            dataset = ImageFolderSegSubDataset(VAL_DATA_DIR, VAL_SEG_DIR, attr_dir,
                                            transform=lambda x: transform(x, processor=processor), 
                                            num_data=num_data,
                                            debug=debug)
        elif split == 'train_val':
            dataset = ImageFolderSegSubDataset(TRAIN_DATA_DIR, TRAIN_SEG_DIR, attr_dir,
                                            transform=lambda x: transform(x, processor=processor),  
                                            num_data=num_data,
                                            debug=debug)
        else:
            raise ValueError(f'split {split} not recognized')
    else:
        raise ValueError(f'dataset {dataset_name} is not the dataset')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    return DatasetOutput(dataset, dataloader)
