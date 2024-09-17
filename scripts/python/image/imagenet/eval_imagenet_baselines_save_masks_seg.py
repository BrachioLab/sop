import json
import os
import argparse

import numpy as np
import random
import torch
from torch import nn, optim
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path
from torch.utils.data import DataLoader, Subset, Dataset
import sys
sys.path.append('lib/exlib/src')
from exlib.modules.sop import SOPImageCls, SOPConfig, get_chained_attr
from exlib.explainers.archipelago import ArchipelagoImageCls
from exlib.explainers.lime import LimeImageCls
from exlib.explainers.common import patch_segmenter
from exlib.explainers import ShapImageCls
from exlib.explainers import RiseImageCls
from exlib.explainers import IntGradImageCls
from exlib.explainers import GradCAMImageCls
from exlib.explainers.fullgrad import FullGradImageCls
from exlib.explainers.attn import AttnImageCls

sys.path.append('src')
from sop.utils.data_utils import ImageFolderSegSubDataset
from sop.utils.imagenet_utils import WrappedModel
from sop.utils.expln_utils import get_explainer

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

import torch.nn.functional as F

# from collections import namedtuple

# WrappedBackboneOutput = namedtuple("WrappedBackboneOutput", 
#                                   ["logits",
#                                    "pooler_output"])


# class WrappedBackboneModel(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
    
#     def forward(self, inputs):
#         outputs = self.model(inputs, output_hidden_states=True)
#         return WrappedBackboneOutput(outputs.logits, outputs.hidden_states[-1][:,0])
    


# class WrappedModel(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
    
#     def forward(self, inputs):
#         outputs = self.model(inputs, output_hidden_states=True)
#         return outputs.logits



# def resize_binary_image(image, size, mode='bilinear'):
    # # Resize the image
    # resized_image = F.interpolate(image, size=size, mode=mode, align_corners=False)

    # # Threshold the image to convert values back to binary (0 or 1)
    # thresholded_image = (resized_image > 0.5).float()

    # return thresholded_image

# class ImageFolderSegSubDataset(Dataset):
#     def __init__(self, data_dir, seg_dir, transform=None, num_data=-1):
#         self.data_dir = data_dir
#         self.seg_dir = seg_dir
#         self.transform = transform
#         self.image_paths = []
#         self.seg_paths = []
#         self.labels = []
#         self.all_labels = []
        
#         self.image_paths_all = []
#         self.seg_paths_all = []
#         self.labels_all = []
#         self.use_indices = []
        
#         for label in tqdm(sorted(os.listdir(data_dir))):
#             seg_dirname = os.path.join(seg_dir, label)
#             img_dirname = os.path.join(data_dir, label)
#             if os.path.isdir(img_dirname):
#                 self.all_labels.append(label)
#             if os.path.isdir(seg_dirname):
#                 count = 0
#                 for i, seg_path in enumerate(sorted(os.listdir(seg_dirname))):
#                     if num_data != -1 and count >= num_data:
#                         break
#                     image_path = seg_path.replace('.png', '.JPEG')
#                     if not os.path.exists(os.path.join(data_dir, label, image_path)):
#                         continue
#                     self.image_paths.append(os.path.join(data_dir, label, image_path))
#                     self.seg_paths.append(os.path.join(seg_dir, label, seg_path))
#                     self.labels.append(label)
#                     count += 1
                    
#             if os.path.isdir(img_dirname):
#                 for i, image_path in enumerate(sorted(os.listdir(img_dirname))):
#                     if os.path.join(data_dir, label, image_path) in self.image_paths:
#                         self.use_indices.append(len(self.image_paths_all)) # add the index in
#                     seg_path = image_path.replace('.JPEG', '.png')
#                     self.image_paths_all.append(os.path.join(data_dir, label, image_path))
#                     self.seg_paths_all.append(os.path.join(seg_dir, label, seg_path))
#                     self.labels_all.append(label)
        
#         print('Loaded {} images and {} classes'.format(len(self.use_indices), len(self.all_labels))) 
    
#     def __len__(self):
#         return len(self.use_indices)
    
#     def __getitem__(self, i):
#         idx = self.use_indices[i]
        
#         image_path = self.image_paths_all[idx]
#         label = self.all_labels.index(self.labels_all[idx])
#         seg_path = self.seg_paths_all[idx]
    
#         image = Image.open(image_path)
#         if self.transform is not None:
#             image = self.transform(image)
#         seg = Image.open(seg_path)
#         seg = torch.tensor(np.asarray(seg))
#         seg = seg.sum(-1)[None,None].float() # (bsz, num_channels, img_dim1, img_dim2)
#         seg = resize_binary_image(seg, size=(image.shape[-2], image.shape[-1]))[0]
#         return image, label, seg, idx


EXPLAINER_NAMES = [
    'lime', 'archipelago', 'rise', 'shap', 'intgrad', 'gradcam', 'fullgrad', 'attn',
    'mfaba', 'agi', 'ampe', 'bcos', 'xdnn', 'bagnet'
]

if __name__ == '__main__':
    explainer_name = sys.argv[1]
    if len(sys.argv) > 2:
        num_data = int(sys.argv[2])
    else:
        num_data = -1
    if explainer_name not in EXPLAINER_NAMES:
        raise ValueError('Invalid explainer name' + explainer_name)

    if len(sys.argv) > 3:
        save_aggr_pred = bool(int(sys.argv[3]))
    else:
        save_aggr_pred = False

    if len(sys.argv) > 4:
        backbone_model_type = sys.argv[4]
    else:
        backbone_model_type = 'vit'

    if len(sys.argv) > 5:
        split = sys.argv[5]
    else:
        split = 'val'

    if len(sys.argv) > 6:
        save_together = int(sys.argv[6])
    else:
        save_together = -1

    if len(sys.argv) > 7:
        teacher_forcing = bool(int(sys.argv[7]))
    else:
        teacher_forcing = True

    if len(sys.argv) > 8:
        num_samples = int(sys.argv[8])
    else:
        num_samples = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SEED = 42
    if SEED != -1:
        # Torch RNG
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # Python RNG
        np.random.seed(SEED)
        random.seed(SEED)

    # model paths
    # backbone_model_name = 'pt_models/vit-base-patch16-224-imagenet10cls'
    backbone_model_name = 'google/vit-base-patch16-224'
    backbone_processor_name = 'google/vit-base-patch16-224'
    # sop_config_path = 'configs/imagenet_m.json'

    # data paths
    TRAIN_DATA_DIR = '/scratch/datasets/imagenet/train'
    VAL_DATA_DIR = '/scratch/datasets/imagenet/val'

    TRAIN_SEG_DIR = '/shared_data0/weiqiuy/github/ImageNet-S/datapreparation/ImageNetS919/train-semi-segmentation'
    VAL_SEG_DIR = '/shared_data0/weiqiuy/github/ImageNet-S/datapreparation/ImageNetS919/validation-segmentation'

    # training args
    batch_size = 1
    lr = 0.000005
    num_epochs = 20
    warmup_steps = 2000
    mask_batch_size = 64

    # experiment args
    exp_dir = f'exps/imagenet_{backbone_model_type}_1' #{num_data}'

    if backbone_model_type == 'resnet':
        backbone_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
        processor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        backbone_model = AutoModelForImageClassification.from_pretrained(backbone_model_name)
        processor = AutoImageProcessor.from_pretrained(backbone_processor_name)
        backbone_config = AutoConfig.from_pretrained(backbone_model_name)

    # config = SOPConfig()
    # config.update_from_json(os.path.join(exp_dir, 'config.json'))

    if backbone_model_type == 'resnet':
        def transform(image):
            # Preprocess the image using the ResNet
            image = image.convert("RGB")
            inputs = processor(image)
            return inputs
    else:
        def transform(image):
            # Preprocess the image using the ViTImageProcessor
            image = image.convert("RGB")
            inputs = processor(image, return_tensors='pt')
            return inputs['pixel_values'].squeeze(0)

    # Load the dataset
    if split == 'train':
        val_dataset = ImageFolderSegSubDataset(TRAIN_DATA_DIR, TRAIN_SEG_DIR, 
                            transform=transform, num_data=num_data)
        # train_dataset = ImageFolderSubDataset(TRAIN_DATA_DIR, transform=transform, num_data=num_data)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        dataset = train_dataset
    elif split == 'val':
        val_dataset = ImageFolderSegSubDataset(VAL_DATA_DIR, VAL_SEG_DIR, 
                            transform=transform, num_data=num_data)
        # val_dataset = ImageFolderSubDataset(VAL_DATA_DIR, transform, num_data=num_data)
        dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        dataset = val_dataset

    if backbone_model_type == 'resnet':
        original_model = backbone_model
    else:
        original_model = WrappedModel(backbone_model, output_type='logits')
    original_model = original_model.to(device)
    original_model.eval();

    explainer = get_explainer(original_model, backbone_model, explainer_name, device, 
                                 num_samples=num_samples, num_patches=7)

    # def segmenter(x):
    #     return patch_segmenter(x, sz=(14, 14))

    # if explainer_name == 'lime':
    #     eik = {
    #         "segmentation_fn": patch_segmenter,
    #         "top_labels": 1000, 
    #         "hide_color": 0, 
    #         "num_samples": num_samples
    #     }
    #     gimk = {
    #         "positive_only": False
    #     }
    #     explainer = LimeImageCls(original_model, 
    #                         explain_instance_kwargs=eik, 
    #                         get_image_and_mask_kwargs=gimk)
    # elif explainer_name == 'shap':
    #     sek = {'max_evals': num_samples}
    #     explainer = ShapImageCls(original_model,
    #                 shap_explainer_kwargs=sek)
    # elif explainer_name == 'rise':
    #     explainer = RiseImageCls(original_model, N=num_samples)
    # elif explainer_name == 'intgrad':
    #     explainer = IntGradImageCls(original_model, num_steps=num_samples)
    # elif explainer_name == 'gradcam':
    #     # import pdb; pdb.set_trace()
    #     if backbone_model_type == 'resnet':
    #         def reshape_transform(tensor):
    #             return tensor
    #         explainer = GradCAMImageCls(original_model, 
    #                                     [original_model.layer4[-1].conv2],
    #                                     reshape_transform=reshape_transform)
    #     else:
    #         explainer = GradCAMImageCls(original_model, 
    #                                     [original_model.model.vit.encoder.layer[-1].layernorm_before])
    # elif explainer_name == 'archipelago':
    #     explainer = ArchipelagoImageCls(original_model, segmenter=segmenter)
    #     # explainer = ArchipelagoImageCls(original_model)
    # elif explainer_name == 'fullgrad':
    #     explainer = FullGradImageCls(original_model)
    # elif explainer_name == 'attn':
    #     explainer = AttnImageCls(backbone_model)
    # elif explainer_name == 'mfaba':
    #     from exlib.explainers.mfaba import MfabaImageCls
    #     explainer = MfabaImageCls(original_model)
    # elif explainer_name == 'agi':
    #     from exlib.explainers.agi import AgiImageCls
    #     explainer = AgiImageCls(original_model)
    # elif explainer_name == 'ampe':
    #     from exlib.explainers.ampe import AmpeImageCls
    #     explainer = AmpeImageCls(original_model, N=5, num_steps=4)
    # elif explainer_name == 'bcos':
    #     from exlib.modules.bcos import BCos
    #     model = BCos()
    #     explainer = model
    # elif explainer_name == 'xdnn':
    #     from exlib.modules.xdnn import XDNN
    #     model = XDNN('xfixup_resnet50', 
    #          '/shared_data0/weiqiuy/github/fast-axiomatic-attribution/pt_models/xfixup_resnet50_model_best.pth.tar').to(device)
    #     explainer = model
    # elif explainer_name == 'bagnet':
    #     from exlib.modules.bagnet import BagNet
    #     model = BagNet()
    #     explainer = model
    # else:
    #     raise ValueError('Invalid explainer name' + explainer_name)
    
    # explainer = explainer.to(device)
        
    consistents = 0
    corrects = 0
    total = 0

    if teacher_forcing:
        tgt = 'label'
    else:
        tgt = 'pred'

    if num_samples != 1000 and explainer_name in ['lime', 'shap', 'rise', 'intgrad']:
        attr_dir = os.path.join(exp_dir, 'attributions_seg', 
            f'{explainer_name}_1_{tgt}_{num_samples}', split)
            # f'{explainer_name}_{num_data}_{tgt}_{num_samples}', split)
    else:
        attr_dir = os.path.join(exp_dir, 'attributions_seg', 
            f'{explainer_name}_1_{tgt}', split)
                # f'{explainer_name}_{num_data}_{tgt}', split)

    print('attr_dir', attr_dir)
    os.makedirs(attr_dir, exist_ok=True)

    count = 0
    print('num classes', len(backbone_config.id2label))

    start_bi = 0
    attributions_results_batch = []
    for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # if bi % (len(val_dataloader) // (num_data * len(backbone_config.id2label))) != 0:
        # if bi % len(backbone_config.id2label) >= num_data:
        #     continue
        inputs, labels, segs, idxs = batch
        inputs, labels, segs = inputs.to(device), labels.to(device), segs.to(device)
        bsz = inputs.shape[0]
        try:
            relative_path = os.path.relpath(dataset.image_paths_all[idxs[-1]], dataset.data_dir)
        except:
            import pdb; pdb.set_trace()
            relative_path = os.path.relpath(dataset.image_paths_all[idxs[-1]], dataset.data_dir)
        if os.path.exists(os.path.join(attr_dir, relative_path + '.pt')):
            count += bsz
            continue

        with torch.no_grad():
            logits = original_model(inputs)
            preds = torch.argmax(logits, dim=-1)

            if explainer_name in ['bcos', 'xdnn', 'bagnet']:
                # if explainer_name == 'bcos':
                #     expln = explainer((inputs + 1)/2)
                # else:
                expln = explainer(inputs)
                
            else:
                if teacher_forcing:
                    expln = explainer(inputs, labels)
                else:
                    expln = explainer(inputs, preds)

            aggr_preds = []
            explns = []
            for j in range(bsz):
                relative_path = os.path.relpath(dataset.image_paths_all[idxs[j]], dataset.data_dir)
                os.makedirs(os.path.dirname(os.path.join(attr_dir, relative_path)), exist_ok=True)
                torch.save(
                    expln.attributions[j], 
                    os.path.join(attr_dir, relative_path + '.pt')
                    )
                # print('explainer_name', explainer_name)

                if explainer_name == 'archipelago':
                    group_results = {
                        'mask_weights': expln.explainer_output['mask_weights'][j],
                        'masks': expln.explainer_output['masks'][j]
                    }
                    torch.save(
                        group_results, 
                        os.path.join(attr_dir, relative_path + '_group_results.pt')
                        )
                elif explainer_name in ['bcos', 'xdnn', 'bagnet']:
                    torch.save(
                        expln, os.path.join(attr_dir, relative_path + '_expln.pt')
                    )
                
                count += 1

            # aggr_preds = torch.stack(aggr_preds)
            # connsist = (aggr_preds == preds).sum().item()
            correct = (preds == labels).sum().item()

            # consistents += connsist
            corrects += correct
            total += bsz
            
    # print('Consistency: ', consistents / total)
    print('Accuracy: ', corrects / total)
    results = {'consistency': consistents / total, 'accuracy': corrects / total}
