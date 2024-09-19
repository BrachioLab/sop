import os
import torch
import sys
sys.path.append('/shared_data0/weiqiuy/exlib/src')

import sys
sys.path.append('/shared_data0/weiqiuy/sop/src')
import sop

from sop.metrics import get_acc
from sop.tasks.images.imagenet import get_explainer

import numpy as np
from tqdm.auto import tqdm

debug = False

methods = [
    'shap_20',
    'rise_20',
    'lime_20',
    'fullgrad',
    'gradcam',
    'intgrad',
    'attn',
    'archipelago',
    'mfaba',
    'agi',
    'ampe',
    'bcos',
    'xdnn',
    'bagnet',
    'sop',
]

method = sys.argv[1]
if method not in methods:
    raise ValueError(f'Unsupported explainer: {method}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sop.utils.seed_all(42)

# config
exp_config = sop.ImageNetConfig()
val_config = exp_config.get_config('val_sm')
val_config['evaluation']['batch_size'] = 16
val_config

backbone_model, original_model, processor, backbone_config, model, config = sop.tasks.imagenet.get_model(val_config['model']['type'],
                                                                 backbone_model_name=val_config['model']['base'],
                                                                 backbone_processor_name=val_config['model']['base'],
                                                                 sop_model_name=val_config['model']['sop'], eval_mode=True
                                                                                                        )

backbone_model = backbone_model.to(device)
original_model = original_model.to(device)
model = model.to(device)


if method == 'sop':
    explainer = model
else:
    explainer = get_explainer(original_model, backbone_model, method.split('_')[0], device)
    
method_list = method.split('_')
explainer_name = method_list[0]

if len(method_list) == 2:
    suffix = f'_{method_list[1]}'
else:
    suffix = ''

if method != 'sop':
    ATTR_VAL_DATA_DIR = f'/shared_data0/weiqiuy/sop/exps/imagenet_vit_1/attributions_seg/{explainer_name}_1_pred{suffix}/val'
else:
    ATTR_VAL_DATA_DIR = None
    
val_dataset, val_dataloader = sop.tasks.imagenet.get_dataset(val_config['dataset']['name'], 
                                          split=val_config['evaluation']['split'], 
                                          num_data=val_config['evaluation']['num_data'],
                                          batch_size=val_config['evaluation']['batch_size'],
                                                        attr_dir=ATTR_VAL_DATA_DIR,
                                          processor=processor, debug=debug)


results_all = {}
for k in tqdm(np.linspace(0.1, 1, 10)):
    results = get_acc(val_dataloader, explainer, method, device, k=k, eval_all=True)
    results_all[k] = results

save_dir = f'/shared_data0/weiqiuy/sop/results/sparsity/{val_config["dataset"]["name"]}/'
os.makedirs(save_dir, exist_ok=True)

results_path = f'{save_dir}/{method}.pt'

torch.save(results, results_path)