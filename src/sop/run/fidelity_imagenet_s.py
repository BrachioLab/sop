import os
import torch
import sys
sys.path.append('/shared_data0/weiqiuy/exlib/src')

import sys
sys.path.append('/shared_data0/weiqiuy/sop/src')
import sop

explainer_names = [
    'lime',
    'shap',
    'rise',
    'intgrad',
    'gradcam',
    'archipelago',
    'fullgrad',
    # 'attn', # need to make it an actual model
    'mfaba',
    'agi',
    'ampe',
    'bcos',
    'xdnn',
    'bagnet'
]

explainer_name = sys.argv[1]

if explainer_name not in explainer_names:
    raise ValueError(f'Unsupported explainer: {explainer_name}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sop.utils.seed_all(42)

# config
exp_config = sop.ImageNetConfig()
val_config = exp_config.get_config('val_sm')
# val_config['evaluation']['batch_size'] = 1
# val_config

# model
backbone_model, processor, backbone_config = sop.utils.imagenet_utils.get_model(val_config['model']['type'],
                                                                 backbone_model_name=val_config['model']['base'],
                                                                 backbone_processor_name=val_config['model']['base'],
                                                                )
backbone_model = backbone_model.to(device)

# get wrapped original model
from sop.utils.imagenet_utils import WrappedModel

original_model = WrappedModel(backbone_model, output_type='logits')
original_model = original_model.to(device)

# config
from exlib.modules.sop import SOPConfig, get_chained_attr

config = SOPConfig(os.path.join(val_config['model']['sop'], 'config.json'))

config.group_sel_scale = 0.05

config.__dict__.update(backbone_config.__dict__)
config.num_labels = len(backbone_config.id2label)

# get sop model
from sop.utils.imagenet_utils import get_model, get_wrapped_models

wrapped_backbone_model, class_weights, projection_layer = get_wrapped_models(
    backbone_model,
    config
)
wrapped_backbone_model = wrapped_backbone_model.to(device)
class_weights = class_weights.to(device)
projection_layer = projection_layer.to(device)

# sop
from exlib.modules.sop import SOPImageCls4

model = SOPImageCls4(config, wrapped_backbone_model, 
                     class_weights=class_weights, 
                     projection_layer=projection_layer)
state_dict = torch.load(os.path.join(val_config['model']['sop'], 
                                     'checkpoint.pth'))
print('Loaded step', state_dict['step'])
model.load_state_dict(state_dict['model'], strict=False)
model = model.to(device)
model.eval();

# data

val_dataset, val_dataloader = sop.utils.get_dataset(val_config['dataset']['name'], 
                                          split=val_config['evaluation']['split'], 
                                          num_data=val_config['evaluation']['num_data'],
                                          batch_size=val_config['evaluation']['batch_size'],
                                          processor=processor)

from sop.metrics import get_all_fidelity

import time
fids_dict = {}

# for explainer_name in explainer_names:
print(explainer_name)
start = time.time()

print('imagenet name', val_config['dataset']['name'])

save_dir = f'/shared_data0/weiqiuy/sop/results/{val_config["dataset"]["name"]}/{explainer_name}'

os.makedirs(save_dir, exist_ok=True)

fids = get_all_fidelity(val_dataloader, original_model, backbone_model, explainer_name, 
                        val_config['model']['num_classes'], device, skip=True, 
                        save_dir=save_dir)
end = time.time()
fids_dict[explainer_name] = {
    'fid': fids,
    'time': end - start
}



results_path = f'{save_dir}/fidelity.pt'

torch.save(fids_dict, results_path)