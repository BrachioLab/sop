import os
import torch
import sys
sys.path.append('/shared_data0/weiqiuy/exlib/src')

import sys
sys.path.append('/shared_data0/weiqiuy/sop/src')
import sop
from sop.metrics import get_ins_del_perc

methods = [
    'lime_20',
    'shap_20',
    'rise_20',
    'intgrad',
    'gradcam',
    'archipelago',
    'fullgrad',
    'attn', 
    'mfaba',
    'agi',
    'ampe',
    'bcos',
    'xdnn',
    'bagnet',
    'sop',
]

method = sys.argv[1]
# debug = True
debug = False

if method not in methods:
    raise ValueError(f'Unsupported explainer: {method}')

if len(sys.argv) > 2:
    step_size = sys.argv[2]
else:
    step_size = 'large'

if step_size == 'small':
    step_size_suffix = '_small'
    ins_steps=196
    start=1
    end=196
else:
    step_size_suffix = '_large' #''
    ins_steps=10
    start=0.1
    end=1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sop.utils.seed_all(42)

# config
exp_config = sop.ImageNetConfig()
val_config = exp_config.get_config('val_sm')
val_config['evaluation']['batch_size'] = 16
val_config

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
original_model.eval();

# config
from exlib.modules.sop import SOPConfig, get_chained_attr

config = SOPConfig(os.path.join(val_config['model']['sop'], 'config.json'))

# config.group_sel_scale = 0.05

config.__dict__.update(backbone_config.__dict__)
config.num_labels = len(backbone_config.id2label)

# get sop model
from sop.utils.imagenet_utils import get_model, get_wrapped_models

wrapped_backbone_model, class_weights, projection_layer = get_wrapped_models(
    backbone_model,
    config
)
wrapped_backbone_model = wrapped_backbone_model.to(device)
wrapped_backbone_model.eval();
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

results_ins = get_ins_del_perc(val_config, original_model, backbone_model, model, processor,
                     method, ins_steps=ins_steps, start=start, end=end, debug=debug)
results_del = get_ins_del_perc(val_config, original_model, backbone_model, model, processor,
                     method, ins_steps=ins_steps, start=start, end=end, debug=debug, deletion=True)

save_dir = f'/shared_data0/weiqiuy/sop/results/ins_del{step_size_suffix}/{val_config["dataset"]["name"]}/'
os.makedirs(save_dir, exist_ok=True)

results_path = f'{save_dir}/{method}.pt'

results = {
    'ins': results_ins,
    'del': results_del
}

torch.save(results, results_path)