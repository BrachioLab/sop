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
]

explainer_name = sys.argv[1]

if explainer_name not in explainer_names:
    raise ValueError(f'Unsupported explainer: {explainer_name}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sop.utils.seed_all(42)

# config
exp_config = sop.tasks.multirc.MultiRcConfig()
val_config = exp_config.get_config('test')
val_config['evaluation']['batch_size'] = 1
# val_config['evaluation']['batch_size'] = 1
# val_config

# model
backbone_model, original_model, original_model_softmax, projection_layer, processor, backbone_config, model, config = sop.tasks.multirc.get_model(val_config['model']['type'],
             backbone_model_name=val_config['model']['base'],
             backbone_processor_name=val_config['model']['base_processor'],
            )
backbone_model = backbone_model.to(device)
original_model = original_model.to(device)
projection_layer = projection_layer.to(device)
model = model.to(device)

# data
val_dataset, val_dataloader = sop.tasks.multirc.get_dataset(val_config['dataset']['name'], 
                                          split=val_config['evaluation']['split'], 
                                          num_data=val_config['evaluation']['num_data'],
                                          batch_size=val_config['evaluation']['batch_size'],
                                          processor=processor)

from sop.metrics import get_all_fidelity_text

import time
fids_dict = {}

# for explainer_name in explainer_names:
print(explainer_name)
start = time.time()

print('imagenet name', val_config['dataset']['name'])

save_dir = f'/shared_data0/weiqiuy/sop/results/{val_config["dataset"]["name"]}/{explainer_name}'

os.makedirs(save_dir, exist_ok=True)

fids = get_all_fidelity_text(val_dataloader, original_model, 
        original_model_softmax, backbone_model, 
        processor, explainer_name, 
        val_config['model']['num_classes'], device, skip=True,
        save_dir=save_dir)
end = time.time()
fids_dict[explainer_name] = {
    'fid': fids,
    'time': end - start
}

results_path = f'{save_dir}/fidelity.pt'

torch.save(fids_dict, results_path)