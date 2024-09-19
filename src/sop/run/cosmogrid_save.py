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
    # 'attn', # don't work with non transformer
    'mfaba',
    # 'agi', # only have implementation for classification
    'ampe',
    # 'bcos',
    # 'xdnn',
    # 'bagnet'
]

explainer_name = sys.argv[1]

if explainer_name not in explainer_names:
    raise ValueError(f'Unsupported explainer: {explainer_name}')


exp_dir = '/shared_data0/weiqiuy/sop/exps/cosmogrid_cnn/attributions'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sop.utils.seed_all(42)

# config
exp_config = sop.tasks.cosmogrid.CosmogridConfig()
val_config = exp_config.get_config('test')
val_config['evaluation']['batch_size'] = 16
val_config

# model
backbone_model, original_model, processor, backbone_config, model, config = sop.tasks.cosmogrid.get_model(
    val_config['model']['type'],
    backbone_model_name=val_config['model']['base'],
    backbone_processor_name=val_config['model']['base'],
    sop_model_name=val_config['model']['sop'],
    eval_mode=True
)
backbone_model = backbone_model.to(device)
original_model = original_model.to(device)
model = model.to(device)

# data
val_dataset, val_dataloader = sop.tasks.cosmogrid.get_dataset(val_config['dataset']['name'], 
                                          split=val_config['evaluation']['split'], 
                                          num_data=val_config['evaluation']['num_data'],
                                          batch_size=val_config['evaluation']['batch_size'],
                                          processor=processor,
                                          config=config)

from tqdm.auto import tqdm

# explainer_name = 'lime'
# explainer_name = 'mfaba'
# explainer_name = 'ampe'
# explainer_name = 'agi' # cls only


explainer = sop.tasks.cosmogrid.get_explainer(original_model, backbone_model, explainer_name, device)

output_dirname = os.path.join(exp_dir, f'{explainer_name}_{val_config["evaluation"]["split"]}')
os.makedirs(output_dirname, exist_ok=True)

count = 0

for batch in tqdm(val_dataloader):

    inputs, labels, masks, masks_i = batch
    inputs = inputs.to(device, dtype=torch.float)
    labels = labels.to(device, dtype=torch.float)
    masks = masks.to(device)
    
    original_logits = backbone_model(inputs).logits
    
    labels_omega = torch.zeros(labels.shape[0]).long().to(device)
    labels_sigma = torch.ones(labels.shape[0]).long().to(device)
    
    expln_omega = explainer(inputs, labels_omega)
    expln_sigma = explainer(inputs, labels_sigma)
    
    bsz = inputs.shape[0]
    
    for i in tqdm(range(bsz)):
        output_filepath = os.path.join(output_dirname, f'{count}.pt')
        # if os.path.exists(output_filepath):
        #     count += 1
        #     continue

        entry = {'image': inputs[i],
                 'original_logits': original_logits[i],
                 'omega_mask': expln_omega.attributions[i],
                 'sigma_mask': expln_sigma.attributions[i],
                'label': labels[i],
                 'omega_group_mask': expln_omega.group_masks[i] if 'group_masks' in expln_omega._fields else None,
                 'sigma_group_mask': expln_sigma.group_masks[i] if 'group_masks' in expln_sigma._fields else None,
                 'omega_group_attr': expln_omega.group_attributions[i] if 'group_attributions' in expln_omega._fields else None,
                 'sigma_group_attr': expln_sigma.group_attributions[i] if 'group_attributions' in expln_sigma._fields else None,
                'num_labels': config.num_labels}

        torch.save(entry, output_filepath)
        count += 1