import torch
from ..utils import get_explainer, get_expln_all_classes
from .utils import fidelity
import os
from tqdm.auto import tqdm

def get_all_fidelity(dataloader, original_model, backbone_model, explainer_name, num_classes, device,
                     reduction='none', skip=False, progress_bar=True, save_dir=None):
    explainer = get_explainer(original_model, backbone_model,
                                    explainer_name, device)
    
    fids = []
    for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        save_path = os.path.join(save_dir, f'{bi}.pt')
        if os.path.exists(save_path):
            # results = torch.load(save_path)
            # fids.append(results['fid'])
            continue

        if skip and bi >= 50: #% 10 != 1:
            break
            # if bi >= 10:
            #     break
            # continue
        if len(batch) == 2:
            inputs, labels = batch
        else: # 4
            inputs, labels, _, _ = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        expln, probs = get_expln_all_classes(original_model,
                                               inputs, explainer, 
                                               num_classes) #val_config['model']['num_classes']
        if len(fids) == 0:
            print('len(expln)', len(expln))
        # import pdb; pdb.set_trace()

        fid = fidelity(expln, probs)

        if save_dir is not None:
            save_path = os.path.join(save_dir, f'{bi}.pt')#save_dir / f'{bi}.pt'
            results = {
                'expln': expln,
                'probs': probs,
                'fid': fid
            }
            torch.save(results, save_path)

        fids.append(fid)
    fids = torch.cat(fids)
    if reduction == 'mean':
        fids = fids.mean()
    return fids