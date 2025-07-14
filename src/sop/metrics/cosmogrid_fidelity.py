import torch
from ..utils import get_expln_all_classes
from ..tasks.images.cosmogrid import get_explainer
from .utils import fidelity
import os
from tqdm.auto import tqdm

import sys
import os

class SuppressOutput:
    def __enter__(self):
        # Save the original stdout and stderr
        self._stdout = sys.stdout
        self._stderr = sys.stderr

        # Redirect stdout and stderr to /dev/null (a null file that discards input)
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        # Close the null file and restore stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


# # Example usage
# print("This will print normally.")

# with SuppressOutput():
#     print("This will not print.")
#     raise Exception("This exception will not print either.")

# print("Printing is restored.")


def get_all_fidelity_cosmo(dataloader, original_model, backbone_model, explainer_name, num_classes, device,
                     reduction='none', skip=False, progress_bar=True, 
                     save_dir='/shared_data0/weiqiuy/sop/exps/cosmogrid_cnn/attributions_fid/'):
    explainer = get_explainer(original_model, backbone_model,
                                    explainer_name, device)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    fids = []
    for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        save_path = os.path.join(save_dir, f'{bi}.pt')
        if os.path.exists(save_path):
            results = torch.load(save_path)
            fids.append(results['fid'])
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
        # inputs = batch['image']
        # labels = batch['label']
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.float()
        
        with SuppressOutput():
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