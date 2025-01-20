import torch
# from ..utils import get_explainer, get_expln_all_classes
from ..tasks.texts.base import get_explainer, get_expln_all_classes
from .utils import fidelity
import os
from tqdm.auto import tqdm

def get_all_fidelity_text(dataloader, original_model, original_model_softmax, backbone_model, processor,
                    explainer_name, num_classes, device,
                     reduction='none', skip=False, progress_bar=True, save_dir=None):
    # explainer = get_explainer(original_model, backbone_model,
    #                                 explainer_name, device)
    print('len(dataloader)', len(dataloader))
    explainer = get_explainer(original_model, original_model_softmax, 
                            backbone_model, processor, explainer_name, device)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
    fids = []
    for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        
        if skip and bi >= 50: #% 10 != 1:
            break
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, f'{bi}.pt')
            if os.path.exists(save_path):
                results = torch.load(save_path)
                fids.append(results['fid'])
                continue
        else:
            save_path = None

        
            # if bi >= 10:
            #     break
            # continue
        if not isinstance(batch['input_ids'], torch.Tensor):
            inputs = torch.stack(batch['input_ids']).transpose(0, 1).to(device)
            if 'token_type_ids' in batch:
                token_type_ids = torch.stack(batch['token_type_ids']).transpose(0, 1).to(device)
            else:
                token_type_ids = None
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(device)

            # concatenated_rows = [torch.stack(sublist) for sublist in batch['segs']]
            # segs = torch.stack(concatenated_rows).permute(2, 0, 1).to(device).float()

            # print('segs', segs.shape)
        else:
            inputs = batch['input_ids'].to(device)
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            else:
                token_type_ids = None
            attention_mask = batch['attention_mask'].to(device)

            attention_mask = batch['attention_mask'].to(device)
            # segs = batch['segs'].to(device).float()
        kwargs = {
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }


        labels = batch['label'].to(device)

        bsz = inputs.shape[0]

        if explainer_name == 'archipelago':
            # trunk all inputs to only top 10 tokens.
            inputs = inputs[:, :10]
            attention_mask = attention_mask[:, :10]
            token_type_ids = token_type_ids[:, :10]
            kwargs['attention_mask'] = attention_mask
            kwargs['token_type_ids'] = token_type_ids

        expln, probs = get_expln_all_classes(original_model,
                                            inputs, explainer, 
                                            num_classes, 
                                            explainer_name,
                                            processor,
                                            kwargs=kwargs) #val_config['model']['num_classes']
        
        if len(fids) == 0:
            print('len(expln)', len(expln))
        # import pdb; pdb.set_trace()

        fid = fidelity(expln, probs)

        if save_path is not None and not os.path.exists(save_path):
            # save_path = os.path.join(save_dir, f'{bi}.pt')#save_dir / f'{bi}.pt'
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