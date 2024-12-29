from torch.utils.data import DataLoader
from datasets import load_dataset
import torch


def sent_seg(input_ids, processor):
    SENT_SEPS = [processor.convert_tokens_to_ids(processor.tokenize(token)[0]) for token in [';',',','.','?','!',';']]
    SEP = processor.convert_tokens_to_ids(processor.tokenize('[SEP]')[0])
    print('SEP', SEP, 'SENT_SEPS', SENT_SEPS)

    segs = []
    count = 1
    for i, input_id in enumerate(input_ids):
        if count in [0, -1]:
            if input_id == SEP:
                count = -1
            segs.append(count)
            continue
        else:
            if input_id in SENT_SEPS:
                segs.append(count)
                count += 1
            elif input_id == SEP:
                if count > 0:
                    count = 0
                    segs.append(count)
                else:
                    segs.append(count)
                    count = -1
            else: # normal character
                segs.append(count)
    return segs

def convert_idx_masks_to_bool_text(masks):
    """
    input: masks (1, seq_len)
    output: masks_bool (num_masks, seq_len)
    """
    unique_idxs = torch.sort(torch.unique(masks)).values
    unique_idxs = unique_idxs[unique_idxs != -1]
    unique_idxs = unique_idxs[unique_idxs != 0]
    idxs = unique_idxs.view(-1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[1])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool

# used for segmenting text
def get_mask_transform_text(num_masks_max=200, processor=None):
    def mask_transform(mask):
        seg_mask_cut_off = num_masks_max
        # print('mask 1', mask)
        # if mask.max(dim=-1) > seg_mask_cut_off:
        # import pdb; pdb.set_trace()
        if mask.max(dim=-1).values.item() > seg_mask_cut_off: # combine segments
            mask_new = (mask / (mask.max(dim=-1).values / seg_mask_cut_off)).int().float() + 1
            # bsz, seq_len = mask_new.shape
            # print('mask 2', mask_new)
            # import pdb; pdb.set_trace()
            mask_new[mask == 0] = 0
            mask_new[mask == -1] = -1
            mask = mask_new
        
        if mask.dtype != torch.bool:
            if len(mask.shape) == 1:
                mask = mask.unsqueeze(0)
            # print('mask', mask.shape)
            mask_bool = convert_idx_masks_to_bool_text(mask)
        # print(mask.shape)
        bsz, seq_len = mask.shape
        mask_bool = mask_bool.float()
        
        

        if bsz < seg_mask_cut_off:
            repeat_count = seg_mask_cut_off // bsz + 1
            mask_bool = torch.cat([mask_bool] * repeat_count, dim=0)

        # add additional mask afterwards
        mask_bool_sum = torch.sum(mask_bool[:seg_mask_cut_off - 1], dim=0, keepdim=True).bool()
        if False in mask_bool_sum:
            # import pdb; pdb.set_trace()
            
            # import pdb; pdb.set_trace()
            compensation_mask = (1 - mask_bool_sum.int()).bool()
            compensation_mask[mask == 0] = False
            compensation_mask[mask == -1] = False
            if compensation_mask.sum() > 0:
                mask_bool = mask_bool[:seg_mask_cut_off - 1]
                mask_bool = torch.cat([mask_bool, compensation_mask])
        mask_bool = mask_bool[:seg_mask_cut_off]
        return mask_bool
    return mask_transform

# mask_transform = get_mask_transform_text(config.num_masks_max)

def augment_input_and_attention(input_ids, attention_mask):
    # Example augmentation: randomly mask tokens and attention mask
    probability_of_masking = 0.1
    for i in range(len(input_ids)):
        if random.random() < probability_of_masking and input_ids[i] not in [0, 1, 2]:  # Assuming 0, 1, 2 are special tokens
            input_ids[i] = processor.mask_token_id
            attention_mask[i] = 0  # Also mask the attention mask
    return input_ids, attention_mask

def transform(batch, is_training=False, processor=None):
    if processor is not None:
        inputs = processor(batch['passage'], 
                           batch['query_and_answer'], 
                           padding='max_length', 
                           truncation=True, 
                           max_length=512)
        
        # Apply augmentation only if it's training data
        import copy
        inputs_original = copy.deepcopy(inputs)
        if is_training:
            input_ids_all = []
            attention_mask_all = []
            for i in range(len(inputs['input_ids'])):
                input_ids, attention_mask = augment_input_and_attention(inputs['input_ids'][i], 
                                                                                        inputs['attention_mask'][i])
                input_ids_all.append(input_ids)
                attention_mask_all.append(attention_mask)
            inputs['input_ids'] = input_ids_all
            inputs['attention_mask'] = attention_mask_all
            
        # seg = sent_seg(inputs['input_ids'])
        # seg_bool = mask_transform(torch.tensor(seg))
        try:
            inputs = {k: torch.tensor(v) for k, v in inputs.items()}
        except:
            import pdb; pdb.set_trace()
        # inputs['segs'] = seg_bool
        inputs['label'] = batch['label']
        
        return inputs
    else:
        return batch


from collections import namedtuple


DatasetOutput = namedtuple('DatasetOutput', ['dataset', 'dataloader'])


def get_dataset(dataset_name, split='val', num_data=-1, start_data=0, batch_size=16, shuffle=False,
                processor=None, attr_dir=None, debug=False, raw_data=False):
    if dataset_name == 'multirc':
        TRAIN_DATA_DIR = '/scratch/datasets/imagenet/train'
        VAL_DATA_DIR = '/scratch/datasets/imagenet/val'
        if split == 'train':
            dataset = load_dataset('eraser_multi_rc', split='train')
        elif split == 'val':
            dataset = load_dataset('eraser_multi_rc', split='validation')
        elif split == 'train_val':
            dataset = load_dataset('eraser_multi_rc', split='train')
        elif split == 'test':
            dataset = load_dataset('eraser_multi_rc', split='test')
        else:
            raise ValueError(f'split {split} not recognized')
    else:
        raise ValueError(f'dataset {dataset_name} is not the dataset')
    # train data is always augmented
    # import pdb; pdb.set_trace()
    if not raw_data:
        dataset = dataset.map(lambda x: transform(x, is_training=(split=='train'), processor=processor), batched=True,
                                remove_columns=['passage', 
                                                'query_and_answer',
                                                'evidences'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    return DatasetOutput(dataset, dataloader)
