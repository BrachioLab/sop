from .general_utils import *
from exlib.modules.sop import convert_idx_masks_to_bool

def fidelity(expln, probs):
    if 'group_attributions' in expln._fields:
        # group_fidelity
        attributions = expln.group_attributions
    else:
        # single_pixel_fidelity
        attributions = expln.attributions

    # import pdb; pdb.set_trace()
    try:
        bsz = attributions.shape[0]
        num_classes = attributions.shape[-1]
        aggr_pred = attributions.view(bsz, -1, num_classes).sum(1)
    except: # archipleago: attributions is a list
        bsz = len(attributions)
        num_classes = len(attributions[0])
        aggr_pred = torch.tensor([[aa.sum() for aa in a] for a in attributions], device=attributions[0][0].device)
    aggr_probs = aggr_pred.softmax(-1)
    return kl_div(probs, aggr_probs, reduction='none').sum(-1)

def get_entropy(masks, segs, reduction='mean', eps=1e-20):
    """
    masks (M, H, W)
    segs (1, H, W)
    """
    segs_bool = convert_idx_masks_to_bool(segs)
        
    intersection = masks[:,None] * segs_bool[None]
    ratios = (intersection.sum(-1).sum(-1) + eps) / \
            (masks.sum(-1).sum(-1)[:,None] + eps * intersection.shape[0])
    entropy = - (ratios * torch.log2(ratios)).sum(1)
    if reduction == 'mean':
        entropy = entropy.mean()
    return entropy

def get_prob_obj(masks, segs, reduction='mean', eps=1e-20):
    """
    masks (M, H, W)
    segs (1, H, W)
    """
    segs_bool = (segs == 1).float()
    # print(segs_bool.shape)
    # segs_bool = convert_idx_masks_to_bool(segs)
    intersection = masks[:,None] * segs_bool
    ratios = (intersection.sum(-1).sum(-1) + eps) / \
            (masks.sum(-1).sum(-1)[:,None] + eps * intersection.shape[0])
    return ratios

def get_prob_obj_coverage(masks, segs, reduction='mean', eps=1e-20):
    """
    masks (M, H, W)
    segs (1, H, W)
    """
    segs_bool = (segs == 1).float()
    intersection = masks[:,None] * segs_bool
    ratios = (intersection.sum(-1).sum(-1) + eps) / \
            (segs_bool.sum(-1).sum(-1)[:,None] + eps * intersection.shape[0])
    return ratios

def get_iou(masks, segs, reduction='mean', eps=1e-20):
    """
    masks (M, H, W)
    segs (1, H, W)
    """
    segs_bool = (segs == 1).float()
    intersection = masks[:,None] * segs_bool
    union = (masks[:,None] + segs_bool).bool().float()
    ratios = (intersection.sum(-1).sum(-1) + eps) / \
            (segs_bool.sum(-1).sum(-1)[:,None] + eps * intersection.shape[0])
    return ratios