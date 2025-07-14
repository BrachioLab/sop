from scipy.stats import pearsonr, spearmanr
import torch
from exlib.modules.sop import convert_idx_masks_to_bool


# spearman_correlation
def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp).to(x.device)
    ranks[tmp] = torch.arange(len(x)).to(x.device)
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank.to(x.device)).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

def pearson_correlation(x: torch.Tensor, y: torch.Tensor):
    return pearsonr(x.detach().cpu().numpy(), y.detach().cpu().numpy()).statistic

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    return spearmanr(x.detach().cpu().numpy(), y.detach().cpu().numpy()).statistic

def kl_div(probs_output, probs_target, reduction='none'):
    return torch.nn.functional.kl_div(torch.log(probs_output), probs_target, reduction=reduction)

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

## Purity
def get_entropy_text(masks, segs, reduction='mean', eps=1e-20):
    """
    masks (M, L)
    segs (1, L)
    """
    # segs_bool = convert_idx_masks_to_bool(segs)
    # segs_bool = segs
    # print(len(segs.view(-1).tolist()))
    segs_bool = torch.nn.functional.one_hot(segs, num_classes=len(set(segs.view(-1).tolist()))).to(torch.bool)

    # Adjust dimensions to match the required output size (1, M, L)
    segs_bool = segs_bool.transpose(1, 2)

        
    intersection = masks[:,None] * segs_bool[None]
    ratios = (intersection.sum(-1) + eps) / \
            (masks.sum(-1)[:,None] + eps * intersection.shape[0])
    entropy = - (ratios * torch.log2(ratios)).sum(1)
    if reduction == 'mean':
        entropy = entropy.mean()
    return entropy

def get_prob_obj_text(masks, segs, reduction='mean', eps=1e-20):
    """
    masks (M, L)
    segs (1, L)
    """
    segs_bool = (segs == 1).float()
    # print(segs_bool.shape)
    # segs_bool = convert_idx_masks_to_bool(segs)
    intersection = masks[:,None] * segs_bool
    ratios = (intersection.sum(-1) + eps) / \
            (masks.sum(-1)[:,None] + eps * intersection.shape[0])
    if reduction == 'mean':
        ratios = ratios.mean()
    return ratios

def get_prob_obj_coverage_text(masks, segs, reduction='mean', eps=1e-20):
    """
    masks (M, L)
    segs (1, L)
    """
    segs_bool = (segs == 1).float()
    intersection = masks[:,None] * segs_bool
    ratios = (intersection.sum(-1) + eps) / \
            (segs_bool.sum(-1)[:,None] + eps * intersection.shape[0])
    if reduction == 'mean':
        ratios = ratios.mean()
    return ratios

def get_iou_text(masks, segs, reduction='mean', eps=1e-20):
    """
    Compute Intersection over Union (IoU) between masks and segmentation.
    
    Args:
    masks (M, L): Tensor containing M masks of length L.
    segs (1, L): Tensor containing 1 segmentation of length L.
    
    Keyword Args:
    reduction (str): Method for reducing IoU scores across masks ('none', 'mean', 'sum').
    eps (float): Small value to avoid division by zero.
    
    Returns:
    torch.Tensor: Tensor of IoU scores.
    """
    masks = masks.to(torch.bool)
    # Convert segmentation to boolean tensor as one-hot encoding
    segs_bool = torch.nn.functional.one_hot(segs.squeeze(0), num_classes=2).to(torch.bool)
    segs_bool = segs_bool[:, 1]  # Assuming class '1' is the foreground

    # Compute intersections and unions
    intersection = (masks & segs_bool).sum(dim=1)
    union = (masks | segs_bool).sum(dim=1)
    
    # Calculate IoU
    iou = (intersection + eps) / (union + eps)

    # Reduction
    if reduction == 'mean':
        iou = iou.mean()
    elif reduction == 'sum':
        iou = iou.sum()

    return iou

