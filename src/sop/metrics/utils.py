from scipy.stats import pearsonr, spearmanr
import torch

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

