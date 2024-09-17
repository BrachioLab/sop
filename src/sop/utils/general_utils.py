import torch
import numpy as np
import random
import torch
from scipy.stats import pearsonr, spearmanr

def seed_all(SEED):
    if SEED != -1:
        # Torch RNG
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # Python RNG
        np.random.seed(SEED)
        random.seed(SEED)

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
