from .general_utils import *

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

