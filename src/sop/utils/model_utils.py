def set_grad_model(model, module_names, verbose=False):
    for name, param in model.named_parameters():
        if any([module_name in name for module_name in module_names]):
            param.requires_grad = True
        # else:
        #     param.requires_grad = False
    if verbose:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'Parameter: {name} - Requires gradients')

# Usage:
# from sop.utils import set_grad_model

# set_grad_model(model, [], verbose=True)