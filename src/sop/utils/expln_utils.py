from exlib.explainers.archipelago import ArchipelagoImageCls
from exlib.explainers.lime import LimeImageCls
from exlib.explainers.common import patch_segmenter
from exlib.explainers import ShapImageCls
from exlib.explainers import RiseImageCls
from exlib.explainers import IntGradImageCls
from exlib.explainers import GradCAMImageCls
from exlib.explainers.fullgrad import FullGradImageCls
from exlib.explainers.attn import AttnImageCls
import torch


def get_explainer(original_model, backbone_model, explainer_name, device, num_samples=20, 
                  num_patches=7, return_params=False):
    # explainer_name = 'lime'
    # original_model is a wrapped model that has .model as the actual model
    # and the output is just the logits
    def segmenter(x):
        num_patches = 7 #14
        return patch_segmenter(x, sz=(num_patches, num_patches))
    param_str = ''
    if explainer_name == 'lime':
        eik = {
            "segmentation_fn": patch_segmenter,
            "top_labels": 1000, 
            "hide_color": 0, 
            "num_samples": num_samples
        }
        gimk = {
            "positive_only": False
        }
        explainer = LimeImageCls(original_model, 
                            explain_instance_kwargs=eik, 
                            get_image_and_mask_kwargs=gimk)
        param_str = f'n{num_samples}'
    elif explainer_name == 'shap':
        sek = {'max_evals': num_samples}
        explainer = ShapImageCls(original_model,
                    shap_explainer_kwargs=sek)
        param_str = f'n{num_samples}'
    elif explainer_name == 'rise':
        explainer = RiseImageCls(original_model, N=num_samples)
        param_str = f'n{num_samples}'
    elif explainer_name == 'intgrad':
        explainer = IntGradImageCls(original_model, num_steps=num_samples)
        param_str = f'n{num_samples}'
    elif explainer_name == 'gradcam':
        explainer = GradCAMImageCls(original_model, 
                                    [original_model.model.vit.encoder.layer[-1].layernorm_before])
        param_str = f'loriginal_model.model.vit.encoder.layer[-1].layernorm_before'
    elif explainer_name == 'archipelago':
        explainer = ArchipelagoImageCls(original_model, segmenter=segmenter)
        # explainer = ArchipelagoImageCls(original_model)
        param_str = f'spatch7'
    elif explainer_name == 'fullgrad':
        explainer = FullGradImageCls(original_model)
    elif explainer_name == 'attn':
        explainer = AttnImageCls(backbone_model)
    elif explainer_name == 'mfaba':
        from exlib.explainers.mfaba import MfabaImageCls
        explainer = MfabaImageCls(original_model)
    elif explainer_name == 'agi':
        from exlib.explainers.agi import AgiImageCls
        max_iter = 4
        topk = 5
        explainer = AgiImageCls(original_model, max_iter=max_iter, topk=topk)
        param_str = f'n{max_iter}_k{topk}'
    elif explainer_name == 'ampe':
        from exlib.explainers.ampe import AmpeImageCls
        N = 5
        num_steps = 4
        explainer = AmpeImageCls(original_model, N=N, num_steps=num_steps)
        param_str = f'n{num_steps}_k{N}'
    elif explainer_name == 'bcos':
        from exlib.modules.bcos import BCos
        model = BCos()
        explainer = model
    elif explainer_name == 'xdnn':
        from exlib.modules.xdnn import XDNN
        xdnn_name = 'xfixup_resnet50'
        model = XDNN(xdnn_name, 
             '/shared_data0/weiqiuy/github/fast-axiomatic-attribution/pt_models/xfixup_resnet50_model_best.pth.tar').to(device)
        explainer = model
        param_str = f'm{xdnn_name}'
    elif explainer_name == 'bagnet':
        from exlib.modules.bagnet import BagNet
        model = BagNet()
        explainer = model
    else:
        raise ValueError('Invalid explainer name' + explainer_name)
    explainer = explainer.to(device)

    if return_params:
        return explainer, param_str
    return explainer


def get_expln_all_classes(original_model, inputs, explainer, num_classes):
    bsz, num_channels, H, W = inputs.shape
    
    logits = original_model(inputs)
    preds = torch.argmax(logits, dim=-1)

    multi_labels = torch.tensor([list(range(num_classes))]).to(inputs.device).expand(bsz, num_classes)
    
    expln = explainer(inputs.clone(), multi_labels.clone(), return_groups=True)

    if 'logits' in expln._fields:
        logits = expln.logits # use the logits of the explainer if it's faithful model

    probs = logits[:, :num_classes].softmax(-1)
    
    return expln, probs
