import torch
from exlib.explainers.common import patch_segmenter


def get_explainer(original_model, backbone_model, explainer_name, device, num_samples=20, 
                  num_patches=11, return_params=False):
    # explainer_name = 'lime'
    # original_model is a wrapped model that has .model as the actual model
    # and the output is just the logits
    def segmenter(x):
        # num_patches = 11 #14
        return patch_segmenter(x, sz=(num_patches, num_patches))
    param_str = ''
    if explainer_name == 'lime': # checked
        from exlib.explainers.lime import LimeImageCls
        eik = {
            "segmentation_fn": lambda x: patch_segmenter(x, sz=(11, 11)),
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
    elif explainer_name == 'shap': # checked
        from exlib.explainers import ShapImageCls
        sek = {'max_evals': num_samples}
        explainer = ShapImageCls(original_model,
                    shap_explainer_kwargs=sek)
        param_str = f'n{num_samples}'
    elif explainer_name == 'rise': # checked
        from exlib.explainers import RiseImageCls
        explainer = RiseImageCls(original_model, input_size=(66, 66), s=11, N=num_samples)
        param_str = f'n{num_samples}'
    elif explainer_name == 'intgrad': # checked
        from exlib.explainers import IntGradImageCls
        explainer = IntGradImageCls(original_model, num_steps=num_samples)
        param_str = f'n{num_samples}'
    elif explainer_name == 'gradcam': # checked
        def reshape_transform(tensor):
            return tensor
        from exlib.explainers import GradCAMImageCls
        explainer = GradCAMImageCls(original_model, [original_model.model.conv3], 
                                    reshape_transform=reshape_transform).to(device)
        param_str = f'original_model.model.conv3'
    elif explainer_name == 'archipelago': # checked
        from exlib.explainers.archipelago import ArchipelagoImageCls
        explainer = ArchipelagoImageCls(original_model, segmenter=segmenter)
        # explainer = ArchipelagoImageCls(original_model)
        param_str = f'spatch7'
    elif explainer_name == 'fullgrad': # checked
        from exlib.explainers.fullgrad import FullGradImageCls
        explainer = FullGradImageCls(original_model, im_size=(1, 66, 66), model_type='cnn')
    elif explainer_name == 'attn': # shouldn't work with non transformer
        from exlib.explainers.attn import AttnImageCls
        explainer = AttnImageCls(backbone_model)
    elif explainer_name == 'mfaba': # checked
        from exlib.explainers.mfaba import MfabaImageCls
        mfaba_args = {
            'use_softmax': False,
        }
        explainer = MfabaImageCls(original_model, mfaba_args=mfaba_args)
    elif explainer_name == 'agi': # currently only have implementation for classification
        from exlib.explainers.agi import AgiImageCls
        max_iter = 4
        topk = 5
        explainer = AgiImageCls(original_model, max_iter=max_iter, topk=topk, pred_mode='reg')
        param_str = f'n{max_iter}_k{topk}'
    elif explainer_name == 'ampe': # checked
        from exlib.explainers.ampe import AmpeImageCls
        N = 5
        num_steps = 4
        explainer = AmpeImageCls(original_model, N=N, num_steps=num_steps, use_softmax=False)
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
    elif explainer_name == 'bagnet': # we should probably train bagnet for cosmogrid
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