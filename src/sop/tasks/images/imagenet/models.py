from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from exlib.modules.sop import SOPConfig, get_chained_attr
import torch.nn as nn
from exlib.modules.sop import SOPImageCls4

from collections import namedtuple
import os
import torch



WrappedBackboneOutput = namedtuple("WrappedBackboneOutput", 
                                  ["logits",
                                   "pooler_output",
                                   "hidden_states"])


class WrappedModel(nn.Module):
    def __init__(self, model, output_type='tuple', num_patch=14):
        super().__init__()
        assert output_type in ['tuple', 'logits', 'hidden_states']
        self.model = model
        self.output_type = output_type
        self.num_patch = num_patch
    
    def forward(self, inputs, **kwargs):
        kwargs['output_hidden_states'] = True
        outputs = self.model(inputs, **kwargs)
        if self.output_type == 'tuple':
            return WrappedBackboneOutput(outputs.logits, outputs.hidden_states[-1][:,0],
                                         outputs.hidden_states)
        elif self.output_type == 'logits':
            return outputs.logits
        else: # hidden_states
            # import pdb; pdb.set_trace()
            hidden_size = outputs.hidden_states[-2].shape[-1]
            return outputs.hidden_states[-2][:,1:].transpose(1,2).reshape(-1, 
                            hidden_size, self.num_patch, self.num_patch)

def get_wrapped_models(model, config, wrap_proj=False):
    wrapped_model = WrappedModel(model, output_type='tuple')
    class_weights = get_chained_attr(wrapped_model, config.finetune_layers[0]).weight #.clone().to(device)
    if wrap_proj:
        projection_layer = WrappedModel(wrapped_model, output_type='hidden_states')
    else:
        projection_layer = WrappedModel(model, output_type='hidden_states')
    return wrapped_model, class_weights, projection_layer

def get_wrapped_model(model, config):
    wrapped_model = WrappedModel(model, output_type='logits')
    return wrapped_model

def get_model(
    model_type='vit', 
    backbone_model_name='google/vit-base-patch16-224',
    backbone_processor_name='google/vit-base-patch16-224',
    sop_model_name='/shared_data0/weiqiuy/sop/exps/imagenet_lr5e-06_tgtnnz0.2_gg0.0600_gs0.0100_ft_identify_fixk_scratch_ks3/best',
    eval_mode=False,
    wrap_proj=False
    ):
    if model_type == 'vit':
        backbone_model = AutoModelForImageClassification.from_pretrained(backbone_model_name)
        processor = AutoImageProcessor.from_pretrained(backbone_processor_name)
        backbone_config = AutoConfig.from_pretrained(backbone_model_name)
        original_model = WrappedModel(backbone_model, output_type='logits')
        config = SOPConfig(os.path.join(sop_model_name, 'config.json'))
        config.__dict__.update(backbone_config.__dict__)
        config.num_labels = len(backbone_config.id2label)
        wrapped_backbone_model, class_weights, projection_layer = get_wrapped_models(
            backbone_model,
            config,
            wrap_proj=wrap_proj
        )
        model = SOPImageCls4(config, wrapped_backbone_model, 
                            class_weights=class_weights, 
                            projection_layer=projection_layer)
        if sop_model_name is not None:
            state_dict = torch.load(os.path.join(sop_model_name, 
                                            'checkpoint.pth'))
            print('Loaded step', state_dict['step'])
            model.load_state_dict(state_dict['model']) #, strict=False)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    if eval_mode:
        backbone_model.eval()
        original_model.eval()
        model.eval()

    return backbone_model, original_model, processor, backbone_config, model, config




# original_model = WrappedModel(backbone_model)
# original_model = original_model.to(device)