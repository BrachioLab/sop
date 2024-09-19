from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from exlib.modules.sop import SOPConfig, get_chained_attr
import torch.nn as nn

from collections import namedtuple

WrappedBackboneOutput = namedtuple("WrappedBackboneOutput", 
                                  ["logits",
                                   "pooler_output"])


class WrappedModel(nn.Module):
    def __init__(self, model, output_type='tuple', num_patch=14):
        super().__init__()
        assert output_type in ['tuple', 'logits', 'hidden_states']
        self.model = model
        self.output_type = output_type
        self.num_patch = num_patch
    
    def forward(self, inputs):
        outputs = self.model(inputs, output_hidden_states=True)
        if self.output_type == 'tuple':
            return WrappedBackboneOutput(outputs.logits, outputs.hidden_states[-1][:,0])
        elif self.output_type == 'logits':
            return outputs.logits
        else: # hidden_states
            # import pdb; pdb.set_trace()
            hidden_size = outputs.hidden_states[-2].shape[-1]
            return outputs.hidden_states[-2][:,1:].transpose(1,2).reshape(-1, 
                            hidden_size, self.num_patch, self.num_patch)


def get_model(
    model_type='vit', 
    backbone_model_name='google/vit-base-patch16-224',
    backbone_processor_name='google/vit-base-patch16-224'
    ):
    if model_type == 'vit':
        backbone_model = AutoModelForImageClassification.from_pretrained(backbone_model_name)
        processor = AutoImageProcessor.from_pretrained(backbone_processor_name)
        backbone_config = AutoConfig.from_pretrained(backbone_model_name)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    return backbone_model, processor, backbone_config


def get_wrapped_models(model, config):
    wrapped_model = WrappedModel(model, output_type='tuple')
    class_weights = get_chained_attr(wrapped_model, config.finetune_layers[0]).weight #.clone().to(device)
    projection_layer = WrappedModel(model, output_type='hidden_states')
    return wrapped_model, class_weights, projection_layer

def get_wrapped_model(model, config):
    wrapped_model = WrappedModel(model, output_type='logits')
    return wrapped_model

# original_model = WrappedModel(backbone_model)
# original_model = original_model.to(device)