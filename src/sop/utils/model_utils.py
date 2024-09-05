from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from exlib.modules.sop import SOPConfig, get_chained_attr
import torch.nn as nn

from collections import namedtuple

WrappedBackboneOutput = namedtuple("WrappedBackboneOutput", 
                                  ["logits",
                                   "pooler_output"])


class WrappedBackboneModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, inputs):
        outputs = self.model(inputs, output_hidden_states=True)
        return WrappedBackboneOutput(outputs.logits, outputs.hidden_states[-1][:,0])
        
class ProjectionLayerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_patch = 14
        
    def forward(self, x):
        return self.model.model(x, output_hidden_states=True).hidden_states[-2][:,1:].transpose(1,2).reshape(-1, 
                            config.hidden_size, self.num_patch, self.num_patch)

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
    wrapped_model = WrappedBackboneModel(model)
    wrapped_model = wrapped_model
    class_weights = get_chained_attr(wrapped_model, config.finetune_layers[0]).weight #.clone().to(device)
    projection_layer = ProjectionLayerWrapper(wrapped_model)
    return wrapped_model, class_weights, projection_layer

# wrapped_backbone_model = WrappedBackboneModel(backbone_model)
# wrapped_backbone_model = wrapped_backbone_model.to(device)
# class_weights = get_chained_attr(wrapped_backbone_model, config.finetune_layers[0]).weight #.clone().to(device)
    
# projection_layer = ProjectionLayerWrapper(wrapped_backbone_model)