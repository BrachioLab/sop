from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from exlib.modules.sop import SOPConfig, get_chained_attr
import torch.nn as nn
import torch

from exlib.modules.sop import SOPImageCls4

from collections import namedtuple
import os

class ModelOutput:
    def __init__(self, logits, pooler_output, hidden_states):
        self.logits = logits
        self.pooler_output = pooler_output
        self.hidden_states = hidden_states
        
class CNNModel(nn.Module):
    def __init__(self, output_num):
        super(CNNModel, self).__init__()
        
        # self.normalization = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4)
        self.relu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.relu2 = nn.LeakyReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=4)
        self.relu3 = nn.LeakyReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1200, 128)
        self.relu4 = nn.LeakyReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu6 = nn.LeakyReLU()
        self.fc4 = nn.Linear(32, output_num)
        
    def forward(self, x, output_hidden_states=False):
        # x = self.normalization(x)
        hidden_states = []
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        if output_hidden_states:
            hidden_states.append(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        if output_hidden_states:
            hidden_states.append(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        if output_hidden_states:
            hidden_states.append(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        pooler_output = self.relu6(x)
        logits = self.fc4(pooler_output)
        # for i, hs in enumerate(hidden_states):
        #     print(hs.shape)
        if output_hidden_states:
            return ModelOutput(logits=logits,
                               pooler_output=pooler_output,
                               hidden_states=hidden_states)
        else:
            return ModelOutput(logits=logits,
                               pooler_output=pooler_output,
                               hidden_states=None)


WrappedBackboneOutput = namedtuple("WrappedBackboneOutput", 
                                  ["logits",
                                   "pooler_output"])


class WrappedModel(nn.Module):
    def __init__(self, model, output_type='tuple', num_patch=14, layer=-1):
        super().__init__()
        assert output_type in ['tuple', 'logits', 'hidden_states']
        self.model = model
        self.output_type = output_type
        self.num_patch = num_patch
        self.layer = layer
    
    def forward(self, inputs):
        if self.output_type == 'tuple':
            outputs = self.model(inputs)
            return WrappedBackboneOutput(outputs.logits, outputs.pooler_output)
        elif self.output_type == 'logits':
            outputs = self.model(inputs)
            return outputs.logits
        else: # hidden_states
            # import pdb; pdb.set_trace()
            outputs = self.model(inputs, output_hidden_states=True)
            return outputs.hidden_states[self.layer]

def get_wrapped_models(model, config):
    wrapped_model = WrappedModel(model, output_type='tuple')
    class_weights = get_chained_attr(wrapped_model, config.finetune_layers[0]).weight #.clone().to(device)
    projection_layer = WrappedModel(model, output_type='hidden_states')
    return wrapped_model, class_weights, projection_layer

def get_wrapped_model(model, config):
    wrapped_model = WrappedModel(model, output_type='logits')
    return wrapped_model


def get_model(
    model_type='cnn', 
    backbone_model_name='/scratch/datasets/cosmogrid/CNN_mass_maps.pth',
    backbone_processor_name='/scratch/datasets/cosmogrid/CNN_mass_maps.pth',
    sop_model_name='/shared_data0/weiqiuy/sop/notebooks/exps/cosmogrid_lr5e-06_tgtnnz0.2_gg0.0600_gs10.0000_ft_identify_fixk_scratch_ks1_segpatch_4h/best',
    backbone_layer=-2,
    eval_mode=False,
    ):
    # sop model
    if sop_model_name is not None:
        config = SOPConfig(os.path.join(sop_model_name, 'config.json'))
    else:
        config = SOPConfig(
            hidden_size=32,
            num_labels=2,
            input_hidden_size=input_hidden_size,
            attn_patch_size=6,
            num_heads=num_heads,
            num_masks_sample=20,
            num_masks_max=200,
            image_size=66,
            num_channels=1,
            finetune_layers=['model.fc4'], # rename to classifier layer
            group_gen_scale=0.06,#0.00027,
            group_sel_scale=10,
            group_gen_temp_alpha=1,
            group_gen_temp_beta=1,
            group_gen_blur_ks1=1,
            group_gen_blur_sigma1=1,
            group_gen_blur_ks2=-1,
            group_gen_blur_sigma2=-1,
            projected_input_scale=1
        )

    config.finetune_layers = ['model.fc4']

    # backbone model
    backbone_model = CNNModel(config.num_labels)
    state_dict = torch.load(backbone_model_name)
    backbone_model.load_state_dict(state_dict=state_dict)
    processor = None
    backbone_config = None

    original_model = WrappedModel(backbone_model, output_type='logits')

    bbl_ihs_mapping = {
        -1: 48,
        -2: 32,
        -3: 16
    }
    input_hidden_size = bbl_ihs_mapping[backbone_layer]

    wrapped_backbone_model, class_weights, projection_layer = get_wrapped_models(
        backbone_model,
        config
    )

    model = SOPImageCls4(config, wrapped_backbone_model, 
                        class_weights=class_weights, 
                        projection_layer=projection_layer)
    if sop_model_name is not None:
        state_dict = torch.load(os.path.join(sop_model_name, 
                                            'checkpoint.pth'))
        print('Loaded step', state_dict['step'])
        model.load_state_dict(state_dict['model']) #, strict=False)
    if eval_mode:
        backbone_model.eval()
        original_model.eval()
        model.eval()

    return backbone_model, original_model, processor, backbone_config, model, config

    # return backbone_model, processor, backbone_config



# original_model = WrappedModel(backbone_model)
# original_model = original_model.to(device)