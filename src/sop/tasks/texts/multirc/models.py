from ..base.models import *
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
# from exlib.modules.sop import SOPConfig, get_chained_attr
# import torch.nn as nn
# from exlib.modules.sop_text import SOPTextCls

# from collections import namedtuple
# import os
# import torch



# WrappedBackboneOutput = namedtuple("WrappedBackboneOutput", 
#                                   ["logits",
#                                    "pooler_output",
#                                    "hidden_states"])


# class WrappedModel(nn.Module):
#     def __init__(self, model, output_type='tuple', num_patch=14):
#         super().__init__()
#         assert output_type in ['tuple', 'logits', 'hidden_states', 'probs']
#         self.model = model
#         self.output_type = output_type
#         self.num_patch = num_patch
    
#     def forward(self, inputs=None, **kwargs):
#         # if 'output_hidden_states' not in kwargs:
#         kwargs['output_hidden_states'] = True

#         if inputs is not None:
#             outputs = self.model(inputs, **kwargs)
#         else:
#             outputs = self.model(**kwargs)
#         # if 'inputs' in kwargs:
#         #     input_ids = kwargs['inputs']
#         #     kwargs_new = {}
#         #     for k, v in kwargs.items():
#         #         if k != 'inputs':
#         #             kwargs_new[k] = v
#         #     print('kwargs_new', kwargs_new)
#         # else:
#         #     outputs = self.model(**kwargs)
#         # else:
#         #     outputs = self.model(**kwargs)
#         if self.output_type == 'tuple':
#             return WrappedBackboneOutput(outputs.logits, outputs.hidden_states[-1][:,0],
#                    outputs.hidden_states)
#         elif self.output_type == 'logits':
#             return outputs.logits
#         elif self.output_type == 'probs':
#             return outputs.logits.softmax(-1)
#         else: # hidden_states
#             # import pdb; pdb.set_trace()
#             return outputs.hidden_states[-2]

# def get_wrapped_models(model, config):
#     wrapped_model = WrappedModel(model, output_type='tuple')
#     class_weights = get_chained_attr(wrapped_model, config.finetune_layers[0]).weight #.clone().to(device)
#     projection_layer = WrappedModel(wrapped_model, output_type='hidden_states')
#     return wrapped_model, class_weights, projection_layer

# def get_wrapped_model(model, config):
#     wrapped_model = WrappedModel(model, output_type='logits')
#     return wrapped_model

# def get_model(
#     model_type='bert', 
#     backbone_model_name='bert-base-uncased',
#     backbone_processor_name='bert-base-uncased',
#     sop_model_name='/shared_data0/weiqiuy/sop/notebooks/exps/multirc_lr5e-06_tgtnnz0.2_gg8.0000_gs2.0000_fz_aug_blur15.0000/best',
#     eval_mode=False,
#     ):
#     if model_type == 'bert':
#         backbone_model = AutoModelForSequenceClassification.from_pretrained(backbone_model_name)
#         processor = AutoTokenizer.from_pretrained(backbone_processor_name)
#         backbone_config = AutoConfig.from_pretrained(backbone_model_name)
#         original_model = WrappedModel(backbone_model, output_type='logits')
#         original_model_softmax = WrappedModel(backbone_model, output_type='probs')
#         config = SOPConfig(os.path.join(sop_model_name, 'config.json'))
#         config.__dict__.update(backbone_config.__dict__)
#         config.num_labels = len(backbone_config.id2label)
#         wrapped_backbone_model, class_weights, projection_layer = get_wrapped_models(
#             backbone_model,
#             config
#         )
#         model = SOPTextCls(config, wrapped_backbone_model, 
#                             class_weights=class_weights, 
#                             projection_layer=projection_layer)
#         if sop_model_name is not None:
#             state_dict = torch.load(os.path.join(sop_model_name, 
#                                             'checkpoint.pth'))
#             print('Loaded step', state_dict['step'])
#             model.load_state_dict(state_dict['model']) #, strict=False)
#     else:
#         raise ValueError(f'Unsupported model type: {model_type}')

#     if eval_mode:
#         backbone_model.eval()
#         original_model.eval()
#         original_model_softmax.eval()
#         projection_layer.eval()
#         model.eval()

#     return backbone_model, original_model, original_model_softmax, projection_layer, processor, backbone_config, model, config




# # original_model = WrappedModel(backbone_model)
# # original_model = original_model.to(device)