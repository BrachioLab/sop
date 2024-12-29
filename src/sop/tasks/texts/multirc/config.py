from ...base_task import BaseConfig

class MultiRcConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__()
        self._config = {
            'dataset': {
                'name': 'multirc',
                'root': 'eraser_multi_rc'
            },
            'training': {
                'batch_size': 16,
                'num_epochs': 20,
                'mask_batch_size': 64,
                'optimizer': {
                    'name': 'adamw',
                    'lr': 0.000005,
                    'weight_decay': 0.01
                },
            },
            'evaluation': {
                'val': {
                    'split': 'val',
                    'num_data': -1,
                    'batch_size': 16,
                },
                'train_val': {
                    'split': 'train',
                    'num_data': 1,
                    'batch_size': 16
                },
                'val_sm': {
                    'split': 'val',
                    'num_data': 100,
                    'batch_size': 1
                },
                'test': {
                    'split': 'test',
                    'num_data': -1,
                    'batch_size': 1
                },
                'test_sm': {
                    'split': 'test',
                    'num_data': 100,
                    'batch_size': 1
                },
            },
            'model': {
                'type': 'bert',
                'base': '/shared_data0/weiqiuy/sop/pt_models/multirc_vanilla/best',
                'base_processor': 'bert-base-uncased',
                'sop': '/shared_data0/weiqiuy/sop/notebooks/exps/multirc_lr5e-06_tgtnnz0.2_gg8.0000_gs2.0000_fz_aug_blur15.0000/best',
                'num_classes': 2,
                'base_exp': '/shared_data0/weiqiuy/sop/exps/multirc_bert'
            }
        }

        # update config based on args and kwargs
        for key, value in kwargs.items():
            if key in self._config:
                self._config[key].update(value)
            else:
                raise ValueError(f'key {key} not found in config')

    def get_config(self, split='val', **kwargs):
        # get config based on split etc
        config = self._config.copy()
        config.update(kwargs)
        # return based on split, e.g. if split is val, then for num data should only return val's num_data
        # but also return other values that are not split dependent
        # e.g. if split is val, then return val's num_data, but also return training's batch_size
        split_config = {}
        # if key not evaluation, then just copy over
        # else if key is evaluation, then only copy over if split is the same
        for key, value in config.items():
            if key != 'evaluation':
                split_config[key] = value
            else:
                if split in value:
                    split_config[key] = value[split]
        return split_config

