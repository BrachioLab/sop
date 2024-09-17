# sop/__init__.py

# Import the config module from sop.configs
from .configs import config_base, config_imagenet
from . import metrics

# Optionally, you can alias the classes for convenience
BaseConfig = config_base.BaseConfig
ImageNetConfig = config_imagenet.ImageNetConfig

# You can also expose utils if needed
from .utils import data_utils
from . import user_study