# sop/configs/__init__.py

# Import and expose the configuration classes
from .config_base import BaseConfig
from .config_imagenet import ImageNetConfig
from .config_cosmogrid import CosmogridConfig

__all__ = ['BaseConfig', 'ImageNetConfig', 'CosmogridConfig']  # Define the public interface