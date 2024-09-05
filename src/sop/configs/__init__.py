# sop/configs/__init__.py

# Import and expose the configuration classes
from .config_base import BaseConfig
from .config_imagenet import ImageNetConfig

__all__ = ['BaseConfig', 'ImageNetConfig']  # Define the public interface