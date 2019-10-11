"""Model creation utilities for DeepLab V3+.

Attributes:
    DEFAULT_CONFIG (dict):
        The default configuration for the model. The keys are identical
        to the constructor arguments.
"""
import json

from abyss_deep_learning.keras.zoo.segmentation.deeplab_v3_plus.model import Deeplabv3
from abyss_deep_learning.utils import append_docstring

DEFAULT_CONFIG = {
    "activation": 'softmax',
    "alpha": 1.0,
    "backbone": "mobilenetv2",
    "classes": 21,
    "input_shape": (512, 512, 3),
    "input_tensor": None,
    "OS": 16,  # Output stride
    "weights": 'pascal_voc',
}


def _model_specific_construction(model, config):
    """Performs model specific construction.

    Args:
        model (TYPE): Description
        config (TYPE): Description
    """
    pass


@append_docstring(Deeplabv3)
def make_model(**kwargs):
    """Make a model with new the heads and run any model specific construction."""
    config = DEFAULT_CONFIG if not kwargs else kwargs
    model = Deeplabv3(**config)
    _model_specific_construction(model, config)
    return model


@append_docstring(Deeplabv3)
def from_config(config):
    """Make model from JSON file or dict and reinitialise the heads.

    Args:
        config (str or dict): Path to JSON to load, or loaded JSON dict.

    Returns:
        tf.keras.model.Model: The initialised model with new heads.

    Raises:
        ValueError: Description

    Config values come from following docstring arguments:
    """
    if isinstance(config, str):
        config = json.read(config)
    elif not isinstance(config, dict):
        raise ValueError("Expected `config` to be path or dict.")
    model = make_model(**config)
    return model
