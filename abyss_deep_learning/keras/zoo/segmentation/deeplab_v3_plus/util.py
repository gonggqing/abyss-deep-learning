"""Utilities for DeepLab V3+

Attributes:
    DEFAULT_CONFIG (TYPE): Description
"""
import json

from keras.models import model_from_json

from abyss_deep_learning.keras.zoo.segmentation.deeplab_v3_plus.model import Deeplabv3


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


def make_model(**kwargs):
    """Make a model and reinitialise the heads.

    Args:
        **kwargs: dict which shares the model

    Returns:
        TYPE: Description
    """
    config = DEFAULT_CONFIG if not kwargs else kwargs
    print(config)
    model = Deeplabv3(**config)
    _model_specific_construction(model, config)
    return model


make_model.__doc__ = Deeplabv3.__doc__
# + "\n\n" + make_model.__doc__


def from_config(config):
    """Make model from JSON file or dict and reinitialise the heads.

    Args:
        config (str or dict): Path to JSON to load, or loaded JSON dict.

    Returns:
        keras.Model: The initialised model with new heads.

    Raises:
        ValueError: Description
    """
    # where config specifies input_shape, ..., regularization?
    if isinstance(config, str):
        config = json.read(config)
    elif not isinstance(config, dict):
        raise ValueError("Expected `config` to be path or dict.")
    model = make_model(**config)
    return model


# should it be totally generic and thus be in keras.models.py?
def load(model_definition, model_weights=None):
    """Summary

    Args:
        model_definition (str): Path to the model definitions.
        model_weights (str): Path to the model weights.

    Returns:
        keras.Model: The loaded model.
    """
    model = model_from_json(model_definition)
    if model_weights:
        model.load_weights(model_weights, by_name=True)
    return model


# should it be totally generic and thus be in keras.models.py?
def save(model, model_definition, model_weights):
    """Save the model definition and weights to files.

    Args:
        model (keras.Model): The model to save.
        model_definition (str): Path to save the model definition.
        model_weights (str): Path to save the model weights.

    """
    model_json = model.to_json()
    with open(model_definition, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_weights)
