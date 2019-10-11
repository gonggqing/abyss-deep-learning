import json

from imgaug import augmenters as iaa


class ScriptAugmenter:
    def __init__(self, **kwargs):
        raise NotImplementedError("TODO: ScriptAugmenter")


class EvalAugmenter:
    def __init__(self, **kwargs):
        raise NotImplementedError("TODO: EvalAugmenter")


def __make_aug_map():
    augmenters = {}
    for attr_name in dir(iaa):
        attr = getattr(iaa, attr_name)
        try:
            if not issubclass(attr, iaa.meta.Augmenter):
                continue
            augmenters[attr_name] = attr
        except TypeError:
            pass
    augmenters["Script"] = ScriptAugmenter
    augmenters["Eval"] = EvalAugmenter
    return augmenters


def sequential_from_dicts(config):
    """Construct a sequential imgaug pipeline from a list of dictionaries.
    The dictionaries must have only a `type` and `params` and optional `enabled` key.

    Args:
        config (list of dicts): A list of dicts which specifies the
            augmenters and parameters to use.
    """
    assert isinstance(config, list) and isinstance(config[0], dict),\
        "Config should be a list of dicts."
    augmenters = []
    for operation in config:
        if not operation.get('enabled', True):
            continue
        try:
            augmenters.append(AUGMENTERS[operation['type']](**operation['params']))
        except KeyError:
            raise ValueError(
                f"imgaug:sequential_from_dicts: Invalid operation `{operation['type']}`.")
        except TypeError:
            raise ValueError(
                f"imgaug:sequential_from_dicts: "
                f"Invalid parameters to `{operation['type']}`: {str(operation['params'])}"
                )
    return iaa.Sequential(augmenters)


def sequential_from_json(path):
    with open(path, "r") as file:
        return sequential_from_json(json.load(file))


AUGMENTERS = __make_aug_map()
