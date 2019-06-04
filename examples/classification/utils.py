import numpy as np

def to_multihot(captions, num_classes):
    """
    Converts a list of classes (int) to a multihot vector
    Args:
        captions: (list of ints). Each class in the caption list
        num_classes: (int) The total number of classes

    Returns:

    """
    hot = np.zeros([num_classes])
    if isinstance(captions, int):
        hot[captions] = 1
    else:
        for c in captions:
            hot[int(c)] = 1
    return hot