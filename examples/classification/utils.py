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


def multihot_gen(gen, num_classes):
    """A stream modifier that converts categorical labels into one-hot vectors.

    Args:
        gen (generator): A keras compatible generator where the targets are a list of categorical labels.
        num_classes (int): Total number of categories to represent.

    Yields:
        generator: A keras compatible generator with the targets modified.
    """
    for image, captions in gen:
        yield image, to_multihot(captions, num_classes)


def compute_class_weights(dataset):
    '''
    computes the ideal weights from each class based on the frequency of each class.
    For example, if there are 12.5 times more of class 0 than class 1, then returns {0: 12.5,
                                                                                     1: 1.0}
    '''
    dataset._calc_class_stats()
    min_val = dataset.stats['class_weights'][
        min(dataset.stats['class_weights'].keys(), key=(lambda k: dataset.stats['class_weights'][k]))]
    return dataset.stats['class_weights'].update((x,y/min_val) for x,y in dataset.stats['class_weights'].items())