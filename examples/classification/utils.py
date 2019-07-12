import numpy as np
import imgaug
from imgaug import augmenters as iaa

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


def create_augmentation_configuration(some_of=None, flip_lr=True, flip_ud=True, gblur=None, avgblur=None, gnoise=None, scale=None, rotate=None, bright=None, colour_shift=None):
        """
        More info at https://imgaug.readthedocs.io/en/latest/source/augmenters.html
        Args:
            some_of: (int) The maximum amount of transforms to apply. Randoms selects 0->some_of transforms to apply.
            flip_lr: (bool) Randomly flip the images from left to right
            flip_ud: (bool) randomly flip the images from left to right
            gblur: (tuple) Apply gaussian blur. This is equal to sigma. gblur=(0.0,3.0)
            avgblur: (tuple) Apply average blur. This is equal to kernel size e.g. avgblur=(2,11)
            gnoise: (tuple) Apply guassian noise. This parameter is equal to scale range. e.g. gnoise=(0,0.05*255)
            scale: (tuple) Apply scale transformations. This parameter is equal to scale. e.g. scale=(0.5,1.5) to scale between 50% and 150% of image size
            rotate: (tuple) Apply rotation transformations. This parameter is equal to rotation degrees. e.g. scale=(-45,45)
            bright: (tuple) Brighten the image by multiplying. This parameter is equal to the range of brightnesses, e.g. bright=(0.9,1.1)
            colour_shift: (tuple) Apply a color slight colour shift to some of the channels in the image. This parameter is equal to the multiplying factor on each channel. E.g. colour_shift=(0.9,1.1)
        """
        aug_list = []
        if flip_lr:
            aug_list.append(iaa.Fliplr(0.5))
        if flip_ud:
            aug_list.append(iaa.Flipud(0.5))

        sometimes_list = []
        if gblur:
            sometimes_list.append(iaa.GaussianBlur(sigma=gblur))
        if avgblur and not gblur:
            # Only use avgblur if gblur is not being used
            sometimes_list.append(iaa.AverageBlur(k=avgblur))
        if gnoise:
            sometimes_list.append(iaa.AdditiveGaussianNoise(scale=gnoise))
        if scale:
            sometimes_list.append(iaa.Affine(scale=scale))
        if rotate:
            sometimes_list.append(iaa.Affine(rotate=rotate))
        if bright:
            sometimes_list.append(iaa.Multiply(bright))
        if colour_shift:
            colours = iaa.SomeOf((0, None),[
                iaa.WithChannels(0, iaa.Multiply(mul=colour_shift)),
                iaa.WithChannels(1, iaa.Multiply(mul=colour_shift)),
                iaa.WithChannels(2, iaa.Multiply(mul=colour_shift))])
            sometimes_list.append(colours)

        aug_list.append(iaa.SomeOf((0,some_of), sometimes_list))

        return aug_list

