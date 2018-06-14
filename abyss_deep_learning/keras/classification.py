'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, target) and not in a batch.
'''

from imgaug import augmenters as iaa
import numpy as np

def augmentation_gen(gen, aug_config, enable=True):
    '''
    Data augmentation for classification task.
    Target is untouched.
    '''
    if not enable:
        while True:
            yield from gen
    seq = iaa.Sequential([
        iaa.Fliplr(aug_config['flip_lr_percentage']),
        iaa.Flipud(aug_config['flip_ud_percentage']),
        iaa.Affine(**aug_config['affine'])
    ])
    for image, target in gen:
        yield seq.augment_image(image), target
