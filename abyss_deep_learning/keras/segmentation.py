'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, masks) and not in a batch.
Masks may be either categorical or binary.
'''
from imgaug import augmenters as iaa
from imgaug.parameters import Uniform, Deterministic, DiscreteUniform
from skimage.color import label2rgb
from skimage.morphology import remove_small_holes
import numpy as np

def pack_masks(masks, mask_classes, num_classes):
    '''Pack a list of instance masks into a categorical mask
    works only on a single image, no batch'''
    num_shapes = len(mask_classes)
    shape = masks.shape
    packed = np.zeros(shape[0:2] + (num_classes,), dtype=np.uint8)
    packed[..., 0] = 1
    for i in range(num_shapes):
        class_id = mask_classes[i]
        mask = masks[..., i]
        packed[..., class_id] |= mask
        packed[..., 0] &= ~mask
    return packed


######### Segmentation generators #########

def instances_to_categorical_gen(gen, num_classes):
    '''converts an instance segmentation generator to a categorical semantic segmentation generator.
    No batching support.'''

    for rgb, instances, classes in gen:
        mask = pack_masks(instances, classes, num_classes)
        yield rgb, mask



def label_display_gen(gen, bg_label=0):
    '''Single-image only, no batching.'''
    for rgb, mask in gen:
        disp = ((rgb + 1) * 255/2).astype(np.uint8)
        disp = label2rgb(
            np.argmax(mask, axis=-1),
            image=disp, bg_label=bg_label)
        yield (disp,)

def binary_targets_gen(gen):
    """
    Flatten target masks into a single channelZ
    """
    for inputs, targets in gen:
        targets[..., 0] = np.logical_or.reduce(targets[..., 1:], axis=-1)
        yield inputs, targets[..., 0:1]

def fill_mask_gen(gen, min_size=50):
    '''Read (rgb, cat-label) pair, fill any holes in cat-label and yield the pair.
       Cannot be used in batches.'''
    for rgb, mask in gen:
        for cat_idx in range(1, mask.shape[-1]):
            remove_small_holes(mask[..., cat_idx], min_size=min_size, in_place=True)
        mask[..., 0] = np.sum(mask[..., 1:], axis=-1) == 0
        yield rgb, mask


def augmentation_gen(gen, aug_config, enable=True):
    '''Must be batchless, no guarantee on balanced sampling'''
    if not enable:
        while True:
            yield from gen
    seq = iaa.Sequential([
        iaa.Affine(**aug_config['affine1']),
        iaa.Fliplr(aug_config['flip_lr_percentage']),
        iaa.Flipud(aug_config['flip_ud_percentage']),
        iaa.Affine(**aug_config['affine2'])
    ])
    ishape = aug_config['input_shape'] # (H, W)
    spacing = aug_config['spacing']
    samples = None
    for image, mask in gen:
        if aug_config['crop']:
            samples = np.argwhere(np.sum(mask[::spacing, ::spacing, 1:], axis=2) > 0) * spacing
            if samples.size == 0:
                continue
            np.random.shuffle(samples)
            samples = samples.tolist()
        i = 0
        while ((samples and i < aug_config['crops_per_image']) or (not aug_config['crop'] and i == 0)):
            if aug_config['crop']:
                sample = samples[0]
                seq[0].translate = (
                    Deterministic(np.floor(ishape[1]/2 - sample[1]).astype(int)),
                    Deterministic(np.floor(ishape[0]/2 - sample[0]).astype(int)))
                del samples[0]
            seq_det = seq.to_deterministic()
            image_c, mask_c = seq_det.augment_image(image), seq_det.augment_image(mask)
            yield image_c, mask_c
            i += 1
