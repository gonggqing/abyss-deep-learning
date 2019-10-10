'''
Utilities and generators for segmentation tasks.
Generators expect yield type to be (image, masks) and not in a batch.
Masks may be either categorical or binary.
'''
# from imgaug import augmenters as iaa
# from imgaug.parameters import Deterministic, DiscreteUniform
# from skimage.transform import resize
import numpy as np
from skimage.morphology import remove_small_holes

from skimage.color import label2rgb
from abyss_deep_learning.utils import instance_to_categorical

from abyss_deep_learning.keras import tasks

####### Methods ######

def jaccard_index(y_true, y_pred):
    '''Compute the jaccard index of a binary mask array.

    Args:
        y_true (np.ndarray): Array of ground truth binary masks of shape [height, width, #instances].
        y_pred (np.ndarray): Array of predicted binary masks of shape [height, width, #instances].

    Returns:
        np.float: The jaccard index of the true and predicted masks.
    '''
    return np.sum(y_pred & y_true) / np.sum(y_pred | y_true)


def jaccard_distance(y_true, y_pred):
    '''Compute the jaccard distance of a binary mask array.

    Args:
        y_true (np.ndarray): Array of ground truth binary masks of shape [height, width, #instances].
        y_pred (np.ndarray): Array of predicted binary masks of shape [height, width, #instances].

    Returns:
        np.float: The jaccard distance of the true and predicted masks.'''
    return 1 - jaccard_index(y_true, y_pred)

######### Segmentation generators #########

def instances_to_categorical_gen(gen, num_cats):
    '''converts an instance segmentation generator to a categorical semantic segmentation generator.
    No batching support.

    Args:
        gen (generator): A keras compatible generator yielding (input, masks, classes) where input is an
          RGB image, masks is a binary mask array of shape [height, width, #instances] and classes is
          an integer array or shape [#instances] that encodes the class number of each instance.
        num_cats (int): The maximum number of categories to represent.

    Yields:
        tuple: A keras compatible tuple containing the RGB input image and the categorical mask.
    '''
    for rgb, instances, classes in gen:
        mask = instance_to_categorical(instances, classes, num_cats)
        yield rgb, mask


def label_display_gen(gen, bg_label=None):
    '''A generator modifier that shows a human readable segmentation.
    Single-image only, no batching.

    Args:
        gen (np.ndarray): A keras compatible generator
        bg_label (int, optional): The value of the background label.

    Yields:
        np.ndarray: A human-readable RGB image with the labels overlaid.
    '''
    for rgb, mask in gen:
        disp = ((rgb + 1) * 255/2).astype(np.uint8)
        disp = label2rgb(
            np.argmax(mask, axis=-1),
            image=disp, bg_label=bg_label, contours='thick')
        yield (disp,)


def binary_targets_gen(gen):
    """
    Flattens the final dimension in an occupancy array by running logical_or over the last dimension.

    Args:
        gen (np.ndarray): A keras compatible generator where the targets are an occupancy grid of
          shape [height, width, C], where C can represent #classes or #instances.

    Yields:
        tuple: A keras compatible tuple containing the RGB input image and an occupancy grid.
    """
    for inputs, targets in gen:
        targets[..., 0] = np.logical_not(np.logical_or.reduce(targets[..., 1:], axis=-1))
        yield inputs, targets[..., 0:1]


def fill_mask_gen(gen, min_size=50):
    '''Read (rgb, occupancy matrix) pair, fill any holes in the occupancy matrix and yield the pair.

    Args:
        gen (generator): A keras compatible generator where targets is an occupancy matrix.
        min_size (int, optional): Minimum hole size to fill.

    Yields:
        tuple: A keras compatible input tuple with tzzhe targets modified.
    '''
    for rgb, mask in gen:
        for cat_idx in range(1, mask.shape[-1]):
            remove_small_holes(mask[..., cat_idx], min_size=min_size, in_place=True)
        mask[..., 0] = np.sum(mask[..., 1:], axis=-1) == 0
        yield rgb, mask

def random_crop_gen(gen, output_shape, decimation=30, max_crops=10):
    '''Reduce the 
    
    Args:
        gen (generator): A keras generator.
        decimation (int): Decimate spatial samples from labels before generating centroid targets.
            Larger numbers produce fewer samples.
    
    Yields:
        TYPE: image, 
    '''
    seq = iaa.Affine(order=0)
    samples = None
    for image, mask in gen:
        spacing = np.max(image.shape[0:2]) // decimation
        samples = np.argwhere(np.sum(mask[::spacing, ::spacing, 1:], axis=2) > 0) * spacing
        if samples.size == 0:
            continue
        np.random.shuffle(samples)
        samples = samples.tolist()
        i = 0
        while samples and (i < max_crops):
            sample = samples.pop()
            seq.translate = (
                Deterministic(np.floor(output_shape[1]/2 - sample[1]).astype(int)),
                Deterministic(np.floor(output_shape[0]/2 - sample[0]).astype(int)))
            seq_det = seq.to_deterministic()
            image_c, mask_c = seq_det.augment_image(image), seq_det.augment_image(mask)
            i += 1
            yield image_c, mask_c

#class Task( tasks.Base ):
class Task( tasks.Base, ClassifierMixin ):
    """image segmentation with user-defined backend"""

    def _create_model( self ):
        from keras.models import Model
        raise NotImplementedError( "todo!" )

    def predict(self, x, batch_size=32, verbose=0, steps=None):
        """Returns the class predictions for the given data.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.
        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.

        Args:
            x (TYPE): Description
            batch_size (int, optional): Description
            verbose (int, optional): Description
            steps (None, optional): Description

        Returns:
            TYPE: Description
        """
        
        #  len(x)xNxM
        
        raise NotImplementedError( "todo!" )
    
    # todo? implement if ClassifierMixin does not work
    #def score( self, X, y, sample_weight = None )
        '''returns the mean accuracy on the given test data and labels

           parameters:
               X: array-like, shape = (n_samples, n_features); test samples
               y: array-like, shape = (n_samples) or (n_samples, n_outputs); true labels
               sample_weight: array-like, shape = [n_samples], optional; sample weights

           returns
               score : float; mean accuracy of self.predict(X) wrt. y
        '''
        #raise NotImplementedError( "todo!" )


#### TODO: Move cropping from below into new augmentation_gen
# def augmentation_gen(gen, aug_config, enable=True):
#     '''Must be batchless, no guarantee on balanced sampling'''
#     if not enable:
#         while True:
#             yield from gen
#     seq = iaa.Sequential([
#         iaa.Affine(**aug_config['affine1']),
#         iaa.Fliplr(aug_config['flip_lr_percentage']),
#         iaa.Flipud(aug_config['flip_ud_percentage']),
#         iaa.Affine(**aug_config['affine2'])
#     ])
#     ishape = aug_config['input_shape'] # (H, W)
#     spacing = aug_config['spacing']
#     samples = None
#     for image, mask in gen:
#         if aug_config['crop']:
#             samples = np.argwhere(np.sum(mask[::spacing, ::spacing, 1:], axis=2) > 0) * spacing
#             if samples.size == 0:
#                 continue
#             np.random.shuffle(samples)
#             samples = samples.tolist()
#         i = 0
#         while ((samples and i < aug_config['crops_per_image']) or (not aug_config['crop'] and i == 0)):
#             if aug_config['crop']:
#                 sample = samples[0]
#                 seq[0].translate = (
#                     Deterministic(np.floor(ishape[1]/2 - sample[1]).astype(int)),
#                     Deterministic(np.floor(ishape[0]/2 - sample[0]).astype(int)))
#                 del samples[0]
#             seq_det = seq.to_deterministic()
#             image_c, mask_c = seq_det.augment_image(image), seq_det.augment_image(mask)
#             yield image_c, mask_c
#             i += 1
