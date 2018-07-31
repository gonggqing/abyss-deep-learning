'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, masks) and not in a batch.
Masks may be either categorical or binary.
'''
import json

from imgaug import augmenters as iaa
from imgaug.parameters import Deterministic, DiscreteUniform
from skimage.color import label2rgb
from skimage.morphology import remove_small_holes
from skimage.transform import resize
import numpy as np

from abyss_deep_learning.utils import ann_to_mask
from abyss_deep_learning.datasets.base import ImageTargetDataset

class SegmentationDataset(ImageTargetDataset):
    '''Realisation of a ImageTargetDataset class, where the dataset is expected to
       have one image and at least one segmentation type annotation.
       The annotation can refer to only one category id.
       The image targets are the stacked masks generated from the annotations.'''
    def __init__(self, annotation_file, output_shape, num_classes=None, **kwargs):
        '''Instantiate a SegmentationDataset.
           If num_classes is None, it is inferred from the dataset.'''
        super(SegmentationDataset, self).__init__(annotation_file, **kwargs)
        self.output_shape = output_shape
        self._num_classes = num_classes or 1 + len(self.cats)

    @property
    def num_classes(self):
        return self._num_classes

    def load_image_targets(self, image_id):
        return self.load_segmentation(image_id)

    def load_segmentation(self, image_id):
        assert np.issubdtype(type(image_id), np.integer), "Must pass exactly one ID"
        anns = self.loadAnns(self.getAnnIds([image_id]))
        masks = np.array([ann_to_mask(ann, self.output_shape) for ann in anns]).transpose((1, 2, 0))
        class_ids = np.array([ann['category_id'] for ann in anns])
        return _pack_masks(masks, class_ids, self.num_classes)


class SegmentationModel(object):
    def __init__(self, config_path):
        """Instantiate an image segmentation detector and initialise it with the configuration specified
        in the JSON at config_path.

        Args:
            config_path (str): Path to the JSON describing the image segmentation detector.
                               See example in workspace/example-project/models/model-1.json
        """
        from keras.models import model_from_json
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)

        with open(self.config['model'], "r") as model_def:
            self.model = model_from_json(model_def.read())

        self.model.load_weights(self.config['weights'])
        if self.config['architecture']['backbone'] == "inceptionv3":
            from keras.applications.inception_v3 import preprocess_input
        elif self.config['architecture']['backbone'] == "vgg16":
            from keras.applications.vgg16 import preprocess_input
        elif self.config['architecture']['backbone'] == "resnet50":
            from keras.applications.resnet50 import preprocess_input
        else:
            raise ValueError(
                "Unknown model architecture.backbone '{:s}'".format(
                    self.config['architecture']['backbone']))
        self._preprocess_model_input = preprocess_input

    def _preprocess_input(self, images):
        images = np.array([
            resize(image, self.config['architecture']['input_shape'], preserve_range=True, mode='constant')
            for image in images])
        return self._preprocess_model_input(images)

    def predict(self, images):
        """Predict on the input image(s).
        This function takes care of all pre-processing required and accepts uint8 or float32 RGB images.

        Args:
            images (np.ndarray): Array of size [batch_size, height, width, channels] on which to predict.

        Returns:
            np.ndarray: Class probabilities of the predictions.
        """
        assert images.shape[-1] == 3, "segmentation.Inference.predict(): Images must be RGB."
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        return self.model.predict(self._preprocess_input(images))


####### Methods ######

def _pack_masks(masks, mask_classes, num_classes):
    '''Pack a list of instance masks into a categorical mask.
    Expects masks to be shape [height, width, num_instances] and mask_classes to be [num_instances].'''
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


def jaccard_index(y_true, y_pred):
    '''y_true and y_pred must be binary masks'''
    return np.sum(y_pred & y_true) / np.sum(y_pred | y_true)


def jaccard_distance(y_true, y_pred):
    '''y_true and y_pred must be binary masks'''
    return 1 - jaccard_index_(y_true, y_pred)

######### Segmentation generators #########

def instances_to_categorical_gen(gen, num_classes):
    '''converts an instance segmentation generator to a categorical semantic segmentation generator.
    No batching support.'''
    for rgb, instances, classes in gen:
        mask = _pack_masks(instances, classes, num_classes)
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
    Flatten target masks into a single channel
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
