'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, target) and not in a batch.
'''
from itertools import cycle
from random import shuffle
import json

from imgaug import augmenters as iaa
from pycocotools.coco import COCO
from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize
import numpy as np

class FromAnnDataset(COCO):
    def __init__(self, *args, **kwargs):
        super(FromAnnDataset, self).__init__(*args, **kwargs)
        
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_path = self.loadImgs([image_id])[0]['path']
        
        image = imread(image_path)
        if not isinstance(image, np.ndarray):
            # Temporary bugfix for PIL error
            image = imread(image_path, plugin='matplotlib') * 255
        if not isinstance(image, np.ndarray):
            print(image)
            raise Exception(
                "skimage.io.read failed: image is of type {:s}".format(str(type(image))))
        # If grayscale. Convert to RGB for consistency.
        if image.ndim == 1 or (image.ndim == 3 and image.shape[2] == 1):
            image = gray2rgb(np.squeeze(image))
        elif image.ndim == 3:
            if image.shape[2] > 3:
                image = image[..., 0:3]  # Remove any alpha channel
            elif image.shape[2] != 3:
                raise Exception(
                    "load_image tried to load an image with dims of {:s}".format(str(image.shape)))
        return image
        
    def load_categories(self, image_id):
        assert isinstance(image_id, int), "Must pass exactly one ID"
        caps = [annotation['category_id']
         for annotation in self.loadAnns(self.getAnnIds([image_id]))]
        return set(caps)
    
    def num_images(self, imgIds=[], catIds=[]):
        return len(self.getAnnIds(imgIds, catIds))
        
    def generator(self, imgIds=[], shuffle_ids=False):
        if not imgIds:
            imgIds = [ann['image_id'] for ann in self.loadAnns(ids=self.getAnnIds())]
        if shuffle_ids:
            shuffle(imgIds)
        for image_id in cycle(imgIds):
            yield self.load_image(image_id), self.load_categories(image_id)

class ClassificationDataset(COCO):
    def __init__(self, caption_map, *args, **kwargs):
        super(ClassificationDataset, self).__init__(*args, **kwargs)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_path = self.loadImgs([image_id])[0]['path']

        image = imread(image_path)
        if not isinstance(image, np.ndarray):
            # Temporary bugfix for PIL error
            image = imread(image_path, plugin='matplotlib') * 255
        if not isinstance(image, np.ndarray):
            print(image)
            raise Exception(
                "skimage.io.read failed: image is of type {:s}".format(str(type(image))))
        # If grayscale. Convert to RGB for consistency.
        if image.ndim == 1 or (image.ndim == 3 and image.shape[2] == 1):
            image = gray2rgb(np.squeeze(image))
        elif image.ndim == 3:
            if image.shape[2] > 3:
                image = image[..., 0:3]  # Remove any alpha channel
            elif image.shape[2] != 3:
                raise Exception(
                    "load_image tried to load an image with dims of {:s}".format(str(image.shape)))
        return image

    def load_caption(self, image_id):
        assert isinstance(image_id, int), "Must pass exactly one ID"
        caps = [caption.split(',')
         for annotation in self.loadAnns(self.getAnnIds([image_id]))
         for caption in annotation['caption']]
        return set([i for f in caps for i in f])

    def num_images(self, imgIds=[], catIds=[]):
        return len(self.getAnnIds(imgIds, catIds))

    def generator(self, imgIds=[], shuffle_ids=False):
        if not imgIds:
            imgIds = [ann['image_id'] for ann in self.loadAnns(ids=self.getAnnIds())]
        if shuffle_ids:
            shuffle(imgIds)
        for image_id in cycle(imgIds):
            yield self.load_image(image_id), self.load_caption(image_id)


class Inference(object):
    def __init__(self, config_path):
        """Instantiate an image classification detector and initialise it with the configuration specified
        in the JSON at config_path.

        Args:
            config_path (str): Path to the JSON describing the image classification detector.
                               See example in workspace/example-project/models/model-1.json
        """
        from keras.models import model_from_json
        with open(config_path, "r") as config_file:
            self.config = json.load(config_file)
        self.model = model_from_json(self.config['model'])
        self.model.load_weights(self.config['weights'])
        if self.config['architecture']['backbone'] == "inceptionv3":
            from keras.applications.inception_v3 import preprocess_input
            self._preprocess_model_input = preprocess_input
        else:
            raise ValueError(
                "Unknown model architecture.backbone '{:s}'".format(
                    self.config['architecture']['backbone']))

    def _preprocess_input(self, images):
        images = np.array([
            resize(image, self.config['architecture']['input_shape'], preserve_range=True)
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
        assert images.shape[-1] == 3, "classification.Inference.predict(): Images must be RGB."
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        return self.model.predict(self._preprocess_input(images))




####### Generators ######

def caption_map_gen(gen, caption_map):
    for image, captions in gen:
        yield image, [caption_map[i] for i in captions]


def onehot_gen(gen, num_classes):
    for image, captions in gen:
        onehot = np.array([1 if i in captions else 0 for i in range(num_classes)])
        yield image, onehot


def augmentation_gen(gen, aug_config, enable=True):
    '''
    Data augmentation for classification task.
    Target is untouched.
    '''
    if not enable:
        while True:
            yield from gen
    aug_list = []
    if 'flip_lr_percentage' in aug_config:
        aug_list += [iaa.Fliplr(aug_config['flip_lr_percentage'])]
    if 'flip_ud_percentage' in aug_config:
        aug_list += [iaa.Flipud(aug_config['flip_ud_percentage'])]
    if 'affine' in aug_config:
        aug_list += [iaa.Affine(**aug_config['affine'])]
    if 'color' in aug_config:
        aug_list += [
            iaa.Sometimes(
                aug_config['color']['probability'], iaa.Sequential([
                    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                    iaa.WithChannels(0, iaa.Add(aug_config['color']['hue'])),
                    iaa.WithChannels(1, iaa.Add(aug_config['color']['saturation'])),
                    iaa.WithChannels(2, iaa.Add(aug_config['color']['value'])),
                    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        ]))]
    if 'custom' in aug_config:
        aug_list += aug_config['custom']
    seq = iaa.Sequential(aug_list)
    for image, target in gen:
        yield seq.augment_image(image), target