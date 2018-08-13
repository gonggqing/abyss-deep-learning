'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, target) and not in a batch.
'''
from itertools import cycle
from random import shuffle
import json

from imgaug import augmenters as iaa
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from pycocotools.coco import COCO
from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize
from tensorboard.plugins.pr_curve import summary as pr_summary
import keras.backend as K
import numpy as np
import tensorflow as tf

from abyss_deep_learning.keras.utils import batching_gen, gen_dump_data


def hamming_loss(y_true, y_pred):
    return K.mean(y_true * (1 - y_pred) + (1 - y_true) * y_pred)


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
        assert images.shape[-1] == 3, "classification.Inference.predict(): Images must be RGB."
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        return self.model.predict(self._preprocess_input(images))


####### Generators ######

def cached_gen(gen, cache_size):
    datagen = ImageDataGenerator()
    data = gen_dump_data(gen, cache_size)
    return batching_gen(datagen.flow(data[0], data[1], batch_size=1), 0)

def skip_bg_gen(gen):
    for image, target in gen:
        if np.sum(target) == 0:
            continue
        yield image, target

def caption_map_gen(gen, caption_map, background=None, skip_bg=False):
    for image, captions in gen:
        if not captions or (background in captions and skip_bg):
            if skip_bg:
                continue
            yield image, []
        else:
            yield image, [
                caption_map[caption]
                for caption in captions
                if caption in caption_map and caption != background]

def cast_dtype_gen(gen, input_dtype, target_dtype):
    for inputs, targets in gen:
        yield inputs.astype(input_dtype), targets.astype(target_dtype)

def set_to_multihot(captions, num_classes):
    return np.array([1 if i in captions else 0 for i in range(num_classes)])

def multihot_gen(gen, num_classes):
    for image, captions in gen:
        yield image, set_to_multihot(captions, num_classes)


def augmentation_gen(gen, aug_config, enable=True):
    '''
    Data augmentation for classification task.
    Target is untouched.
    '''
    if not enable:
        while True:
            yield from gen
    seq = iaa.Sequential(aug_config)
    for image, target in gen:
        yield seq.augment_image(image), target
