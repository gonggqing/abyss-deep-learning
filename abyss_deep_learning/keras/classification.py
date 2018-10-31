'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, target) and not in a batch.
'''
import json
from itertools import cycle
from random import shuffle

import keras.backend as K
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from pycocotools.coco import COCO
from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.deprecation import deprecated
from tensorboard.plugins.pr_curve import summary as pr_summary

from abyss_deep_learning.keras.utils import batching_gen, gen_dump_data
from abyss_deep_learning.utils import cat_to_onehot, warn_on_call



@deprecated("Use ImageClassifier instead.")
class Inference(object):

    """Summary
    
    Attributes:
        config (TYPE): Description
        model (TYPE): Description
    """
    
    def __init__(self, config_path):
        """Instantiate an image classification detector and initialise it with the configuration specified
        in the JSON at config_path.
        
        Args:
            config_path (str): Path to the JSON describing the image classification detector.
                               See example in workspace/example-project/models/model-1.json
        
        Raises:
            ValueError: Description
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
        """Summary
        
        Args:
            images (TYPE): Description
        
        Returns:
            TYPE: Description
        """
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
    """A stream modifier that caches the first cache_size results in a generator.
    
    Args:
        gen (generator): A keras compatible data generator.
        cache_size (int): The number of elements to read from the generator and offer cached.
    
    Returns:
        generator: A generator that offers the newly cached data.
    """
    datagen = ImageDataGenerator()
    data = gen_dump_data(gen, cache_size)
    return batching_gen(datagen.flow(data[0], data[1], batch_size=1), 0)

def skip_bg_gen(gen):
    """Skip tuples in the generator that do not have any labels.
    
    Args:
        gen (generator): A keras generator where the target is either a score or occupancy array.
    
    Yields:
        generator: A keras generator that skips tuples that do not have any labels.
    """
    for image, target in gen:
        if np.sum(target) == 0:
            continue
        yield image, target

@deprecated("Use the new dataset translators interface.")
def caption_map_gen(gen, caption_map, background=None, skip_bg=False):
    """A stream modifier that maps the target labels given a mapping dict.
    
    Args:
        gen (generator): A keras compatible generator.
        caption_map (dict): A dict where keys are initial labels and values are re-mapped labels.
        background (object): The background label, when given skip any rows with this label.
        skip_bg (bool, optional): Toggle flag for skipping BGs.
    
    Yields:
        TYPE: A keras generator with the inputs and targets modified.
    """
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

@deprecated("Use the new dataset translators interface.")
def cast_dtype_gen(gen, input_dtype, target_dtype):
    """A stream modifier that converts the array data types of the input and target.
    
    Args:
        gen (generator): A keras compatible generator.
        input_dtype (np.dtype or str): The data type to convert the inputs to.
        target_dtype (np.dtype or str): The data type to convert the targets to.
    
    Yields:
        generator: A keras compatible generator with the inputs and targets modified.
    """
    for inputs, targets in gen:
        yield inputs.astype(input_dtype), targets.astype(target_dtype)

@deprecated("Use the new dataset translators interface.")
def onehot_gen(gen, num_classes):
    """A stream modifier that converts categorical labels into one-hot vectors.
    
    Args:
        gen (generator): A keras compatible generator where the targets are a list of categorical labels.
        num_classes (int): Total number of categories to represent.
    
    Yields:
        generator: A keras compatible generator with the targets modified.
    """
    for image, captions in gen:
        yield image, cat_to_onehot(captions, num_classes)


def augmentation_gen(gen, aug_config, enable=True):
    '''Data augmentation for classification task.
    Image is augmented and target is untouched.
    
    Args:
        gen (generator): A keras compatible generator.
        aug_config (dict): An imgaug config.
        enable (bool, optional): Flag enabling this generator.
    
    Yields:
        generator: A keras compatible generator with modified inputs.
    '''
    if not enable:
        while True:
            yield from gen
    seq = iaa.Sequential(aug_config)
    for image, target in gen:
        yield seq.augment_image(image), target


########## OTHER FUNCTIONS #############

def hamming_loss(y_true, y_pred):
    """Returns the hamming loss betweeen y_true and y_pred.
    
    Args:
        y_true (np.ndarray): Array of ground truth score arrays.
        y_pred (np.ndarray): Array of predicted score arrays.

    Returns:
        np.float: The hamming loss between the true and predicted masks.
    """
    return K.mean(y_true * (1 - y_pred) + (1 - y_true) * y_pred)