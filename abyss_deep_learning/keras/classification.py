'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, target) and not in a batch.
'''
import os
import json
from itertools import cycle
from random import shuffle
import sys
import warnings

import keras.backend as K
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa

from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.deprecation import deprecated
from sklearn.utils.validation import check_is_fitted
from tensorboard.plugins.pr_curve import summary as pr_summary

from abyss_deep_learning.keras.utils import batching_gen, gen_dump_data
from abyss_deep_learning.utils import cat_to_onehot, warn_on_call
import skimage.color

from keras.callbacks import TensorBoard
import keras.optimizers
from keras.preprocessing.image import ImageDataGenerator
import keras.regularizers
from keras.utils import multi_gpu_model
from abyss_deep_learning.keras import tasks

from abyss_deep_learning.utils import cat_to_onehot, warn_on_call

class Task( tasks.Base, ClassifierMixin ):
    """image classifier with user-defined backend"""

    def _create_model( self ):
        from keras.applications.xception import Xception
        from keras.models import Model
        from keras.layers import Dense
        # Load the model with imagenet weights, they will be re-initialized later weights=None
        # todo! move to something like keras.models.Classification or alike (certainly do better design and naming)
        config = dict(
            include_top=False,
            weights=self.init_weights,
            input_shape=self.input_shape,
            pooling=self.pooling)
        if self.backbone == 'xception':
            model = Xception( include_top = config['include_top']
                            , weights = config['weights']
                            , input_shape = config['input_shape']
                            , pooling = config['pooling'] )
        else:
            raise ValueError( "expected valid backbone; got: '{}'".format( self.backbone ) )
        model = Model(
            model.inputs,
            Dense(self.classes, activation=self.output_activation, name='logits')(model.outputs[0]))
        self.model_ = model
        self.classes_ = np.arange(self.classes) # Sklearn API recomendation
        self.set_trainable(self.trainable)
        if self.init_weights != 'imagenet':
            self.set_weights(self.init_weights)
        if self.l12_reg:
            self.add_regularisation(self.l12_reg[0], self.l12_reg[1])

    def predict_proba(self, x, batch_size=32, verbose=0, steps=None):
        """Returns the class predictions scores for the given data.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
        # Returns
            scores: array-like, shape `(n_samples, n_classes)`
                Class prediction scores.

        Args:
            x (TYPE): Description
            batch_size (int, optional): Description
            verbose (int, optional): Description
            steps (None, optional): Description

        Returns:
            TYPE: Description
        """
        check_is_fitted(self, "model_")
        return self.model_.predict(x, batch_size=batch_size, verbose=verbose, steps=steps)

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
        proba = self.predict_proba(x, batch_size=batch_size, verbose=verbose, steps=steps)
        classes = proba.argmax(axis=-1) if proba.shape[-1] > 1 else (proba > 0.5).astype('int32')
        return self.classes_[classes]

def sequential_process(images):
    # images = np.array([
    #     skimage.color.rgb2grey(image)
    #     for image in images])

    return images.astype("float") / 255.0

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
