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
from sklearn.utils.validation import check_is_fitted
from tensorboard.plugins.pr_curve import summary as pr_summary

from abyss_deep_learning.keras.utils import batching_gen, gen_dump_data

def hamming_loss(y_true, y_pred):
    return K.mean(y_true * (1 - y_pred) + (1 - y_true) * y_pred)

@deprecated("Use ImageClassifier instead.")
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




class ImageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 backbone='xception', output_activation='softmax',
                 input_shape=(299, 299, 3), pooling='avg', classes=2,
                 init_weights='imagenet', init_epoch=0, init_lr=1e-3,
                 trainable=True, loss='categorical_crossentropy'):
        self.backbone = backbone
        self.loss = loss
        self.output_activation = output_activation
        self.input_shape = input_shape
        self.pooling = pooling
        self.classes = classes
        self.trainable = trainable
        self.init_weights = init_weights
        self.init_lr = init_lr
        self.init_epoch = init_epoch
        
    def set_weights(self, weights):
        # Initialize model weights and sklearn weights parameter
        model = self.model_
        if weights is None:
            # Re-initialize all weights (clear imagenet weights)
            session = K.get_session()
            for weight in model.weights:
                weight.initializer.run(session=session)
        elif isinstance(weights, str):
            model.load_weights(weights, by_name=True)
        elif isinstance(weights, list) and isinstance(weights[0], np.ndarray):
            # Load weights from a list of np.ndarrays, to support sklearn model serialization
            model.set_weights(weights)
        else:
            raise ValueError("ImageClassifier::set_weights(): Invalid weights.")

    def _create_model(self):
        from keras_applications.xception import Xception, preprocess_input
        from keras.models import Model
        from keras.layers import Dense

        if hasattr(self, "model_"):
            return

        # Load the model with imagenet weights, they will be re-initialized later weights=None
        model_config = dict(
            include_top=False,
            weights=self.init_weights,
            input_shape=self.input_shape,
            pooling=self.pooling)

        if self.backbone == 'xception':
            model = Xception(**model_config)
        else:
            raise ValueError(
                "ImageClassifier::__init__(): Invalid backbone '{}'".format(self.backbone))

        # Add the classification head
        model = Model(
            model.inputs,
            Dense(self.classes, activation=self.output_activation, name='logits')(model.outputs[0]))

        self.model_ = model
        self.classes_ = np.arange(self.classes) # Sklearn API recomendation
        self.set_trainable(self.trainable)
        if self.init_weights != 'imagenet':
            self.set_weights(self.init_weights)

    def set_lr(self, lr):
        self.lr = lr
        K.set_value(self.model_.optimizer.lr, lr)
    
    def set_trainable(self, trainable):
        """Sets the network layers ability to train or freeze.
        #TODO doco"""
        check_is_fitted(self, 'model_', msg="ImageClassifier::set_trainable(): Model not yet constructed, use constructor to set trainable.")
        if isinstance(trainable, bool):
            # Make all layers trainable or non trainable
            for layer in self.model_.layers:
                layer.trainable = trainable
        elif isinstance(trainable, dict):
            # Use a dictionary to map layer names to trainable flags
            for layer in self.model_.layers:
                if layer.name in trainable:
                    layer.trainable = trainable[layer.name]
        elif isinstance(trainable, list):
            # Use a list to map layer index to trainable flags
            for layer, do_train in zip(self.model_.layers, trainable):
                layer.trainable = do_train
        else:
            raise ValueError(
                "ImageClassifier::set_trainable(): Invalid argument type (accepts bool, dict and list).")
        self.trainable = trainable

    def _maybe_compile(self):
        '''If the model has not been created, create and compile the model.'''
        check_is_fitted(self, "model_")
        if not self.model_._is_compiled:
            if self.trainable is False:
                # Don't compile models that are not trainable (predict only)
                return
            if self.loss is None:
                raise ValueError(
                    "ImageClassifier::fit(): Trying to compile a model without a loss function.")
            self.model_.compile('nadam', loss=self.loss)
            self.set_lr(self.init_lr)
            

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
            callbacks=None, validation_split=0.0, validation_data=None,
            shuffle=True, class_weight=None, sample_weight=None,
            steps_per_epoch=None, validation_steps=None):
        '''Fit numpy arrays to the model.'''
        self._create_model()
        self._maybe_compile()
        # Fit model
        self.history_ = self.model_.fit(
            x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
            callbacks=callbacks, validation_split=validation_split, validation_data=validation_data,
            shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight,
            initial_epoch=self.init_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
        # Update sklearn params
        return self

    def fit_generator(
            self, generator, steps_per_epoch=None, epochs=1, verbose=1,
            callbacks=None, validation_data=None, validation_steps=None,
            class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False,
            shuffle=True):
        '''Fit a generator to the model.'''
        self._create_model()
        self._maybe_compile()
        # Fit model
        self.history_ = self.model_.fit_generator(
            generator, steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_data, validation_steps=validation_steps,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=self.init_epoch)

        return self
    
    def fit_dataset(
            self, dataset_train, dataset_val=None, steps_per_epoch=None, epochs=1, verbose=1,
            callbacks=None, validation_steps=None,
            class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False,
            shuffle=True):
        self._create_model()
        self._maybe_compile()
        params = dict(
            dataset_train.generator(endless=True),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle,
            initial_epoch=self.init_epoch)
        if dataset_val is not None:
            params = params.update(dict(
                validation_data=dataset_val.generator(endless=True),
                validation_steps=validation_steps))
        return self.fit_generator(**params)
        
    def predict_proba(self, x, batch_size=32, verbose=0, steps=None):
        """Returns the class predictions scores for the given data.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
        # Returns
            scores: array-like, shape `(n_samples, n_classes)`
                Class prediction scores.
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
        """
        proba = self.predict_proba(x, batch_size=batch_size, verbose=verbose, steps=steps)

        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = (proba > 0.5).astype('int32')
        return self.classes_[classes]
    
    def save(self, filepath, include_optimizer=False):
        from keras.utils.io_utils import h5dict
        from keras.engine.saving import save_weights_to_hdf5_group#, _serialize_model
        from json import dumps
        import h5py
        
        self.model_.save_weights(filepath)
        with h5py.File(filepath, 'a') as f:
            topology = f.create_dataset("topology", data=self.model_.to_json())
            topology.attrs['format'] = 'json'
            parameters = f.create_dataset("parameters", data=dumps(self.get_params()))
            parameters.attrs['format'] = 'json'
    
    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        from keras.utils.io_utils import h5dict
        from keras.engine.saving import load_weights_from_hdf5_group_by_name#, _deserialize_model
        from json import loads
        from keras.models import model_from_json

        f = h5dict(filepath, mode='r')
        parameters = loads(str(f['parameters']))
        model = ImageClassifier(**parameters)
    #     topology = str(f['topology'])
    #     model.model_ = model_from_json(topology)
        model._create_model()
        model.model_.load_weights(filepath)
        f.close()
        return model

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
