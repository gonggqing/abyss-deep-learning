'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, target) and not in a batch.
'''
import warnings

import keras.backend as K
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.utils.deprecation import deprecated
from sklearn.utils.validation import check_is_fitted
import os
import json
from keras.utils import multi_gpu_model


from abyss_deep_learning.utils import cat_to_onehot, warn_on_call


class ModelPersistence:
    def save(self, filepath):
        """Save a model, its state and its hyperparameters to file.

        Args:
            filepath (TYPE): Description
            include_optimizer (bool, optional): Description
        """
        from keras.engine.saving import save_weights_to_hdf5_group#, _serialize_model
        from json import dumps
        import h5py

        self.model_.save_weights(filepath)
        with h5py.File(filepath, 'a') as f:
            topology = f.create_dataset("topology", data=self.save_model_.to_json())
            topology.attrs['format'] = 'json'
            parameters = f.create_dataset("parameters", data=dumps(self.get_params()))
            parameters.attrs['format'] = 'json'

        # also explicitly save the model definition
        (dirname, filename) = os.path.split(filepath)
        with open(os.path.join(dirname, "model-definition.json"), 'w') as f:
            f.write(self.save_model_.to_json())

    def load(self, filepath):
        raise NotImplementedError("ModelPersistence::load has not been overridden for this class.")

    @staticmethod
    def _load_model(filepath, model_class):
        """Load a model, its state and its hyperparameters from file.

        Args:
            filepath (TYPE): Path to model to load
        """
        from keras.utils.io_utils import h5dict
        from keras.engine.saving import load_weights_from_hdf5_group_by_name#, _deserialize_model
        from json import loads
        from keras.models import model_from_json

        f = h5dict(filepath, mode='r')
        parameters = loads(str(f['parameters']))
        model = model_class(**parameters)
        model._maybe_create_model()
        model.model_.load_weights(filepath, by_name=True)
        f.close()
        return model


def loadImageClassifierByDict(json_path):
    """
    Allows the user to instantiate the Image Classifier using a dictionary of arguments. The same one that is created by ImageClassifier.dump_args.

    This is useful for loading the classifier for validation/prediction.

    Args:
        json_path: path to the params json

    Returns:
        ImageClassifier: The initialised Image Classifier.

    """
    params = json.load(open(json_path, 'r'))

    if params['metrics']:
        mets = params['metrics'].split(',')
    else:
        mets = None

    return ImageClassifier(
        backbone=params['backbone'],
        output_activation=params['output_activation'],
        input_shape=params['input_shape'],
        pooling=params['pooling'],
        classes=params['classes'],
        init_weights=params['init_weights'],
        init_epoch=params['init_epoch'],
        init_lr=params['init_lr'],
        trainable=params['trainable'],
        loss=params['loss'],
        metrics=mets
    )


class ImageClassifier(BaseEstimator, ClassifierMixin, ModelPersistence):

    """A generic image classifier that can use multiple backends, complete with persistance
    functionality, and hides some complex implementation details.
    """

    def __init__(self,
                 backbone='xception', output_activation='softmax',
                 input_shape=(299, 299, 3), pooling='avg', classes=2,
                 init_weights='imagenet', init_epoch=0, init_lr=1e-3,
                 trainable=True, loss='categorical_crossentropy', metrics=None, gpus=None):
        """Summary

        Args:
            backbone (string): One of {'xception'}. More to come as needed.
            output_activation (str, optional): Description
            input_shape (tuple, optional): Description
            pooling (str, optional): Description
            classes (int, optional): Description
            init_weights (str, optional): Description
            init_epoch (int, optional): Description
            init_lr (float, optional): Description
            trainable (bool, optional): Description
            loss (str, optional): Description
        """
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
        self.metrics = metrics
        self.gpus = gpus

    def dump_args(self, json_path):
        """
        Dump the arguments to initialise this class to file.
        Args:
            json_path: Path to the JSON file.

        Returns:

        """
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path))

        params = {}
        params['backbone'] = self.backbone
        if isinstance(self.loss, str):
            params['loss'] = self.loss
        else:
            params['loss'] = None
        params['output_activation'] = self.output_activation
        params['input_shape'] = self.input_shape
        params['pooling'] = self.pooling
        params['classes'] = self.classes
        params['trainable'] = self.trainable
        params['init_weights'] = self.init_weights
        params['init_epoch'] = self.init_epoch
        params['init_lr'] = self.init_lr
        if isinstance(self.metrics, str):
            params['metrics'] = self.metrics
        elif isinstance(self.metrics, list):
            try:
                mets = ','.join(self.metrics)
            finally:
                mets = None
            params['metrics'] = mets
        else:
            params['metrics'] = None

        json.dump(params, open(json_path, 'w'), indent=4, sort_keys=True)


    def set_weights(self, weights):
        """Set the weights of the model.

        Args:
            weights (None, string or list of np.ndarrays): Can be either of:
              * None: Do not load any weights
              * str: Load the weights at the given path
              * list of np.ndarrays: Load the weight values given from model.get_weights().

        Raises:
            ValueError: If the form of the weights is not recognised.
        """
        self._maybe_create_model()

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

    def _maybe_create_model(self, force=False):
        """Create the model if it has not already been created

        Raises:
            ValueError: If backbone is invalid.
        """
        # from keras_applications.xception import Xception#, preprocess_input
        from keras.applications.xception import Xception
        from keras.models import Model
        from keras.layers import Dense

        if not force and hasattr(self, "model_"):
            return

        self.model_ = None
        K.clear_session()

        # Load the model with imagenet weights, they will be re-initialized later weights=None
        config = dict(
            include_top=False,
            weights=self.init_weights,
            input_shape=self.input_shape,
            pooling=self.pooling)

        import tensorflow as tf
        if self.backbone == 'xception':
#          with tf.device('/cpu:0'):
                model = Xception(
                        include_top=config['include_top'],
                        weights=config['weights'],
                        input_shape=config['input_shape'],
                        pooling=config['pooling'])
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
        """Sets the model learning rate.

        Args:
            lr (float): The learning rate to set the model optimizer.
        """
        self._maybe_create_model()
        self._maybe_compile()
        self.lr = lr
        K.set_value(self.model_.optimizer.lr, lr)

    def set_trainable(self, trainable):
        """Freezes or unfreezes certain parts of the model.

        Args:
            trainable (bool, dict or list of bools): Can be either of:
              * bool: Set all layers in the model to trainable (True) or frozen (False).
              * dict: A dict(str -> bool) mapping layer name to that layer's trainable state.
              * list: A list of booleans mapping one-to-one to that model's layers trainable state.

        Raises:
            ValueError: Invalid type for 'trainable'..
        """
        check_is_fitted(self, 'model_', msg="ImageClassifier::set_trainable(): Model not yet constructed, use constructor to set trainable.")
        # * str: A CSV combination of {'features', 'head'}. Specifies which layers are
        #        to be unfrozen, all others will be frozen.
        # TODO: Subclass this properly and enable the string specialization.
        # if isinstance(trainable, str):
        #     parts = trainable.split(',')
        #     valid_parts = ['head', 'features']
        #     if not all([part in valid_parts for part in parts]):
        #         raise ValueError("ImageClassifier::set_trainable(): Invalid string value in 'trainable'.")
        #     n_layers = len(self.model_.layers)
        #     train_layers = [False for layer in range(n_layers)]
        #     for part in parts:
        #         if part == "features":
        #             train_layers[:-1] = [True] * (n_layers - 1)
        #         elif part == "head":
        #             train_layers[-1] = True
        #     super().set_trainable(train_layers)
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
        '''If the model has not been created, create and compile the model.

        Raises:
            ValueError: If no loss function is specified.
        '''
        check_is_fitted(self, "model_")
        if not self.model_._is_compiled:
            # Don't compile models that are not trainable (predict only)
            if self.trainable:
                if self.loss is None:
                    raise ValueError(
                        "ImageClassifier::fit(): Trying to compile a model without a loss function.")
                self.save_model_= self.model_
                if self.gpus and self.gpus > 1:
                     self.model_ = multi_gpu_model(self.model_, self.gpus)
                self.model_.compile('nadam', loss=self.loss, metrics=self.metrics)
                self.set_lr(self.init_lr)
            else:
                warnings.warn(
                    "Trying to compile a model that has no trainable layers.",
                    category=UserWarning)

    def recompile(self):
        '''Temporarily saves the model weights and config, deletes the model and restores it.
        Required before freezing layers.

        Raises:
            ValueError: If no loss function is specified.
        '''
        check_is_fitted(self, "model_")
        if not self.model_._is_compiled:
            return
        weights = self.model_.get_weights()
        self._maybe_create_model(force=True)
        self.model_.set_weights(weights)
        self._maybe_compile()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
            callbacks=None, validation_split=0.0, validation_data=None,
            shuffle=True, class_weight=None, sample_weight=None,
            steps_per_epoch=None, validation_steps=None):
        '''Fit numpy arrays to the model.

        Args:
            x (None, optional): Description
            y (None, optional): Description
            batch_size (None, optional): Description
            epochs (int, optional): Description
            verbose (int, optional): Description
            callbacks (None, optional): Description
            validation_split (float, optional): Description
            validation_data (None, optional): Description
            shuffle (bool, optional): Description
            class_weight (None, optional): Description
            sample_weight (None, optional): Description
            steps_per_epoch (None, optional): Description
            validation_steps (None, optional): Description

        Returns:
            TYPE: Description
        '''
        self._maybe_create_model()
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
        '''Fit a generator to the model.

        Args:
            generator (TYPE): Description
            steps_per_epoch (None, optional): Description
            epochs (int, optional): Description
            verbose (int, optional): Description
            callbacks (None, optional): Description
            validation_data (None, optional): Description
            validation_steps (None, optional): Description
            class_weight (None, optional): Description
            max_queue_size (int, optional): Description
            workers (int, optional): Description
            use_multiprocessing (bool, optional): Description
            shuffle (bool, optional): Description

        Returns:
            TYPE: Description
        '''
        self._maybe_create_model()
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
        """Summary

        Args:
            dataset_train (TYPE): Description
            dataset_val (None, optional): Description
            steps_per_epoch (None, optional): Description
            epochs (int, optional): Description
            verbose (int, optional): Description
            callbacks (None, optional): Description
            validation_steps (None, optional): Description
            class_weight (None, optional): Description
            max_queue_size (int, optional): Description
            workers (int, optional): Description
            use_multiprocessing (bool, optional): Description
            shuffle (bool, optional): Description

        Returns:
            TYPE: Description
        """
        self._maybe_create_model()
        self._maybe_compile()
        params = dict(
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            class_weight=class_weight,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            shuffle=shuffle)
        if dataset_val is not None:
            params.update(dict(
                validation_data=dataset_val.generator(endless=True),
                validation_steps=validation_steps))
        return self.fit_generator(
            dataset_train.generator(endless=True), **params)

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

        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = (proba > 0.5).astype('int32')
        return self.classes_[classes]

    @staticmethod
    def load(filepath):
        """Load a model, its state and its hyperparameters from file.

        Args:
            filepath (TYPE): Path to model to load
        """

        return ModelPersistence._load_model(filepath, ImageClassifier)



class FcnCrfSegmenter(ImageClassifier, ModelPersistence):

    """A jointly trained FCN and CRF that complete with persistance
    functionality, and hides some complex implementation details.
    Inherits some methods from ImageClassifier.
    """

    def __init__(self,
                 output_activation='softmax',
                 input_shape=(500, 500, 3), pooling='avg', classes=2,
                 crf_iterations=None,
                 init_weights='imagenet', init_epoch=0, init_lr=1e-3,
                 trainable=True, loss='categorical_crossentropy'):
        """Summary

        Args:
            output_activation (str, optional): Description
            input_shape (tuple, optional): Description
            pooling (str, optional): Description
            classes (int, optional): Description
            init_weights (str, optional): Description
            init_epoch (int, optional): Description
            init_lr (float, optional): Description
            trainable (bool, optional): Description
            loss (str, optional): Description
        """
        self.loss = loss
        self.output_activation = output_activation
        self.input_shape = tuple(input_shape)
        if self.input_shape != (500, 500, 3):
            raise ValueError("input_shape must be (500, 500, 3). Got {}".format(str(self.input_shape)))
        self.pooling = pooling
        self.classes = classes
        self.trainable = trainable
        self.init_weights = init_weights
        self.init_lr = init_lr
        self.init_epoch = init_epoch
        self.crf_iterations = crf_iterations

    def _maybe_create_model(self, force=False):
        """Create the model if it has not already been created

        Raises:
            ValueError: If backbone is invalid.
        """
        from crfrnn.crfrnn_model import get_crfrnn_model_def
        from keras.models import Model
        from keras.layers import Dense
        from abyss_deep_learning.keras.utils import initialize_conv_transpose2d

        if not force and hasattr(self, "model_"):
            return

        # Load the model with imagenet weights, they will be re-initialized later weights=None
        model = get_crfrnn_model_def(
            num_classes=self.classes, input_shape=self.input_shape,
            num_iterations=self.crf_iterations, with_crf=bool(self.crf_iterations))

#         initialize_conv_transpose2d(
#             model, ['score2', 'score4', 'upsample'], trainable=True)
        if bool(self.crf_iterations):
            crf_params = [
                'crfrnn/spatial_ker_weights:0',
                'crfrnn/bilateral_ker_weights:0',
                'crfrnn/compatibility_matrix:0']
            model.get_layer(name='crfrnn').set_weights([
                np.eye(self.classes), np.eye(self.classes), 1 - np.eye(self.classes)
            ])

        self.model_ = model
        self.classes_ = np.arange(self.classes) # Sklearn API recomendation
        self.set_trainable(self.trainable)
        if self.init_weights != 'imagenet':
            self.set_weights(self.init_weights)

    def set_trainable(self, trainable):
        """Freezes or unfreezes certain parts of the model.

        Args:
            trainable (bool, dict or list of bools): Can be either of:
              * str: A CSV combination of {'fcn', 'crf'}. Specifies which layers are
                     to be unfrozen, all others will be frozen.
              * bool: Set all layers in the model to trainable (True) or frozen (False).
              * dict: A dict(str -> bool) mapping layer name to that layer's trainable state.
              * list: A list of booleans mapping one-to-one to that model's layers trainable state.

        Raises:
            ValueError: Invalid type for 'trainable'..
        """

        check_is_fitted(self, 'model_',
            msg="FcnCrfSegmenter::set_trainable(): Model not yet constructed, use constructor to set trainable.")
        if isinstance(trainable, str):
            parts = trainable.split(',')
            valid_parts = ['fcn', 'crf']
            if not all([part in valid_parts for part in parts]):
                raise ValueError("FcnCrfSegmenter::set_trainable(): Invalid string value in 'trainable'.")
            n_layers = len(self.model_.layers)
            train_layers = [False for layer in range(n_layers)]
            for part in parts:
                if part == "fcn":
                    if self.crf_iterations:
                        train_layers[:-1] = [True] * (n_layers - 1)
                    else:
                        train_layers = [True] * n_layers
                elif part == "crf":
                    if not self.crf_iterations:
                        raise ValueError(
                            "FcnCrfSegmenter::set_trainable(): Given string 'crf' but no CRF is present in this model.")
                    train_layers[-1] = True
            super().set_trainable(train_layers)
        else:
            super().set_trainable(trainable)

    @staticmethod
    def load(filepath):
        """Load a model, its state and its hyperparameters from file.

        Args:
            filepath (TYPE): Path to model to load
        """
        return ModelPersistence._load_model(filepath, FcnCrfSegmenter)
