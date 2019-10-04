'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, target) and not in a batch.
'''
import os
import json
from itertools import cycle
from random import shuffle
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

from abyss_deep_learning.utils import cat_to_onehot, warn_on_call

def sequential_process(images):
    # images = np.array([
    #     skimage.color.rgb2grey(image)
    #     for image in images])

    return images.astype("float") / 255.0

class ModelPersistence: # todo: refactor
    def save(self, filepath):
        """Save a model, its state and its hyperparameters to file.

        Args:
            filepath (TYPE): Description
            include_optimizer (bool, optional): Description
        """
        from keras.engine.saving import save_weights_to_hdf5_group#, _serialize_model
        from json import dumps
        import h5py

        self.save_model_.save_weights(filepath)
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

def loadTaskByDict(json_path):
    """
    Allows the user to instantiate the Image Classifier using a dictionary of arguments. The same one that is created by Task.dump_args.

    This is useful for loading the classifier for validation/prediction.

    Args:
        json_path: path to the params json

    Returns:
        Task: The initialised Image Classifier.

    """
    params = json.load(open(json_path, 'r'))

    if params['metrics']:
        mets = params['metrics'].split(',')
    else:
        mets = None

    return Task(
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
        optimizer=params['optimizer'],
        optimizer_args=params['optimizer_args'],
        metrics=mets
    )


class Task(BaseEstimator, ClassifierMixin, ModelPersistence):

    """A generic image classifier that can use multiple backends, complete with persistance
    functionality, and hides some complex implementation details.
    """

    def __init__(self,
                 backbone='xception',
                 output_activation='softmax',
                 input_shape=(299, 299, 3),
                 pooling='avg',
                 classes=2,
                 init_weights='imagenet',
                 init_epoch=0,
                 init_lr=1e-3,
                 trainable=True,
                 optimizer='adam',
                 optimizer_args={},
                 loss='categorical_crossentropy',
                 metrics=None,
                 gpus=None,
                 l12_reg=(None,None)):
        """Summary

        Args:
            backbone (string): One of {'xception'}.
            output_activation (str, optional): The activation to use on the final layer.
            input_shape (tuple, optional): The input image shape in (Width,Height,Channels)
            pooling (str, optional): The pooling to use. One of {'avg','max'}
            classes (int, optional): The number of classes
            init_weights (str, optional): The weights used to initialise the network. Options are {'imagenet','/path/to/weights.h5'}
            init_epoch (int, optional): The starting epoch for training.
            init_lr (float, optional): The initial learning rate.
            trainable (bool, dict, list, optional): Whether to set layers to trainable. Bool to set the entire network to trainable/not. Dict mapping each layer to its trainable state. List of bools mapping the layer sequence to the trainable state.
            optimizer (str, optional): The optimizer to use. Can either be a string and one of {'adam', 'nadam', 'sgd', 'adagrad, 'adadelta', 'adamax'}.
            optimizer_args (dict, optional): The optimizer arguments to use.
            loss (str, optional): The loss function to use. Can also pass in custom losses as function handles.
            metrics (list, optional): The list of metrics to use.
            gpus (int, optional): The number of GPUs to use.
            l12_reg (list, optional): The L1 and L2 regularisation to use. Defaults to None.
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
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

        # For adding regularisation
        self.l12_reg = l12_reg

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
        params['optimizer'] = self.optimizer
        params['optimizer_args'] = self.optimizer_args
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


    def add_regularisation(self, l1=None, l2=None):
        """
        Add regularisation to the Convolution layers
        Args:
            l1: (float) - the l1 penalty to apply
            l2: (float) - the l2 penalty to apply

        Returns:
            None
        """
        # Iterate through the layers of the model
        for layer in self.model_.layers:
            # Only add it to the conv layers. This will include separable convs as well
            if "conv" in layer.name:
                # Add l1 and l2 regularisation
                if l1 and l2:
                    w_reg = keras.regularizers.l1_l2(l1=l1, l2=l2)
                # Add l1 regularisation
                elif l1 and not l2:
                    w_reg = keras.regularizers.l1(l1)
                # Add l2 regularisation
                elif l2 and not l1:
                    w_reg = keras.regularizers.l2(l2)
                else:
                    w_reg = None
                # Set the regularisation
                if w_reg:
                    layer.W_regularizer = w_reg


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
            raise ValueError("Task::set_weights(): Invalid weights.")

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

        if self.backbone == 'xception':
                model = Xception(
                        include_top=config['include_top'],
                        weights=config['weights'],
                        input_shape=config['input_shape'],
                        pooling=config['pooling'])
        else:
            raise ValueError(
                "Task::__init__(): Invalid backbone '{}'".format(self.backbone))

        # Add the classification head
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

    def  set_lr(self, lr):
        """Sets the model learning rate.

        Args:
            lr (float): The learning rate to set the model optimizer.
        """
        self._maybe_create_model()
        self._maybe_compile()
        self.lr = lr
        K.set_value(self.model_.optimizer.lr, lr)

    def create_optimizer(self, optimizer, optimizer_args):
        if 'lr' not in optimizer_args:
            optimizer_args['lr'] = self.init_lr
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(**optimizer_args)
        elif optimizer == 'nadam':
            opt = keras.optimizers.Nadam(**optimizer_args)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(**optimizer_args)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(**optimizer_args)
        elif optimizer == 'adagrad':
            opt = keras.optimizers.Adagrad(**optimizer_args)
        elif optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(**optimizer_args)
        elif optimizer == 'adamax':
            opt = keras.optimizers.Adamax(**optimizer_args)
        else:
            raise ValueError("Optimizer selection not valid")
        return opt

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
        check_is_fitted(self, 'model_', msg="Task::set_trainable(): Model not yet constructed, use constructor to set trainable.")
        # * str: A CSV combination of {'features', 'head'}. Specifies which layers are
        #        to be unfrozen, all others will be frozen.
        # TODO: Subclass this properly and enable the string specialization.
        # if isinstance(trainable, str):
        #     parts = trainable.split(',')
        #     valid_parts = ['head', 'features']
        #     if not all([part in valid_parts for part in parts]):
        #         raise ValueError("Task::set_trainable(): Invalid string value in 'trainable'.")
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
                "Task::set_trainable(): Invalid argument type (accepts bool, dict and list).")
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
                        "Task::fit(): Trying to compile a model without a loss function.")
                self.save_model_= self.model_
                if self.gpus and self.gpus > 1:
                    self.model_ = multi_gpu_model(self.model_, self.gpus)
                opt = self.create_optimizer(self.optimizer, self.optimizer_args)
                self.model_.compile(opt, loss=self.loss, metrics=self.metrics)
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

        return ModelPersistence._load_model(filepath, Task)

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
