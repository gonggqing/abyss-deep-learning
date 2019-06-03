import os
import argparse
import json
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.xception import preprocess_input
#import keras.callbacks
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, Callback, TerminateOnNaN
import tensorflow as tf

from abyss_deep_learning.datasets.coco import ImageClassificationDataset
from abyss_deep_learning.datasets.translators import  AbyssCaptionTranslator, CaptionMapTranslator, AnnotationTranslator
from abyss_deep_learning.keras.classification import caption_map_gen, onehot_gen
from abyss_deep_learning.keras.models import ImageClassifier
from abyss_deep_learning.keras.utils import lambda_gen, batching_gen
NN_DTYPE = np.float32
def to_multihot(captions, num_classes):
    """
    Converts a list of classes (int) to a multihot vector
    Args:
        captions: (list of ints). Each class in the caption list
        num_classes: (int) The total number of classes

    Returns:

    """
    hot = np.zeros([num_classes])
    if isinstance(captions, int):
        hot[captions] = 1
    else:
        for c in captions:
            hot[int(c)] = 1
    return hot
def multihot_gen(gen, num_classes):
    """A stream modifier that converts categorical labels into one-hot vectors.

    Args:
        gen (generator): A keras compatible generator where the targets are a list of categorical labels.
        num_classes (int): Total number of categories to represent.

    Yields:
        generator: A keras compatible generator with the targets modified.
    """
    for image, captions in gen:
        yield image, to_multihot(captions, num_classes)
def compute_class_weights(dataset):
    '''
    computes the ideal weights from each class based on the frequency of each class.
    For example, if there are 12.5 times more of class 0 than class 1, then returns {0: 12.5,
                                                                                     1: 1.0}
    '''
    dataset._calc_class_stats()
    min_val = dataset.stats['class_weights'][
        min(dataset.stats['class_weights'].keys(), key=(lambda k: dataset.stats['class_weights'][k]))]
    return dataset.stats['class_weights'].update((x,y/min_val) for x,y in dataset.stats['class_weights'].items())

'''
implementation from:
# https://github.com/keras-team/keras/issues/5400
 you can use it like this
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=[mcor, recall, f1])
'''
from keras import backend as K
def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

class HotTranslator(AnnotationTranslator):
    """
    A translator to convert annotations to multihot encoding
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def filter(self, annotation):
        """
        Filters the annotations
        """
        return True
    def translate(self, annotation):
        """
        Translates the annotation into a multihot vector
        Args:
            annotation:

        Returns:

        """
        return to_multihot(annotation, self.num_classes)
class MultipleTranslators(AnnotationTranslator):
    """
    Used when multiple sequential translations are needed to transform the annotations
    """
    def __init__(self, translators):
        for tr in translators:
            assert isinstance(tr, (AnnotationTranslator, type(None)))
        self.translators = translators
    def filter(self, annotation):
        """
        Filters the annotations
        """
        for tr in self.translators:
            if not tr.filter(annotation):
                return False
        return True
    def translate(self, annotation):
        """
        Translates the annotation
        """
        for tr in self.translators:
            annotation = tr.translate(annotation)
        return annotation
class SaveModelCallback(Callback):
    """
    Saves the model at the end of an epoch. To be used with an ImageClassifier class.

    Usage:
        callbacks = [SaveModelCallback(classifier.save, 'models')]
    """
    def __init__(self, save_fn, model_dir, save_interval=1):
        self.model_dir = model_dir
        self.save_fn = save_fn
        self.save_interval = save_interval
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.save_interval == 0:
            self.save_fn(os.path.join(self.model_dir, 'model_%d.h5'%epoch))
        return
    def on_train_end(self, epoch):
        self.save_fn(os.path.join(self.model_dir, 'best_model.h5'))
class TrainValTensorBoard(TensorBoard):
    '''
    Additional and improved logging parameters with tensorboard. Wraps around tensorboard callback.
    adapted from:
    https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
    '''
    def __init__(self, log_dir='./logs', update_freq='epoch', update_step=1, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        if update_freq not in ['epoch', 'batch']:
            raise ValueError("TrainValTensorBoard callback update frequency is invalid. "
                             "Select either: update_freq = epoch or batch.")
        self.update_freq = update_freq
        self.update_step = update_step
        self.losses = [] # this is here accumulate metrics every batch, rather than the default on_epoch_end
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)
    def handle_validation(self, count, logs=None):
        '''
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        takes the logs, and processing any validation ones so they will be plotted on the same tensorboard graph as
        their training counterparts. Any remaining, no processed logs are returned.

        '''
        logs = logs or {}
        if self.update_freq is 'batch':
            val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, count)
            self.val_writer.flush()
        return {k: v for k, v in logs.items() if not k.startswith('val_')}
    def on_epoch_end(self, epoch, logs=None):
            logs = self.handle_validation(epoch, logs)
            logs.update({'learning_rate': K.eval(self.model.optimizer.lr)}) # additioanlly log learning rate
            super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)
    def on_batch_end(self, batch, logs=None):
        if batch % self.update_step is 0 or self.update_freq is 'epoch':
            logs = self.handle_validation(batch, logs)
            # Pass the remaining logs to `TensorBoard.on_epoch_end`
            super(TrainValTensorBoard, self).on_epoch_end(epoch=batch, logs=logs)
    def on_train_end(self, logs=None):
        self.val_writer.close()

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
    This script is designed to get eric going with CCTV training.
    """)
    parser.add_argument("coco_path", type=str, help="Path to the coco dataset")
    parser.add_argument("--val-coco-path", type=str, help="Path to the validation coco dataset")
    parser.add_argument("--scratch_dir", type=str, default="scratch/", help="Where to save models, logs, etc.")
    parser.add_argument("--caption-map", type=str, help="Path to the caption map")
    parser.add_argument("--image-shape", type=str, default="320,240,3", help="Image shape")
    parser.add_argument("--batch-size", type=int, default=2, help="Image shape")
    parser.add_argument("--epochs", type=int, default=2, help="Image shape")
    args = parser.parse_args()
    return args

def main(args):
    # Set up logging and scratch directories
    os.makedirs(args.scratch_dir, exist_ok=True)
    model_dir = os.path.join(args.scratch_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(args.scratch_dir, 'logs')

    # do the caption translations and any preprocessing set-up
    caption_map = json.load(open(args.caption_map, 'r'))  # Load the caption map - caption_map should live on place on servers
    caption_translator = CaptionMapTranslator(mapping=caption_map)  # Initialise the translator
    num_classes = len(set(caption_map.values()))  # Get num classes from caption map
    hot_translator = HotTranslator(num_classes)  # Hot translator encodes as a multi-hot vector
    translator = MultipleTranslators([caption_translator, hot_translator])  # Apply multiple translators
    image_shape = [int(x) for x in args.image_shape.split(',')]

    train_dataset = ImageClassificationDataset(args.coco_path, translator=caption_translator)
    train_gen = train_dataset.generator(endless=True, shuffle_ids=True)
    if args.val_coco_path:
        val_dataset = ImageClassificationDataset(args.val_coco_path, translator=caption_translator)
        val_gen = val_dataset.generator(endless=True, shuffle_ids=True)
    else:
        val_gen = None

    def preprocess(image, caption):
        """
        A preprocessing function to resize the image
        Args:
            image: (np.ndarray) The image
            caption: passedthrough

        Returns:
            image, caption

        """
        image = resize(image, image_shape, preserve_range=True)
        return preprocess_input(image.astype(NN_DTYPE)), caption
    def pipeline(gen, num_classes, batch_size):
        """
        A sequence of generators that perform operations on the data
        Args:
            gen: the base generator (e.g. from dataset.generator())
            num_classes: (int) the number of classes, to create a multihot vector
            batch_size: (int) the batch size, for the batching generator

        Returns:

        """
        return (batching_gen(multihot_gen(lambda_gen(gen, func=preprocess), num_classes=num_classes),
                             batch_size=batch_size))

    # limit the process GPU usage. Without this, eric gets CUDNN_STATUS_INTENERAL_ERROR
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    # create classifier model
    classifier = ImageClassifier(
        backbone='xception',
        output_activation='softmax',
        pooling='avg',
        classes=num_classes,
        input_shape=tuple(image_shape),
        init_weights='imagenet',
        init_epoch=0,
        init_lr=1e-3,
        trainable=True,
        loss='categorical_crossentropy',
        metrics=['accuracy', mcor, recall, f1, precision]
    )

    ## callbacks
    callbacks = [SaveModelCallback(classifier.save, model_dir, save_interval=5),  # A callback to save the model
                 TrainValTensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=32, write_graph=True,
                                             write_grads=True, write_images=False, embeddings_freq=0,
                                             embeddings_layer_names=None, embeddings_metadata=None,
                                             embeddings_data=None, update_freq='batch', update_step=10),
                 ReduceLROnPlateau(monitor='acc', factor=0.2,
                                   patience=5, min_lr=1e-4),
                 EarlyStopping(monitor='acc', min_delta=1e-4, patience=6, verbose=1, mode='auto',
                                               baseline=None, restore_best_weights=True),
                 TerminateOnNaN()
                 ]

    train_steps = np.floor(len(train_dataset) / args.batch_size)
    val_steps = np.floor(len(val_dataset) / args.batch_size) if val_dataset is not None else None
    class_weights = compute_class_weights(train_dataset)
    classifier.fit_generator(generator=pipeline(train_gen, num_classes=num_classes, batch_size=args.batch_size),  # The generator wrapped in the pipline loads x,y
                             steps_per_epoch=train_steps,
                             # validation_data= val_data if val_dataset is not None else None,  # Pass in the validation data array
                             validation_data=pipeline(val_gen, num_classes=num_classes, batch_size=args.batch_size),
                             validation_steps=val_steps,
                             epochs=args.epochs,
                             class_weight=class_weights,
                             verbose=1,
                             shuffle=True,
                             callbacks=callbacks)

if __name__ == "__main__":
    main(get_args())
