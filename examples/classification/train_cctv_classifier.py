#!/usr/bin/env python3
import argparse
import keras
import keras.backend as K
import numpy as np
import json
import os
from skimage.transform import resize
import keras.callbacks
import matplotlib.pyplot as plt
from keras.applications.xception import preprocess_input
from keras.utils import to_categorical
from abyss_deep_learning.datasets.coco import ImageClassificationDataset, ClassificationTask
from abyss_deep_learning.datasets.translators import AbyssCaptionTranslator, CaptionMapTranslator, AnnotationTranslator
from keras.callbacks import TensorBoard

from abyss_deep_learning.keras.models import ImageClassifier
from abyss_deep_learning.keras.utils import batching_gen, lambda_gen, gen_dump_data
from abyss_deep_learning.keras.classification import (onehot_gen, augmentation_gen)

from callbacks import SaveModelCallback, PrecisionRecallF1Callback
from abyss_deep_learning.keras.tensorboard import ImprovedTensorBoard
# from metrics import f1_m, recall_m, precision_m
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

# TODO use this to replace multihot gen. abyss_deep_learning.datasets.coco.py needs to be fixed first.
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

def get_args():
        parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
        This script is designed to test loading a coco image classification dataset
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

def postprocess(image):
    """
    Converts an image from a float with range -1,1 to a rgb image.
    Args:
        image: (float np array) an image with range -1,1
    Returns: (uint8 np array) a rgb8 image
    """
    return ((image + 1) * 127.5).astype(np.uint8)

def main(args):
    # Set up logging and scratch directories
    os.makedirs(args.scratch_dir, exist_ok=True)
    model_dir = os.path.join(args.scratch_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(args.scratch_dir, 'logs')

    # do the caption translations and any preprocessing set-up
    caption_map = json.load(open(args.caption_map, 'r')) # Load the caption map - caption_map should live on place on servers
    caption_translator = CaptionMapTranslator(mapping=caption_map) # Initialise the translator
    num_classes = len(set(caption_map.values())) # Get num classes from caption map
    hot_translator = HotTranslator(num_classes) # Hot translator encodes as a multi-hot vector
    translator = MultipleTranslators([caption_translator, hot_translator])# Apply multiple translators
    image_shape = [int(x) for x in args.image_shape.split(',')]

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

    # Create the dataset
    dataset = ImageClassificationDataset(args.coco_path, translator=caption_translator)

    # Create the validation dataset
    if args.val_coco_path:
        val_dataset = ImageClassificationDataset(args.val_coco_path, translator=caption_translator)
    else:
        val_dataset = None

    # Create the pipeline
    def pipeline(gen, num_classes, batch_size):
        """
        A sequence of generators that perform operations on the data
        Args:
            gen: the base generator (e.g. from dataset.generator())
            num_classes: (int) the number of classes, to create a multihot vector
            batch_size: (int) the batch size, for the batching generator

        Returns:

        """
        return (batching_gen(multihot_gen(lambda_gen(gen, func=preprocess), num_classes=num_classes), batch_size=batch_size))

    def val_pipeline(gen, num_classes):
        """
        A sequence of generators that perform operations on the data.
        The val pipeline doesn't have batching_gen as it is getting cached. Also augmentation_gen would be skipped as well.

        Args:
            gen:  the base generator (e.g. from dataset.generator())
            num_classes: (int) the number of classes, to create a multihot vector

        Returns:

        """
        return (multihot_gen(lambda_gen(gen, func=preprocess), num_classes=num_classes))

    classifier = ImageClassifier(
        backbone='xception',
        output_activation='softmax',
        pooling='avg',
        classes=2,
        input_shape=tuple(image_shape),
        init_weights='imagenet',
        init_epoch=0,
        init_lr=1e-3,
        trainable=True,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # A callback to save the model
    save_callback = SaveModelCallback(classifier.save, model_dir)
    # A tensorboard callback
    tb_callback = ImprovedTensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=args.batch_size, write_graph=True,
                                      write_grads=True, num_classes=num_classes, pr_curve=True)
    # Construct a list of callbacks to send the model
    callbacks = [save_callback, tb_callback]
    # Dump the validation data into a tuple(x,y). Necessary to get PR Curve.
    if val_dataset is not None:
        val_data = gen_dump_data(gen=val_pipeline(val_dataset.generator(endless=True), num_classes),
                                 num_images=len(val_dataset))
        # A Precision Recall F1 Score Callback
        prf1_callback = PrecisionRecallF1Callback(val_data)
        callbacks.append(prf1_callback)

    training_pipeline = pipeline(dataset.generator(endless=True), num_classes, args.batch_size)
    # Train the classifier
    classifier.fit_generator(generator=training_pipeline,  # The generator wrapped in the pipline loads x,y
                             validation_data= val_data if val_dataset is not None else None,  # Pass in the validation data array
                             validation_steps= np.floor(val_data[0].shape[0] / args.batch_size) if val_dataset is not None else None,  # Validate on the entire val set
                             epochs=args.epochs,
                             verbose=1,
                             shuffle=True,
                             steps_per_epoch=np.floor(len(dataset) / args.batch_size),
                             callbacks=callbacks)
    K.clear_session()
    print("Done training image classification network.\n")

if __name__ == "__main__":
    main(get_args())
