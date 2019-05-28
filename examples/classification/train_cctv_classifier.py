#!/usr/bin/env python3
import argparse
import keras
import keras.backend as K
import numpy as np
import json
from skimage.transform import resize

import matplotlib.pyplot as plt
from keras.applications.xception import preprocess_input
from keras.utils import to_categorical
from abyss_deep_learning.datasets.coco import ImageClassificationDataset, ClassificationTask
from abyss_deep_learning.datasets.translators import AbyssCaptionTranslator, CaptionMapTranslator, AnnotationTranslator

from abyss_deep_learning.keras.models import ImageClassifier
from abyss_deep_learning.keras.utils import batching_gen, lambda_gen
from abyss_deep_learning.keras.classification import (onehot_gen, augmentation_gen)

NN_DTYPE = np.float32

def to_multihot(captions, num_classes):
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
    def __init__(self, num_classes):
        self.num_classes = num_classes
    def filter(self, annotation):
        return True
    def translate(self, annotation):
        return keras.utils.to_categorical(annotation, self.num_classes)

class MultipleTranslators(AnnotationTranslator):
    def __init__(self, translators):
        for tr in translators:
            assert isinstance(tr, (AnnotationTranslator, type(None)))
        self.translators = translators
    def filter(self, annotation):
        for tr in self.translators:
            if not tr.filter(annotation):
                return False
        return True
    def translate(self, annotation):
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

def preprocess(image, caption):
    image = resize(image, image_dims, preserve_range=True)
    return preprocess_input(image.astype(NN_DTYPE)), caption



def main(args):
    # Load the caption map - caption_map should live on place on servers
    caption_map = json.load(open(args.caption_map, 'r'))
    # Initialise the translator
    caption_translator = CaptionMapTranslator(mapping=caption_map)
    # Get num classes from caption map
    num_classes = len(set(caption_map.values()))
    # Hot translator encodes as a multi-hot vector
    hot_translator = HotTranslator(num_classes)
    # Apply multiple translators
    translator = MultipleTranslators([caption_translator, hot_translator])

    image_shape = [int(x) for x in args.image_shape.split(',')]

    def preprocess(image, caption):
        image = resize(image, image_shape, preserve_range=True)
        return preprocess_input(image.astype(NN_DTYPE)), caption

    # Create the dataset - how to use with translator
    dataset = ImageClassificationDataset(args.coco_path, translator=caption_translator)

    # Create the pipeline
    def pipeline(gen, num_classes, batch_size):
        return (batching_gen(multihot_gen(lambda_gen(gen, func=preprocess), num_classes=num_classes), batch_size=batch_size))

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
        loss='categorical_crossentropy'
    )

    # for x,y in pipeline(dataset.generator(), dataset.num_classes):
    #     print(y.shape)
    classifier.fit_generator(generator=pipeline(dataset.generator(), num_classes, args.batch_size),
                             epochs=args.epochs,
                             verbose=1,
                             shuffle=True,
                             steps_per_epoch=3)

if __name__ == "__main__":
    main(get_args())
