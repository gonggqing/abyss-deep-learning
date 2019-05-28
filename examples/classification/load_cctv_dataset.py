#!/usr/bin/env python3
import argparse
import keras
import keras.backend as K
import numpy as np
import json
import matplotlib.pyplot as plt
from keras.applications.xception import preprocess_input
from keras.utils import to_categorical
from abyss_deep_learning.datasets.coco import ImageClassificationDataset, ClassificationTask
from abyss_deep_learning.datasets.translators import AbyssCaptionTranslator, CaptionMapTranslator, AnnotationTranslator

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

    # Create the dataset - how to use with translator???
    dataset = ImageClassificationDataset(args.coco_path, translator=caption_translator)

    for i,(x,y) in enumerate(dataset.generator()):
        print(x.shape)
        print(y)

if __name__ == "__main__":
    main(get_args())
