#!/usr/bin/env python3

import os
import argparse
import json
import warnings
import ast
from pycocotools.coco import COCO
import numpy as np
import random
import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


from abyss_deep_learning.datasets.coco import ImageSemanticSegmentationDataset
from abyss_deep_learning.datasets.translators import CategoryTranslator
from abyss_deep_learning.keras.utils import lambda_gen

# TODO change to import the abyss_deep_learning.keras.segmentation.Task class (?)
from abyss_deep_learning.keras.zoo.segmentation.deeplab_v3_plus import Deeplabv3

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
    Train a segmentation network
    """)
    parser.add_argument("coco_path", type=str, help="Path to the coco dataset")
    parser.add_argument("--val-coco-path", type=str, help="Path to the validation coco dataset")
    args = parser.parse_args()
    return args

def main(args):
    raise NotImplementedError("TODO")

if __name__ == "__main__":
    main(get_args())





