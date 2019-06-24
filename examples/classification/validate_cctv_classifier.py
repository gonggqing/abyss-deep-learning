#!/usr/bin/env python3

import keras.models
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
from abyss_deep_learning.keras.models import ImageClassifier, loadImageClassifierByDict
from abyss_deep_learning.keras.utils import lambda_gen, batching_gen

from callbacks import SaveModelCallback, PrecisionRecallF1Callback, TrainValTensorBoard
from utils import to_multihot
from translators import MultipleTranslators, HotTranslator
from abyss_deep_learning.keras.tensorboard import ImprovedTensorBoard
from utils import multihot_gen, compute_class_weights

import keras.backend as K

NN_DTYPE = np.float32


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
    This script is to show how to visualise the data
    """)
    parser.add_argument("coco_path", type=str, help="Path to the coco dataset")
    parser.add_argument("--logdir", type=str, help="Path to keras model")
    parser.add_argument("--checkpoint", type=int, help="The checkpoint to load from. Usually corresponds to epoch")
    parser.add_argument("--visual", "--vis", action="store_true", help="Show a selection of images, classes and predictions")
    parser.add_argument("--num-samples", type=int, default=10, help="Show a selection of images, classes and predictions")
    parser.add_argument("--image-shape", type=str, default="320,240,3", help="Comma seperated image shape, e.g. 320,240,3")
    parser.add_argument("--caption-map", type=str, help="Path to the caption map file")

    args = parser.parse_args()
    return args

def main(args):
    image_shape = [int(x) for x in args.image_shape.split(',')]

    # do the caption translations and any preprocessing set-up
    caption_map = json.load(open(args.caption_map, 'r'))  # Load the caption map - caption_map should live on place on servers
    caption_translator = CaptionMapTranslator(mapping=caption_map)  # Initialise the translator

    dataset = ImageClassificationDataset(args.coco_path, translator=caption_translator)
    num_classes = len(set(caption_map.values()))  # Get num classes from caption map

    ####################################################################################
    #################### Following code needed to manage new cudnn error ###############
    ####################################################################################

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    ####################################################################################


    classifier = ImageClassifier.load(os.path.join(args.logdir, 'models', 'model_%d.h5'%args.checkpoint))


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

    preds = []
    trues = []
    for batch, (inp, tgt) in enumerate(pipeline(dataset.generator(endless=True), num_classes=num_classes, batch_size=1)):
        if batch >= args.num_samples:
            break
        pred = int(np.squeeze(classifier.predict(inp, batch_size=1),axis=0))
        preds.append(pred)
        trues.append(np.argmax(np.squeeze(tgt, axis=0)))



    print(preds)
    print(trues)

    K.clear_session()





if __name__ == "__main__":
    main(get_args())
