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

from callbacks import SaveModelCallback, PrecisionRecallF1Callback, TrainValTensorBoard
from utils import to_multihot, multihot_gen, compute_class_weights
from translators import MultipleTranslators, HotTranslator
from abyss_deep_learning.keras.tensorboard import ImprovedTensorBoard

NN_DTYPE = np.float32


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
    This script is designed to get eric going with CCTV training.
    """)
    parser.add_argument("coco_path", type=str, help="Path to the coco dataset")
    parser.add_argument("--val-coco-path", type=str, help="Path to the validation coco dataset")
    parser.add_argument("--scratch-dir", type=str, default="scratch/", help="Where to save models, logs, etc.")
    parser.add_argument("--caption-map", type=str, help="Path to the caption map")
    parser.add_argument("--image-shape", type=str, default="320,240,3", help="Image shape")
    parser.add_argument("--batch-size", type=int, default=2, help="Image shape")
    parser.add_argument("--epochs", type=int, default=2, help="Image shape")
    parser.add_argument("--lr", type=float, default=1e-4, help="Sets the learning rate of the optimizer")
    parser.add_argument("--save-model-interval", type=int, default=1, help="How often to save the model interval")
    parser.add_argument("--resume-from-ckpt", type=str, help="Resume the model from the given .h5 checkpoint, as saved by the ImageClassifier.save function. This loads in all weights and parameters from the previous training.")
    parser.add_argument("--weights", type=str, default='imagenet', help="Path to the weights to load into this model. Re-initalises all the checkpoints etc. Default is 'imagenet' which loads the 'imagenet' trained weights")
    parser.add_argument("--gpu-fraction", type=float, default=0.8, help="Limit the amount of GPU usage tensorflow uses. Defaults to 0.8")
    args = parser.parse_args()
    return args

def main(args):
    # Set up logging and scratch directories
    os.makedirs(args.scratch_dir, exist_ok=True)
    model_dir = os.path.join(args.scratch_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(args.scratch_dir, 'logs')

    # do the caption translations and any preprocessing set-up
    caption_map = json.load(open(args.caption_map, 'r'))# Load the caption map - caption_map should live on place on servers
    caption_translator = CaptionMapTranslator(mapping=caption_map, filter_by='category_id')# Initialise the translator
    num_classes = len(set(caption_map.values()))  # Get num classes from number of unique values in caption map
    hot_translator = HotTranslator(num_classes)  # Hot translator encodes as a multi-hot vector
    translator = MultipleTranslators([caption_translator, hot_translator])  # Apply multiple translators
    image_shape = [int(x) for x in args.image_shape.split(',')]

    train_dataset = ImageClassificationDataset(args.coco_path, translator=caption_translator, use_captions=True)
    train_gen = train_dataset.generator(endless=True, shuffle_ids=True)
    if args.val_coco_path:
        val_dataset = ImageClassificationDataset(args.val_coco_path, translator=caption_translator, use_captions=True)
        val_gen = val_dataset.generator(endless=True, shuffle_ids=True)
    else:
        val_gen = None
        val_dataset = None

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

    # limit the process GPU usage. Useful for
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
    set_session(tf.Session(config=config))
    # create classifier model
    if args.resume_from_ckpt:
        classifier = ImageClassifier.load(args.resume_from_ckpt)
    else:
        classifier = ImageClassifier(
            backbone='xception',
            output_activation='softmax',
            pooling='avg',
            classes=num_classes,
            input_shape=tuple(image_shape),
            init_weights=args.weights,
            init_epoch=0,
            init_lr=args.lr,
            trainable=True,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    classifier.dump_args(os.path.join(args.scratch_dir, 'params.json'))

    ## callbacks
    callbacks = [SaveModelCallback(classifier.save, model_dir, save_interval=10),  # A callback to save the model
                 ImprovedTensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=args.batch_size, write_graph=True,
                                     write_grads=True, num_classes=num_classes, pr_curve=True, val_generator=pipeline(val_gen, num_classes=num_classes, batch_size=1) if val_gen else None, val_steps=len(val_dataset), tfpn=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=5, min_lr=1e-8),
                 # EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=6, verbose=1, mode='auto',
                 #                               baseline=None, restore_best_weights=True),
                 TerminateOnNaN()
                 ]

    train_steps = np.floor(len(train_dataset) / args.batch_size)
    val_steps = np.floor(len(val_dataset) / args.batch_size) if val_dataset is not None else None
    class_weights = compute_class_weights(train_dataset)
    print("Using class weights: ", class_weights)
    classifier.fit_generator(generator=pipeline(train_gen, num_classes=num_classes, batch_size=args.batch_size),  # The generator wrapped in the pipline loads x,y
                             steps_per_epoch=train_steps,
                             validation_data=pipeline(val_gen, num_classes=num_classes, batch_size=args.batch_size) if val_gen else None,
                             validation_steps=val_steps,
                             epochs=args.epochs,
                             class_weight=class_weights,
                             verbose=1,
                             shuffle=True,
                             callbacks=callbacks,
                             use_multiprocessing=True)

if __name__ == "__main__":
    main(get_args())
