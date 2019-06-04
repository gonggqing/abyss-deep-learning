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
from abyss_deep_learning.datasets.translators import  AbyssCaptionTranslator, CaptionMapTranslator, AnnotationTranslator, CategoryTranslator
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
    parser.add_argument("--category-map", type=str, help="Path to the caption map")
    parser.add_argument("--image-shape", type=str, default="320,240,3", help="Image shape")
    parser.add_argument("--batch-size", type=int, default=2, help="Image shape")
    parser.add_argument("--epochs", type=int, default=2, help="Image shape")
    parser.add_argument("--save-model-interval", type=int, default=1, help="How often to save the mdoel interval")
    args = parser.parse_args()
    return args

def main(args):
    # Set up logging and scratch directories
    os.makedirs(args.scratch_dir, exist_ok=True)
    model_dir = os.path.join(args.scratch_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(args.scratch_dir, 'logs')

    # do the caption translations and any preprocessing set-up
    raw_cat_map = json.load(open(args.category_map, 'r'))  # Load the caption map - caption_map should live on place on servers
    cat_map = {}
    for k,v in raw_cat_map.items():
        cat_map[int(k)] = v
    cat_translator = CategoryTranslator(mapping=cat_map)
    num_classes = len(set(cat_map.values()))  # Get num classes from caption map
    hot_translator = HotTranslator(num_classes)  # Hot translator encodes as a multi-hot vector
    image_shape = [int(x) for x in args.image_shape.split(',')]

    train_dataset = ImageClassificationDataset(args.coco_path, translator=cat_translator)
    train_gen = train_dataset.generator(endless=True, shuffle_ids=True)
    if args.val_coco_path:
        val_dataset = ImageClassificationDataset(args.val_coco_path, translator=cat_translator)
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
        metrics=['accuracy']
    )
    classifier.dump_args(os.path.join(args.scratch_dir, 'params.json'))

    ## callbacks
    callbacks = [SaveModelCallback(classifier.save, model_dir, save_interval=10),  # A callback to save the model
                 ImprovedTensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=args.batch_size, write_graph=True,
                                     write_grads=True, num_classes=num_classes, pr_curve=True, val_generator=pipeline(val_gen, num_classes=num_classes, batch_size=1) if val_gen else None, val_steps=len(val_dataset), tfpn=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=5, min_lr=1e-4),
                 # EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=6, verbose=1, mode='auto',
                 #                               baseline=None, restore_best_weights=True),
                 TerminateOnNaN()
                 ]

    # TEST GENERATOR

    for i, (inp, tgt) in enumerate(pipeline(train_gen, num_classes=num_classes, batch_size=args.batch_size)):
        print(tgt)
        if i >= 5:
            break

    train_steps = np.floor(len(train_dataset) / args.batch_size)
    val_steps = np.floor(len(val_dataset) / args.batch_size) if val_dataset is not None else None
    # class_weights = compute_class_weights(train_dataset)
    classifier.fit_generator(generator=pipeline(train_gen, num_classes=num_classes, batch_size=args.batch_size),  # The generator wrapped in the pipline loads x,y
                             steps_per_epoch=train_steps,
                             validation_data=pipeline(val_gen, num_classes=num_classes, batch_size=args.batch_size) if val_gen else None,
                             validation_steps=val_steps,
                             epochs=args.epochs,
                             verbose=1,
                             shuffle=True,
                             callbacks=callbacks,
                             use_multiprocessing=True)

if __name__ == "__main__":
    main(get_args())
