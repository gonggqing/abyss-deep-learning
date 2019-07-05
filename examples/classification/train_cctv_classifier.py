import os
import argparse
import json
import warnings
import cv2
import ast
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
from abyss_deep_learning.keras.models import ImageClassifier, loadImageClassifierByDict
from abyss_deep_learning.keras.utils import lambda_gen, batching_gen, gen_dump_data

from callbacks import SaveModelCallback, PrecisionRecallF1Callback, TrainValTensorBoard, TrainsCallback
from utils import to_multihot, multihot_gen, compute_class_weights
from translators import MultipleTranslators, HotTranslator
from abyss_deep_learning.keras.tensorboard import ImprovedTensorBoard

import trains
from trains import Task

NN_DTYPE = np.float32


def enforce_one_vs_all(image, labels):
    """
    A Function to be used with the lambda_gen in the pipeline. If a background label and another label is detected, it removes the background label
    Args:
        image: passedthrough
        labels: a multihot vector

    Returns:
        image, labels

    """
    if np.sum(labels[1:]) >= 1.0:
        labels[0] = 0
    return image,labels



def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
    This script is designed to get eric going with CCTV training.
    """)
    parser.add_argument("coco_path", type=str, help="Path to the coco dataset")
    parser.add_argument("--val-coco-path", type=str, help="Path to the validation coco dataset")
    parser.add_argument("--cache-val", action="store_true",
                        help="Whether to cache the validation dataset. Results in major speedup after initial data load.")
    parser.add_argument("--scratch-dir", type=str, help="The scratch directory for this experiment.")
    parser.add_argument("--category-map", type=str, help="Path to the category map")
    parser.add_argument("--image-shape", type=str, default="240,320,3", help="Image shape")
    parser.add_argument("--batch-size", type=int, default=2, help="Image shape")
    parser.add_argument("--epochs", type=int, default=2, help="Image shape")
    parser.add_argument("--lr", type=float, default=1e-4, help="Sets the initial learning rate of the optimiser.")
    parser.add_argument("--save-model-interval", type=int, default=1, help="How often to save the model")
    parser.add_argument("--load-params-json", type=str,
                        help="Use the params.json file to initialise the model. Using this ignores the command line arguments.")
    parser.add_argument("--backbone", type=str, default="xception",
                        help="The backbone to use, options are \{xception\}")
    parser.add_argument("--output-activation", type=str, default="softmax",
                        help="The output activation to use. Options are \{softmax,sigmoid\}")
    parser.add_argument("--pooling", type=str, default="avg",
                        help="The pooling to use after the convolution layers. Options are \{avg,max\}")
    parser.add_argument("--loss", type=str, default="categorical_crossentropy",
                        help="The loss function to use. Options are \{categorical_crossentropy,binary_crossentropy\}")
    parser.add_argument("--l12-regularisation", type=str, default="None,None",
                        help="Whether to add l1 l2 regularisation to the convolutional layers of the model. Format (l1,l2), if absent leave as None. For example (None,0.01) to just add l2 regularisation ")
    parser.add_argument("--resume-from-ckpt", type=str,
                        help="Resume the model from the given .h5 checkpoint, as saved by the ImageClassifier.save function. This loads in all weights and parameters from the previous training.")
    parser.add_argument("--weights", type=str, default='imagenet',
                        help="Path to the weights to load into this model. Re-initalises all the checkpoints etc. Default is 'imagenet' which loads the 'imagenet' trained weights")
    parser.add_argument("--gpu-fraction", type=float, default=0.8,
                        help="Limit the amount of GPU usage tensorflow uses. Defaults to 0.8")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers to use")
    parser.add_argument("--gpus", type=int, default=1, help="The number of GPUs to use")
    parser.add_argument("--trains-project", type=str, help="The project to use for the TRAINS server")
    parser.add_argument("--no-trains", action="store_false", help="Turn off experiment tracking with trains")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="The patience in number of epochs for network loss to improved before early-stopping of training.")
    parser.add_argument("--class-weights", type=str, default=None,
                    help="Class weights as a list, (e.g., 1,10,12 for three weighted classes), or for optimal class-weights, set to 1.")
    args = parser.parse_args()
    return args


def main(args):
    # -------------------
    # Check Parameters
    # -------------------
    if not args.scratch_dir:
        raise ValueError("Scratch directory needs to be given")

    l12_reg = ast.literal_eval(args.l12_regularisation)

    # --------------------------------------
    # Set up logging and scratch directories
    # --------------------------------------
    os.makedirs(args.scratch_dir, exist_ok=True)
    model_dir = os.path.join(args.scratch_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join(args.scratch_dir, 'logs')


    if not args.no_trains and not args.trains_project:
        raise ValueError("If experiment tracking is on, the --trains-project needs to be given")
    elif not args.no_trains:
        warnings.warn("Experiment tracking is turned off!")
        task = None
    else:
        experiment_name = os.path.basename(args.scratch_dir)  # TODO evaluate this
        task = Task.init(args.trains_project, experiment_name)
        with open(os.path.join(args.scratch_dir, 'task_id.txt'), 'w') as task_id_file:
            task_id_file.write(task.id)
        print("TRAINS - Project %s - Experiment %s" %(args.trains_project, experiment_name))


    # ------------------------------
    # Read + initalise category map
    # ------------------------------s
    raw_cat_map = json.load(open(args.category_map, 'r'))  # Load the caption map - caption_map should live on place on servers
    cat_map = {}
    for k,v in raw_cat_map.items():
        cat_map[int(k)] = v
    cat_translator = CategoryTranslator(mapping=cat_map)
    num_classes = len(set(cat_map.values()))  # Get num classes from caption map

    image_shape = [int(x) for x in args.image_shape.split(',')]

    # ------------------------
    # Initialise datasets
    # ------------------------
    train_dataset = ImageClassificationDataset(args.coco_path, translator=cat_translator)
    train_gen = train_dataset.generator(endless=True, shuffle_ids=True)
    if args.val_coco_path:
        val_dataset = ImageClassificationDataset(args.val_coco_path, translator=cat_translator)
        val_gen = val_dataset.generator(endless=True, shuffle_ids=True)
    else:
        val_gen = None
        val_dataset = None

    # -------------------------
    # Create data pipeline
    # -------------------------

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
        return (batching_gen(lambda_gen(multihot_gen(lambda_gen(gen, func=preprocess), num_classes=num_classes), func=enforce_one_vs_all),
                             batch_size=batch_size))

    # limit the process GPU usage. Without this, can get CUDNN_STATUS_INTERNAL_ERROR
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
    set_session(tf.Session(config=config))

    # -------------------------
    # Create Classifier Model
    # -------------------------
    if args.resume_from_ckpt:
        classifier = ImageClassifier.load(args.resume_from_ckpt)
    elif args.load_params_json:
        classifier = loadImageClassifierByDict(args.load_params_json)
    else:
        classifier = ImageClassifier(
            backbone=args.backbone,
            output_activation=args.output_activation,
            pooling=args.pooling,
            classes=num_classes,
            input_shape=tuple(image_shape),
            init_weights=args.weights,
            init_epoch=0,
            init_lr=args.lr,
            trainable=True,
            loss=args.loss,
            metrics=['accuracy'],
            gpus=args.gpus,
            l12_reg=l12_reg
        )
    classifier.dump_args(os.path.join(args.scratch_dir, 'params.json'))

    # --------------------------------------
    # Calculate number of steps (batches)
    # --------------------------------------

    train_steps = np.floor(len(train_dataset) / args.batch_size)
    val_steps = int(np.floor(len(val_dataset) / args.batch_size)) if val_dataset is not None else None
    # ------------------------------
    # Configure the validation data
    # ------------------------------

    # Set the validation pipeline - shouldn't have image augmentation
    val_pipeline = pipeline(val_gen, num_classes=num_classes, batch_size=args.batch_size) if val_gen else None
    # If the
    if args.cache_val and val_gen:
        val_data = gen_dump_data(gen=val_pipeline, num_images=len(val_dataset))
    else:
        val_data = val_pipeline

    # ------------------
    # Callbacks
    # ------------------
    callbacks = [SaveModelCallback(classifier.save, model_dir, save_interval=args.save_model_interval),  # A callback to save the model
                ImprovedTensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=args.batch_size, write_graph=True,
                                    write_grads=True, num_classes=num_classes, pr_curve=True, val_generator=pipeline(val_gen, num_classes=num_classes, batch_size=1) if (val_gen and not args.cache_val) else None, val_steps=val_steps),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=1e-8),
                TerminateOnNaN()
                ]
    if args.early_stopping_patience:
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=args.early_stopping_patience, verbose=1, mode='auto', baseline=None, restore_best_weights=True))
    if task:
        callbacks.append(TrainsCallback(logger=task.get_logger()))

    # ----------------------------
    # Train
    # ----------------------------
    if args.class_weights == 1:
        args.class_weights = train_dataset.class_weights
    elif args.class_weights:
        args.class_weights = { i : args.class_weights[i] for i in range(0, len(args.class_weights) ) } # convert list to class_weight dict.
    print("Using class weights: ", args.class_weights)

    classifier.fit_generator(generator=pipeline(train_gen, num_classes=num_classes, batch_size=args.batch_size),
                             steps_per_epoch=train_steps,
                             validation_data=val_data,
                             validation_steps=val_steps,
                             epochs=args.epochs,
                             verbose=1,
                             shuffle=True,
                             callbacks=callbacks,
                             use_multiprocessing=True,
                             workers=args.workers,
                             class_weight=args.class_weights)


if __name__ == "__main__":
    main(get_args())
