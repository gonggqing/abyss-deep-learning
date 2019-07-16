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
from keras.preprocessing.image import ImageDataGenerator
#import keras.callbacks
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, Callback, TerminateOnNaN
import tensorflow as tf
import keras.optimizers

from abyss_deep_learning.datasets.coco import ImageClassificationDataset
from abyss_deep_learning.datasets.translators import  AbyssCaptionTranslator, CaptionMapTranslator, AnnotationTranslator, CategoryTranslator
from abyss_deep_learning.keras.classification import caption_map_gen, onehot_gen
from abyss_deep_learning.keras.models import ImageClassifier, loadImageClassifierByDict
from abyss_deep_learning.keras.utils import lambda_gen, batching_gen, gen_dump_data, head_gen

from callbacks import SaveModelCallback, PrecisionRecallF1Callback, TrainValTensorBoard, TrainsCallback
from utils import to_multihot, multihot_gen, compute_class_weights, create_augmentation_configuration
from translators import MultipleTranslators, HotTranslator
from abyss_deep_learning.keras.tensorboard import ImprovedTensorBoard, produce_embeddings_tsv
from abyss_deep_learning.keras.classification import augmentation_gen
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
    parser.add_argument("--augmentation-configuration", type=str, help="The augmentation configuration as args to examples/classification/utils/create_augmentation_configuration  . E.g. {\"some_of\": None, \"bright\": (0.75,1.1)} to use brightness augmentation")
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
    parser.add_argument('--optimizer', type=str, default="adam", help="The optimizer to use. Options are {adam,nadam,sgd,rmsprop,adagrad,adadelta,adamax}")
    parser.add_argument('--optimizer-args', type=str, default="{}", help="The arguments to configure the optimizer. LR is added from --lr argument. For example to add  momentum to SGD, optimizer_args could be {'momentum': 0.9}")
    parser.add_argument("--l12-regularisation", type=str, default="None,None",
                        help="Whether to add l1 l2 regularisation to the convolutional layers of the model. Format (l1,l2), if absent leave as None. For example (None,0.01) to just add l2 regularisation ")
    parser.add_argument("--resume-from-ckpt", type=str,
                        help="Resume the model from the given .h5 checkpoint, as saved by the ImageClassifier.save function. This loads in all weights and parameters from the previous training.")
    parser.add_argument("--weights", type=str, default='imagenet',
                        help="Path to the weights to load into this model. Re-initalises all the checkpoints etc. Default is 'imagenet' which loads the 'imagenet' trained weights")
    parser.add_argument("--gpu-fraction", type=float, default=0.8,
                        help="Limit the amount of GPU usage tensorflow uses. Defaults to 0.8")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers to use")
    parser.add_argument("--gpus", type=int, default=1, help="The number of GPUs to use. To select a specific GPU use CUDA_VISIBLE_DEVICES environment variable, e.g. CUDA_VISIBLE_DEVICES=0 python3 train_cctv_classifier.py ...")
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
    os.makedirs(log_dir, exist_ok=True)


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


    # -----------------------------------
    # Create Augmentation Configuration
    # -----------------------------------
    if args.augmentation_configuration:
        aug_config = ast.literal_eval(args.augmentation_configuration)
    else:
        aug_config = {
            "some_of":None,  # Do all (None=do all, 1=do one augmentation)
            "flip_lr":True,  # Flip 50% of the time
            "flip_ud":True,  # Flip 50% of the time
            "gblur":None,  # No Gaussian Blur
            "avgblur":None,  # No Average Blur
            "gnoise":(0,0.05*255),  # Add a bit of Gaussian noise
            "scale":(0.8, 1.2),  # Don't scale
            "rotate":(-22.5, 22.5),  # Don't rotate
            "bright":(0.75,1.25),  # Darken/Brighten (as ratio)
            "colour_shift":(0.9,1.1)  # Colour shift (as ratio)
        }
    augmentation_cfg = create_augmentation_configuration(**aug_config)

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

    def pipeline(gen, num_classes, batch_size, do_data_aug=False):
        """
        A sequence of generators that perform operations on the data
        Args:
            gen: the base generator (e.g. from dataset.generator())
            num_classes: (int) the number of classes, to create a multihot vector
            batch_size: (int) the batch size, for the batching generator

        Returns:

        """

        return (batching_gen(
                    augmentation_gen(
                        lambda_gen(
                            multihot_gen(
                                lambda_gen(gen, func=preprocess), num_classes=num_classes),
                            func=enforce_one_vs_all),
                        aug_config=augmentation_cfg, enable=do_data_aug),
                    batch_size=batch_size))


       # limit the process GPU usage. Without this, can get CUDNN_STATUS_INTERNAL_ERROR
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
    set_session(tf.Session(config=config))

    # -------------------------
    # Initialise Optimizer
    # -------------------------
    optimizer_args = ast.literal_eval(args.optimizer_args)
    optimizer_args['lr'] = args.lr  # The init_lr argument to ImageClassifier is what actually sets the optimizer learning rate


    # -------------------------
    # Create Classifier Model
    # -------------------------
    if args.resume_from_ckpt:
        classifier = ImageClassifier.load(args.resume_from_ckpt)
        classifier.set_lr(args.lr) # update learning rate
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
            optimizer=args.optimizer,
            optimizer_args=optimizer_args,
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
    # val_steps = 100
    # train_steps = 100
    # ------------------------------
    # Configure the validation data
    # ------------------------------
    # Set the validation pipeline - shouldn't have image augmentation
    val_pipeline = pipeline(val_gen, num_classes=num_classes, batch_size=args.batch_size, do_data_aug=False) if val_gen else None
    # If the
    if args.cache_val and val_gen:
        print("CACHING VAL")
        def cache_pipeline(gen, num_classes):
            return (lambda_gen(multihot_gen(lambda_gen(gen, func=preprocess), num_classes=num_classes), func=enforce_one_vs_all))
        val_data = gen_dump_data(gen=cache_pipeline(val_gen, num_classes), num_images=val_steps, verbose=True)
    else:
        val_data = val_pipeline

    # ------------------
    # Callbacks
    # ------------------
    do_embeddings = True 
    if do_embeddings:
        assert(not args.gpus > 1, 'Due to a bug, if calcualting embeddings, only 1 gpu can be used')
        embeddings_data = gen_dump_data(gen=pipeline(val_gen, num_classes=149, batch_size=1), num_images=1000)  #dump some images for the embeddingsi
        print(embeddings_data[1].squeeze())
        produce_embeddings_tsv(os.path.join(log_dir, 'metadata.tsv'), headers=[str(i) for i in np.arange(0,149)], labels=embeddings_data[1].squeeze())

    callbacks = [SaveModelCallback(classifier.save, model_dir, save_interval=args.save_model_interval),  # A callback to save the model
                ImprovedTensorBoard(log_dir=log_dir, batch_size=args.batch_size, write_graph=True, embeddings_freq=60, embeddings_metadata=os.path.join(args.scratch_dir,'metadata.tsv'), embeddings_data=embeddings_data[0].squeeze(), embeddings_layer_names=['global_average_pooling2d_1'], num_classes=num_classes, val_generator=pipeline(val_gen, num_classes=num_classes, batch_size=args.batch_size) if (val_gen and not args.cache_val) else None, val_steps=val_steps),
                #ImprovedTensorBoard(log_dir=log_dir, histogram_freq=3, batch_size=args.batch_size, write_graph=True, write_grads=True, num_classes=num_classes, pr_curve=False, val_generator=pipeline(val_gen, num_classes=num_classes, batch_size=1) if (val_gen and not args.cache_val) else None, val_steps=val_steps),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=10, min_lr=1e-8),
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
        args.class_weights = args.class_weights.split(",")
        args.class_weights = { i : float(args.class_weights[i]) for i in range(0, len(args.class_weights) ) } # convert list to class_weight dict.
    print("Using class weights: ", args.class_weights)

    classifier.fit_generator(generator=pipeline(train_gen, num_classes=num_classes, batch_size=args.batch_size, do_data_aug=True),
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
