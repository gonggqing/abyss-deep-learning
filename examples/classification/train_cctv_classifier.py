import os
import argparse
import json
import warnings
import ast
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, Callback, TerminateOnNaN, LearningRateScheduler
import tensorflow as tf
import keras.optimizers
import math
import sys

from abyss_deep_learning.datasets.coco import ImageClassificationDataset
from abyss_deep_learning.datasets.translators import CategoryTranslator
from abyss_deep_learning.keras.classification import caption_map_gen, onehot_gen
from abyss_deep_learning.keras import classification
from abyss_deep_learning.keras.utils import lambda_gen, batching_gen, gen_dump_data, head_gen

from callbacks import SaveModelCallback, PrecisionRecallF1Callback, TrainValTensorBoard, TrainsCallback, create_lr_schedule_callback
from utils import to_multihot, multihot_gen, compute_class_weights, create_augmentation_configuration
from translators import MultipleTranslators, HotTranslator
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
    This script trains CCTV classification networks.
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
    parser.add_argument("--training-steps", type=int, help="The number of training steps. If not specified, defaults to length of the dataset")
    parser.add_argument("--validation-steps", type=int, help="The number of validation steps. If not specified, defaults to length of the dataset")
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
    parser.add_argument('--optimizer-args', type=str, default="{}", help="The arguments to configure the optimizer. LR is added from --lr argument. For example to add  momentum to SGD, optimizer_args could be \"{'momentum':0.9}\"")
    parser.add_argument("--l12-regularisation", type=str, default="None,None",
                        help="Whether to add l1 l2 regularisation to the convolutional layers of the model. Format (l1,l2), if absent leave as None. For example (None,0.01) to just add l2 regularisation ")
    parser.add_argument("--resume-from-ckpt", type=str,
                        help="Resume the model from the given .h5 checkpoint, as saved by the classification.Task.save function. This loads in all weights and parameters from the previous training.")
    parser.add_argument("--weights", type=str, default='imagenet',
                        help="Path to the weights to load into this model. Re-initalises all the checkpoints etc. Default is 'imagenet' which loads the 'imagenet' trained weights")
    parser.add_argument("--gpu-fraction", type=float, default=0.8,
                        help="Limit the amount of GPU usage tensorflow uses. Defaults to 0.8")
    parser.add_argument("--gpu-allow-growth", action="store_true", help="set allow_growth flag in gpu options")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers to use")
    parser.add_argument("--gpus", type=int, default=1, help="The number of GPUs to use. To select a specific GPU use CUDA_VISIBLE_DEVICES environment variable, e.g. CUDA_VISIBLE_DEVICES=0 python3 train_cctv_classifier.py ...")
    parser.add_argument("--trains-project", type=str, help="The project to use for the TRAINS server")
    parser.add_argument("--trains-experiment", type=str, help="The experiment name to use for the TRAINS experiment tracking. If left blank, defaults to basename of scratch directory")
    parser.add_argument("--no-trains", action="store_false", help="Turn off experiment tracking with trains")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="The patience in number of epochs for network loss to improved before early-stopping of training.")
    parser.add_argument("--class-weights", type=str, default=None,
                    help="Class weights as a list, (e.g., 1,10,12 for three weighted classes), or for optimal class-weights, set to 1.")
    parser.add_argument("--embeddings", action="store_true", help="Whether to do the embeddings")
    parser.add_argument("--embeddings-freq", type=int, default=1, help="How often to calculate the embeddings")
    parser.add_argument("--histogram-freq", type=int, default=1, help="The frequency at which to calculate histograms. Set to 0 to turn off. Will be set to 0 if not using --cache-val.")
    parser.add_argument("--pr-curves", action="store_true", help="Whether to calculate pr curves. Will be set to false if not using --cache-val.")
    parser.add_argument("--tfpn", action="store_true", help="Whether to calculate TFPN. Will be set to false if not using --cache-val.")
    parser.add_argument("--lr-schedule", type=str, help="The LR schedule to use, options are {step,exp,cyclic}. If left blank, no LR scheduling will be used. For example --lr-schedule cyclic")
    parser.add_argument("--lr-schedule-params", type=str, help="The parameters initialising the learning rate schedule. This is a dictionary that is used to initialise the LearningRateScheduler you are using. If left blank, default parameters will be used. For example for '--lr-schedule cyclic' do {'max_lr':0.006,'step_size':2000.}")
    parser.add_argument("--only-train-head", action="store_true", help="Freezes the entire graph, but not the head, so only it is trained.")
    args = parser.parse_args()
    return args


def main(args):
    # -------------------
    # Checks, for anything really bad that happens due to bugs
    # -------------------
    assert(args.tfpn is False, "ERROR: Currently, using TFPN makes networks not train. It is most likely due to do "
                               "weights not updating, or some such. This needs fixing, eventually.")
    assert (args.gpus == 1, "ERROR: Currently, using multiple GPUs will crash training when it comes time to compute the "
                            "embeddings etc. This needs to be fixed.")


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


    # ---------------------------------------
    # Check datasets exist
    # ---------------------------------------
    if not os.path.isfile(args.coco_path):
        raise ValueError("Training dataset %s does not exist" %args.coco_path)
    if args.val_coco_path and not os.path.isfile(args.val_coco_path):
        raise ValueError("Validation dataset %s does not exist" % args.val_coco_path)



    if not args.no_trains and not args.trains_project:
        raise ValueError("If experiment tracking is on, the --trains-project needs to be given")
    elif not args.no_trains:
        warnings.warn("Experiment tracking is turned off!")
        task = None
    else:
        if args.trains_experiment:
            experiment_name = args.trains_experiment
        else:
            experiment_name = os.path.basename(args.scratch_dir)
        task = Task.init(args.trains_project, experiment_name)
        with open(os.path.join(args.scratch_dir, 'task_id.txt'), 'w') as task_id_file:
            task_id_file.write(task.id)
        print("TRAINS - Project %s - Experiment %s" %(args.trains_project, experiment_name))


    # ------------------------------
    # Read + initalise category map
    # ------------------------------
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
            "gnoise":(0,0.05),  # Add a bit of Gaussian noise
            "scale":(0.8, 1.2),  # Don't scale
            "rotate":(-22.5, 22.5),  # Don't rotate
            "bright":(0.9,1.1),  # Darken/Brighten (as ratio)
            "colour_shift":(0.95,1.05),  # Colour shift (as ratio)
            "cval":-1
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

    def pipeline(gen, num_classes, batch_size, do_data_aug=False, ):
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

    def cache_pipeline(gen, num_classes):
        return (lambda_gen(multihot_gen(lambda_gen(gen, func=preprocess), num_classes=num_classes),
                           func=enforce_one_vs_all))

       # limit the process GPU usage. Without this, can get CUDNN_STATUS_INTERNAL_ERROR
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
    config.gpu_options.allow_growth = args.gpu_allow_growth;
    set_session(tf.Session(config=config))

    # -------------------------
    # Initialise Optimizer
    # -------------------------
    optimizer_args = ast.literal_eval(args.optimizer_args)
    optimizer_args['lr'] = args.lr  # The init_lr argument to classification.Task is what actually sets the optimizer learning rate


    # -------------------------
    # Create Classifier Model
    # -------------------------
    if args.resume_from_ckpt:
        classifier = classification.Task.load(args.resume_from_ckpt)
        classifier.set_lr(args.lr) # update learning rate
    elif args.load_params_json:
        classifier = classification.Task.from_json( args.load_params_json )
    else:
        classifier = classification.Task(
            backbone=args.backbone,
            output_activation=args.output_activation,
            pooling=args.pooling,
            classes=num_classes,
            input_shape=tuple(image_shape),
            init_weights=args.weights,
            init_epoch=0,
            init_lr=args.lr,
            trainable=False if args.only_train_head else True, # if only_train_head, then all layers are frozen. Head layer is set to trainable later
            optimizer=args.optimizer,
            optimizer_args=optimizer_args,
            loss=args.loss,
            metrics=['accuracy'],
            gpus=args.gpus,
            l12_reg=l12_reg
        )
        if args.only_train_head:
            classifier._maybe_create_model()
            classifier.set_trainable({'logits': True})
    classifier.dump_args(os.path.join(args.scratch_dir, 'params.json'))

    # --------------------------------------
    # Calculate number of steps (batches)
    # --------------------------------------
    if args.training_steps:
        train_steps = args.training_steps
    else:
        train_steps = np.floor(len(train_dataset) / args.batch_size)
    if args.validation_steps:
        val_steps = args.validation_steps
    else:
        val_steps = int(np.floor(len(val_dataset) / args.batch_size)) if val_dataset is not None else None

    # ------------------------------
    # Configure the validation data
    # ------------------------------
    # Set the validation pipeline - shouldn't have image augmentation
    val_pipeline = pipeline(val_gen, num_classes=num_classes, batch_size=args.batch_size, do_data_aug=False) if val_gen else None
    # If the
    if args.cache_val and val_gen:
        print("CACHING VAL", file = sys.stderr )
        val_data = gen_dump_data(gen=cache_pipeline(val_gen, num_classes), num_images=val_steps, verbose=True)
        tfpn = args.tfpn
        histogram_freq = args.histogram_freq
        pr_curves = args.pr_curves
    else:
        val_data = val_pipeline
        if args.tfpn:
            warnings.warn("TFPN doesn't work properly unless val is cached. Used --cache-val to cache val. Setting to False")
        if args.histogram_freq > 0:
            warnings.warn("Histograms don't work unless val is cached. Used --cache-val to cache val. Setting to 0")
        if args.pr_curves:
            warnings.warn("PR Curves don't work properly unless val is cached. Used --cache-val to cache val. Setting to False")
        tfpn = False
        histogram_freq = 0
        pr_curves = False


    # ------------------
    # Callbacks
    # ------------------

    if args.embeddings:
        print("CACHING EMBEDDING", file = sys.stderr )
        from abyss_deep_learning.keras.tensorboard import produce_embeddings_tsv
        assert args.gpus == 1, 'Due to a bug, if calculating embeddings, only 1 gpu can be used'
        embeddings_data = gen_dump_data(gen=cache_pipeline(val_gen, num_classes),
                                        num_images=int(np.floor(len(val_dataset) / args.batch_size)), verbose=True)
        produce_embeddings_tsv(os.path.join(log_dir, 'metadata.tsv'),
                               headers= [ str(i) for i in range(num_classes) ],
                               labels= embeddings_data[1] )
        embeddings_freq = args.embeddings_freq
    else:
        embeddings_data = [None, None]
        embeddings_freq = 0

    try:
        from abyss_deep_learning.keras.tensorboard import ImprovedTensorBoard
        improved_tensorboard = ImprovedTensorBoard(log_dir=log_dir, batch_size=args.batch_size, write_graph=True, embeddings_freq=embeddings_freq, embeddings_metadata=os.path.join(log_dir,'metadata.tsv'), embeddings_data=embeddings_data[0], embeddings_layer_names=['global_average_pooling2d_1'], num_classes=num_classes, val_generator=pipeline(val_gen, num_classes=num_classes, batch_size=args.batch_size) if (val_gen and not args.cache_val) else None, val_steps=val_steps, tfpn=tfpn, pr_curve=pr_curves, histogram_freq=histogram_freq)
    except:
        improved_tensorboard = None
        warnings.warn( "failed to import tensorboard; running without it" )
    if improved_tensorboard is None:
        callbacks = [ SaveModelCallback( classifier.save, model_dir, save_interval = args.save_model_interval )  # A callback to save the model
                    , ReduceLROnPlateau( monitor = 'val_loss', factor = 0.5, patience = 25, min_lr = 1e-8 )
                    , TerminateOnNaN() ]
    else:
        callbacks = [SaveModelCallback(classifier.save, model_dir, save_interval=args.save_model_interval),  # A callback to save the model
                    improved_tensorboard,
                    #ImprovedTensorBoard(log_dir=log_dir, histogram_freq=3, batch_size=args.batch_size, write_graph=True, write_grads=True, num_classes=num_classes, pr_curve=False, val_generator=pipeline(val_gen, num_classes=num_classes, batch_size=1) if (val_gen and not args.cache_val) else None, val_steps=val_steps),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=25, min_lr=1e-8),
                    TerminateOnNaN()
                    ]
    if args.early_stopping_patience:
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=args.early_stopping_patience, verbose=1, mode='auto', baseline=None, restore_best_weights=True))
    if task:
        callbacks.append(TrainsCallback(logger=task.get_logger()))

    if args.lr_schedule:
        lr_schedule_params = ast.literal_eval(args.lr_schedule_params) if args.lr_schedule_params else None  # Load the lr schedule params
        lr_schedule_callback = create_lr_schedule_callback(args.lr_schedule, args.lr, lr_schedule_params)
        if lr_schedule_callback:
            callbacks.append(lr_schedule_callback)

    # ----------------------------
    # Train
    # ----------------------------
    if args.class_weights == 1 or args.class_weights == "1":
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
