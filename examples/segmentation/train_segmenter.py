#!/usr/bin/env python3.6
"""Train a segmentation network on COCO JSON datasets.
Images are loaded as RGB uint8, optionally augmented, then converted to
tensorflow standard format (float32 in range [-1:1]) before being fed to
the network.

Attributes:
    DEFAULT_CONFIG (dict): Default training/validation configuration.
"""

# import os
import json
import logging
import argparse

import abyss_deep_learning.imgaug as abyss_imgaug
import abyss_deep_learning.keras.zoo.segmentation.deeplab_v3_plus.util as deeplab_util
from abyss_deep_learning.keras.utils import lambda_gen
from abyss_deep_learning.datasets.coco import ImageSemanticSegmentationDataset

import tensorflow as tf
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

DEFAULT_CONFIG = {
    'batch_size': 4,
    'epochs': 5,
    'learning_rate': 1e-4,
    'new_heads': True,
}


def get_args(cmd_line=None):
    """Construct arguments from command line.

    Returns:
        argparse.namespace: The arguments for this application

    Args:
        cmd_line (str, optional): Command line string to use instead of
            reading from sys.argv.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
        Train a segmentation network.
        Usage:

        # Train a model, initializing new heads (defined in config.json:/model).
            train_segmenter.py coco.train.json config.json --dataset-val coco.val.json

        # Continue training a model (given the same config.json, and model.h5)
            train_segmenter.py coco.train_set_2.json config.json \\
            --dataset-val coco.val.json \\
            --load model.h5""")
    parser.add_argument(
        'config',
        help="Specifies the path to the config JSON for the model.")
    parser.add_argument('dataset_train', type=str,
                        help="Path to the coco dataset")
    parser.add_argument("--dataset-val", type=str,
                        help="Path to the validation coco dataset")
    parser.add_argument(
        "--load", type=str,
        help=(
            "Path to model weights to load."
            "Also accepts 'pascal_voc' (21 classes) or 'cityscapes' (19)."))
    parser.add_argument(
        "--output-config",
        help="TODO (update): Outputs the default config and exits.",
        action='store_true')
    parser.add_argument(
        '--verbose', '-v',
        action='store_const', const=logging.INFO,
        dest='loglevel', help="verbose output to stderr")
    args = parser.parse_args(cmd_line)
    return args


class AnnotationClassMapper:
    """Maps category to classes given an explicit mapping.

    Attributes:
        category_to_class_map (dict): Mapping from category_id to class_id
    """

    def __init__(self, category_to_class_map):
        self.category_to_class_map = category_to_class_map

    def filter(self, annotation):
        return True

    def translate(self, annotation):
        output = dict(annotation)
        output['category_id'] = self.category_to_class_map[annotation['category_id']]
        return output


def get_class_mapping(dataset, categories_config):
    """Return a class map for the given dataset and categories
        json mapping.

    Args:
        dataset (adl.datasets.coco.CocoInterface): The dataset
            to perform the mapping on.
        categories_config (dict): The categories configuration
            dict mapping `coco name` to `class id`.

    Returns:
        AnnotationClassMapper: Dataset translator to provide to
            the dataset.
    """
    category_map = {  # Maps category name to category_id
        category['name']: category['id']
        for category in dataset.coco.cats.values()
    }
    category_to_class_map = {
        category_map[pair['name']]: pair['id']
        for pair in categories_config
    }
    logging.info('category_to_class_map')
    logging.info(category_to_class_map)
    return AnnotationClassMapper(category_to_class_map)


def hack_dataset(generator):
    """A hack to get generators to work with tf.keras datasets.

    Args:
        generator: A generator yielding (input, targets) tuples.

    Returns:
        tf.keras.datasets.Dataset: A tf.keras compatible dataset.
    """
    def wrap_callable():
        return generator
    return wrap_callable


def setup_dataset(coco_path, model_config, process_config, categories_config):
    """Set up a COCO dataset given the model, process (train/val),
        and categories configuration.

    Args:
        coco_path (str): Description
        model_config (dict): Model configuration dict.
        process_config (dict): Training or validation configuration dict.
        categories_config (dict): Categories configuration dict.

    Returns:
        adl.datasets.coco.CocoInterface: The adl.datasets COCO dataset.
        tf.data.Dataset: The tf.keras compatible dataset.
    """
    def preprocess_inputs(image, targets):
        nonlocal augmentation
        image2, targets2 = augmentation(
            image=image,
            segmentation_maps=SegmentationMapsOnImage(targets, image.shape))
        targets2 = targets2.get_arr()
        return image2.astype('float32') / 127.5 - 1, targets2

    augmentation = abyss_imgaug.sequential_from_dicts(
        process_config['augmentation'])
    logging.info("Using augmentation:")
    logging.info([i for i in augmentation])
    output_shapes = (
        model_config['input_shape'],
        model_config['input_shape'][:-1] + (model_config['classes'],))
    dataset = ImageSemanticSegmentationDataset(
        coco_path, num_classes=model_config['classes'])
    translator = get_class_mapping(dataset, categories_config)
    dataset.translator = translator

    generator = dataset.generator(endless=True)
    generator = lambda_gen(generator, func=preprocess_inputs)
    generator = tf.data.Dataset.from_generator(
        hack_dataset(generator),
        output_types=(tf.float32, tf.float32),
        output_shapes=output_shapes)\
        .prefetch(2 * process_config['batch_size'])\
        .batch(process_config['batch_size'])
    return dataset, generator


def load_config(config_path):
    """Load the default configuration, and update it with the config file.

    Args:
        config_path (str): Path to the config.json

    Returns:
        dict: The configuration dict.

    Raises:
        ValueError: Bad model type.
        module: The model_util module associated with the config.
    """
    with open(config_path, 'r') as file:
        user_config = json.load(file)

    # Setup model utilities
    if user_config['model']['type'] == 'deeplab':
        model_util = deeplab_util
    else:
        raise ValueError(f"Unknown model type `{user_config['model']['type']}`")
    del user_config['model']['type']  # TODO: Review if this is good, or should be CLI

    # Update configs
    config = {
        'model': dict(model_util.DEFAULT_CONFIG),
        'training': dict(DEFAULT_CONFIG),
        'validation': dict(DEFAULT_CONFIG),
        'categories': user_config['categories'] or dict()
    }

    config['model'].update(user_config['model'])
    config['training'].update(user_config['training'])
    config['validation'].update(user_config['validation'])
    config['model']['input_shape'] = tuple(user_config['model']['input_shape'])
    del user_config

    logging.info("Updated config:")
    logging.info(config)
    return config, model_util


def main(args):
    """Train a segmentation network given the program args.

    Args:
        args (argparse.Namespace): Program args from get_args()

    Returns:
        int: Error code. 0 if no errors occurred.
    """
    logging.basicConfig(
        format='%(filename)s: %(asctime)s.%(msecs)d: %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.loglevel)
    logging.info('--verbose enabled')
    logging.info(args)

    # Load config
    config, model_util = load_config(args.config)
    if args.output_config:
        print(json.dumps(config))
        return 0

    # Set up dataset
    dataset_train, gen_train = setup_dataset(
        args.dataset_train, config['model'], config['training'], config['categories'])
    validation_steps, gen_val = config['validation']['steps'], None

    if args.dataset_val:
        logging.info("Using validation.")
        dataset_val, gen_val = setup_dataset(
            args.dataset_val, config['model'], config['validation'], config['categories'])
        if validation_steps is None:
            validation_steps = len(
                dataset_val.data_ids) // config['validation']['batch_size']
    steps_per_epoch = len(
        dataset_train.data_ids) // config['training']['batch_size']

    # # Set batch norm policy
    # for layer in model.layers:
    #     if type(layer).__name__.startswith('BatchNormalization'):
    #         layer.trainable = layer.trainable and config['training']['train_batch_norm']

    # Set up training
    if config['training']['new_heads']:
        logging.info("Creating new head.")
        model = model_util.make_model(**config['model'])
    else:
        logging.info("Loading model.")
        model = tf.keras.models.load_model(args.load)
    callbacks = [
        tf.keras.callbacks.TensorBoard(".", write_graph=False),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model_{epoch:03d}_{val_loss:.3f}.h5',
            save_best_only=True, save_weights_only=False, verbose=1)
    ]

    with open("model_def.json", 'w') as file:
        file.write(model.to_json())

    # Start training
    optimizer = tf.keras.optimizers.Adam(lr=config['training']['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    model.fit(
        gen_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=gen_val,
        validation_steps=validation_steps,
        epochs=config['training']['epochs'],
        callbacks=callbacks,
        verbose=1)
    return 0


if __name__ == '__main__':
    exit(main(get_args()))
