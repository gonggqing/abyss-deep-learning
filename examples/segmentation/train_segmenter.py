#!/usr/bin/env python3.6
"""Train a segmentation network on COCO JSON datasets.
"""

# import os
import json
import argparse
import logging

import numpy as np
# import imgaug as ia
import tensorflow as tf
# import matplotlib.pyplot as plt
# from imgaug import augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapOnImage

from abyss_deep_learning.keras.utils import lambda_gen
# from abyss_deep_learning.datasets.translators import CategoryTranslator
from abyss_deep_learning.datasets.coco import ImageSemanticSegmentationDataset
import abyss_deep_learning.keras.zoo.segmentation.deeplab_v3_plus.util as deeplab


DEFAULT_CONFIG = {
    "batch_size": 4,
    "epochs": 5,
    "learning_rate": 1e-4,
    "new_heads": True,
}


def get_args(cmd_line=None):
    """Construct arguments from command line.

    Returns:
        TYPE: Description
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
    Train a segmentation network
    """)
    parser.add_argument("dataset_train", type=str,
                        help="Path to the coco dataset")
    parser.add_argument("--dataset-val", type=str,
                        help="Path to the validation coco dataset")
    parser.add_argument(
        "--weights", type=str,
        help="Path to model weights to load. Also accepts 'pascal_voc' (21 classes) or 'cityscapes' (19).")
    parser.add_argument(
        "--config",
        help="Specifies the path to the config JSON for the model.")
    # parser.add_argument(
    #     "--new-heads",
    #     help="Create new heads.", action='store_true')
    parser.add_argument(
        "--output-config",
        help="Outputs the default config and exits.", action='store_true')
    parser.add_argument(
        '--verbose', '-v',
        action='store_const',
        const=logging.INFO,
        dest='loglevel',
        help="verbose output to stderr",
    )
    args = parser.parse_args(cmd_line)
    return args


def hack_dataset(dataset):
    def wrap_callable():
        return dataset
    return wrap_callable


def postprocess(img):
    img = (img + 1)*127.5
    return img.astype(np.uint8)


def preprocess_inputs(img, tgt):
    img = img.astype(np.float32) / 127.5 - 1
    return img[:512, :512, ...], tgt[:512, :512, ...]


def setup_dataset(coco_path, model_config, training_config):
    output_shapes = (
        model_config['input_shape'],
        model_config['input_shape'][:-1] + (model_config['classes'],))
    dataset_train = ImageSemanticSegmentationDataset(
        coco_path, num_classes=model_config["classes"])
    gen_train = dataset_train.generator(endless=True)
    gen_train = lambda_gen(gen_train, func=preprocess_inputs)
    gen_train = tf.data.Dataset.from_generator(
        hack_dataset(gen_train),
        output_types=(tf.float32, tf.float32),
        output_shapes=output_shapes).batch(training_config["batch_size"])
    return dataset_train, gen_train


def main(args):
    """Summary

    Args:
        args (TYPE): Description
    """
    logging.basicConfig(
        format='%(filename)s: %(asctime)s.%(msecs)d: %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.loglevel)
    logging.info('--verbose enabled')
    logging.info(args)

    # Load configs
    model_config = deeplab.DEFAULT_CONFIG
    training_config = DEFAULT_CONFIG
    if args.output_config:
        print(json.dumps(model_config))
        return 0
    with open(args.config, "r") as file:
        config = json.load(file)
        model_config.update(config['model'])
        training_config.update(config['training'])
    model_config["input_shape"] = tuple(model_config["input_shape"])
    logging.info("Model config:")
    logging.info(model_config)
    logging.info("Training config:")
    logging.info(training_config)

    # Training
    if training_config["new_heads"]:
        logging.info("Creating new head.")
        model = deeplab.make_model(**model_config)
    else:
        logging.info("Loading model.")
        model = deeplab.load(args.model_json, args.weights)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='weights.{epoch:03d}-{val_loss:.3f}.h5',
            save_best_only=True, save_weights_only=True, verbose=1)
    ]
    dataset_train, gen_train = setup_dataset(
        args.dataset_train, model_config, training_config)
    validation_steps, gen_val = None, None
    if args.dataset_val:
        dataset_val, gen_val = setup_dataset(
            args.dataset_val, model_config, training_config)
        validation_steps = len(
            dataset_val.data_ids) // training_config["batch_size"]
    steps_per_epoch = len(dataset_train.data_ids) // training_config["batch_size"]

    optimizer = tf.keras.optimizers.Nadam(lr=1e-4)
    with open("model_def.json", "w") as file:
        file.write(model.to_json())
    exit(1)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    model.fit(
        gen_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=gen_val,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1)
    return 0


if __name__ == "__main__":
    exit(main(get_args()))
