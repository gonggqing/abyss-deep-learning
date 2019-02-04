#!/usr/bin/env python3

# example of running it
#cd /home/users/spo/src/projects/anadarko && \
#CUDA_VISIBLE_DEVICES=0 \
#~/src/deep-learning/examples/predict_deeplab_tiled.py \
#--verbose --output-format '/tmp/pred/{filename:}{extension:}' \
#frozen_inference_graph.pb \
#'cubes_small/CD\ -\ 08*_dn_f1.jpg'

import argparse
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import numpy as np
from abyss_deep_learning.utils import tile_gen, detile, image_streamer, config_gpu
import abyss_deep_learning.draw as draw

COLOR_MAP = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
]

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())
    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def setup_graph(args):
    graph = load_graph(args.graph_path)
    graph_ops = graph.get_operations()
    args.inputs_tensor = graph.get_tensor_by_name(args.input_tensor if args.inputs_tensor else graph_ops[0].name + ":0")
    args.outputs_tensor = graph.get_tensor_by_name(args.input_tensor if args.outputs_tensor else graph_ops[-1].name + ":0")
    assert args.inputs_tensor is not None, "Input tensor not found"
    assert args.outputs_tensor is not None, "Predictions tensor not found"
    return graph


def get_args():
    '''Get args from the command line args'''
    parser = argparse.ArgumentParser(
        description="Predict")
    parser.add_argument("graph_path", help="Path to the frozen inference graph.")
    parser.add_argument("images", nargs='+', help="Path to the images to predict on, globs allowed.")
    parser.add_argument(
        '--output-format',
        help="The str.format string to use when saving predictions; default '{dirname:}/{filename:}_prediction.png",
        default="{dirname:}/{filename:}_prediction.png")
    parser.add_argument("--inputs-tensor", help="The inputs tensor to feed images to. Defaults to first operation in the graph.")
    parser.add_argument("--outputs-tensor", help="The output tensor to retrieve the predicted labels from. Defaults to last operation in the graph.")
    parser.add_argument("--tile-size", help="The size at which the images should be fed to the graph. Default: 513,513.)", default="513,513", type=str)
    parser.add_argument("--verbose", "-v", help="Display progress", action="store_true")
    args = parser.parse_args()
    args.tile_size = tuple([int(i) for i in args.tile_size.split(',')])
    assert len(args.tile_size) == 2, "Tile size must be 2D."
    return args


def main(args):
    gpus = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(",")] if 'CUDA_VISIBLE_DEVICES' in os.environ else []
    config_gpu(gpus, allow_growth=True, log_device_placement=False)
    graph = setup_graph(args)
    with tf.Session(graph=graph) as sess:
        for image_path, frame_no, image in image_streamer(args.images):
            if args.verbose:
                print(image_path, file=sys.stderr)
            filename, extension = os.path.splitext(os.path.basename(image_path))
            parts = {
                'dirname': os.path.dirname(image_path),
                'extension': extension,
                'filename': filename,
                'frame_no': frame_no,
            }
            predicted = detile([
                sess.run(
                    args.outputs_tensor,
                    feed_dict={args.inputs_tensor: tile[np.newaxis, ...]})[0]
                for tile in tile_gen(image, args.tile_size)], args.tile_size, image.shape[:2])
            predicted_rgb = draw.masks(
                predicted, image,
                image_alpha=1, alpha=0.5, border=True, bg_label=0, colors=COLOR_MAP)
            imsave(args.output_format.format(**parts), predicted_rgb)


if __name__ == '__main__':
    main(get_args())

