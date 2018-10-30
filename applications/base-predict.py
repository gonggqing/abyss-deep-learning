#!/usr/bin/env python3
"""This application is a non-functional template for future applications.
It adheres to the Abyss extensions for the sklearn interface to models, and
the Abyss Dataset interface (abyss_deep_learning.datasets.base).
"""
from __future__ import print_function
import argparse
import os
import pickle
import sys

from abyss_deep_learning.utils import warn_once

### Helper functions ###

class PredictorBaseApp(object):
    def __init__(self, args):
        from queue import Queue
        self.args = args
        self.model = self.initialize_model()
        self.input_queue = Queue()
        self.output_stream = None
        self.batch = []

    def initialize_model(self):
        """Initialize the model in class.
        
        Raises:
            NotImplementedError: If this method hasn't been specialized.
            ValueError: If the specified model weights do not exist.
        """

        model = None
        ### Load your model here
        if model is None:
            raise NotImplementedError(
                "Model is None. You may have forgotten to specialize the initialise_model function for your model.")
        if self.args.weights is not None:
            if not os.path.exists(self.args.weights):
                raise ValueError("Model weights specified but do not exist.")
            model.load_weights(self.args.weights, by_name=True)
        return model

    @warn_once("Method not overwritten, using default behavior.")
    def filter_input(self, path, args):
        """Determine whether or not to use a particular input file.
            By default accepts all inputs.
        Args:
            path (str): Path to the file.
            args (dict): The args used for the model.

        Returns:
            boolean: True if using the image, else False.
        """
        return True


    @warn_once("Method not overwritten, using default behavior.")
    def filter_predictions(self, path, image, results, args):
        """After a frame has been predicted this function can transform the results before
            they are used in the output stage. By default does nothing and returns the result unaltered.

        Args:
            path (str): Path to the input.
            image (np.ndarray): The frame that was used for prediction.
            results (object): The results of the prediciton for this frame.
            args (argparse.Namespace): The CLI args specified.

        Returns:
            object: The filtered result.

        """
        return results

    def process_frame(self, frame, path, frame_no, csv_stream):
        """Predict on the frame and write appropriate outputs.
        Filter frames according to filter_input and filter_predictions.

        Args:
            frame (np.ndarray): The frame image to predict on.
            path (str): The path to the input.
            frame_no (int): The frame number of the image in the input.
            csv_stream (file): The output CSV stream for this input.

        Raises:
            NotImplementedError: TODO
        """
        if not self.filter_input(self, path, args):
            return
        raise NotImplementedError("TODO")
        return self.filter_predictions(self, path, image, results, args)

### STUBS ###



def output_classification(csv_handle, path, image, results, args):
    """Outputs the classification top-N results (specified in args.topN) to CSV.
        CSV consists of top-N class probabilities and names.

    Args:
        csv_handle (file): File handle to the output CSV file.
        path (str): Path to the input.
        image (np.ndarray): The frame that was used for prediction.
        result (object): The results of the prediciton for this frame.
        args (argparse.Namespace): The CLI args specified.

    """
    raise NotImplementedError("TODO")


def output_instance_detection(csv_handle, path, image, results, args):
    """Outputs the instance detection top-N results (specified in args.topN) to CSV and images.
        CSV consists of top-N class probabilities and names and bounding box extents.
        Output images of bounding box overlays and most probable class name.

    Args:
        csv_handle (file): File handle to the output CSV file.
        path (str): Path to the input.
        image (np.ndarray): The frame that was used for prediction.
        result (object): The results of the prediciton for this frame.
        args (argparse.Namespace): The CLI args specified.

    """
    raise NotImplementedError("TODO")


def output_semantic_segmentation(csv_handle, path, image, results, args):
    """Outputs the instance detection top-N results (specified in args.topN) to CSV and images.
        CSV consists of top-N class probabilities, names and pixel count.
        Output images of instances outlined and shaded by class color, and most probable class name.

    Args:
        csv_handle (file): File handle to the output CSV file.
        path (str): Path to the input.
        image (np.ndarray): The frame that was used for prediction.
        result (object): The results of the prediciton for this frame, assumed to be [H, W, # classes].
        args (argparse.Namespace): The CLI args specified.

    """



def output_instance_segmentation(csv_handle, path, image, results, args):
    """Outputs the instance detection top-N results (specified in args.topN) to CSV and images.
        CSV consists of top-N class probabilities, names, bounding box extents and pixel count.
        Output images of instances outlined and shaded by class color, and most probable class name.

    Args:
        csv_handle (file): File handle to the output CSV file.
        path (str): Path to the input.
        image (np.ndarray): The frame that was used for prediction.
        result (object): The results of the prediciton for this frame.
        args (argparse.Namespace): The CLI args specified.

    """
    raise NotImplementedError("TODO")


def predict(model, batch, args):
    """Use the model to predict the batch of inputs.

    Args:
        model (sklearn model): The initialized model to use to predict.
        batch (list of objects): The list of input objects, varies according to model inputs.
        args (argparse.Namespace): The CLI args specified.
    """
    raise NotImplementedError("TODO")

### MAIN APPLICATION ###


def get_args():
    '''Process the command line args, and load args.config with the JSON/YAML config.
    
    Returns:
        argparse.Namespace: The CLI args, with the config loaded.
    '''
    import argparse
    from abyss_deep_learning.utils import import_config

    class MultiInputAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Set optional arguments to True or False
            if option_string:
                attr = True if values else False
                setattr(namespace, self.dest, attr)
            # Modify value of "input" in the namespace
            if hasattr(namespace, 'inputs'):
                current_values = getattr(namespace, 'inputs')
                try:
                    current_values.extend(values)
                except AttributeError:
                    current_values = values
                finally:
                    setattr(namespace, 'inputs', current_values)
            else:
                setattr(namespace, 'inputs', values)

    parser = argparse.ArgumentParser(
        description="Simultaneous training and validation of Resnet Mask RCNN")
    parser.add_argument(
        "config", help="Path to the model config JSON/YAML.")
    parser.add_argument(
        "devices", help="Use the given devices (csv-list)", default=None)
    parser.add_argument('inputs', nargs='+', action=MultiInputAction)

    args = parser.parse_args()
    args.config = import_config(args)
    return args


def main(args):
    """Run the application main procedure.

    Args:
        args (TYPE): An argparse namespace for CLI arguments.
    """

    from abyss_deep_learning.utils import config_gpu

    # Set up devices
    config_gpu(args.devices)

    # Process input strings

    # Filter inputs

    # Form input queue

    # Create frame stream
    stream_in = None

    # Process each frame in stream
    for frame, path, frame_no, csv_stream in stream_in:
        process_frame(frame, path, frame_no, csv_stream)

if __name__ == '__main__':
    main(get_args())
