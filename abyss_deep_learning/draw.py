"""
Module for visualizing various machine learning outputs.

"""

import numpy as np
import cv2
import skimage.color as skic

def masks(labels, image=None, colors=None, alpha=0.3, image_alpha=1, bg_label=-1, bg_color=(0, 0, 0), kind='overlay'):
    """Draws mask on image or just mask, if image not present
    
    todo: document parameters
    
    Returns
    -------
    np.ndarray
        RGB base-256 image
    """
    overlay = skic.label2rgb( labels, alpha=1, bg_label=bg_label, bg_color=bg_color, kind=kind )
    overlay *= 255
    overlay = overlay.astype( np.uint8 )
    if image is None: return overlay
    return cv2.addWeighted( image, image_alpha, overlay, alpha, 0 )

def draw_semantic_seg(labels, rgb, class_idxs=None, num_classes=None):
    """Draws the semantic segmentation on to an RGB image.
    If labels are a score map the maximum scoring class is used at each pixel.
    
    Parameters
    ----------
    labels : np.ndarray
        The segmentation to use. Can be one of three forms:
          * Score array of shape [height, width, #classes] of np.float type, where each value
            indicates the score of that pixel being in the class specified by the last dimension.
          * Occupancy array of shape [height, width, #instances] of np.bool type, where each value
            with class_idxs of shape [#instances] and specifying the class for each instance.
          * Label image of shape [height, width] of integer type, where each value represents a class.
    rgb : np.ndarray
        The image to impose the results on to.
    class_idxs : np.ndarray, optional
        An integer array that specifies the class of each instance  if an occupancy array is being used.
    num_classes : int, optional
        The maximum number of classes to represent if an occupancy array is being used.
    
    Returns
    -------
    np.ndarray
        The image with the semantic segmentation overlay.
    """
    if labels.ndim == 3:
        if labels.shape[2] == 1: # An alias for a label image
            labels = labels[..., 0]
        elif class_idxs: # Occupancy array with last dimension mapping instances
            labels = instance_to_categorical(labels, class_idxs, num_classes)
        else: # Score array with last dimension mapping class
            labels = labels.argmax(-1)
    if labels.ndim != 2 or not issubclass(labels.dtype.type, np.integer):
        raise ValueError("labels must be shape [height, width] and of integer type.")
    return label2rgb(labels, rgb, bg_label=0, gray_bg=False, contours='thick')

def draw_instance_seg(labels, rgb, class_idxs=None):
    """Draws the instance segmentation on to an RGB image.
    
    Parameters
    ----------
    labels : np.ndarray
        The segmentation to use. Can be one of three forms:
          * Score map of shape [height, width, #classes] of np.float type, where each value
            indicates the score of that pixel being in the class specified by the last dimension.
          * Score map of shape [height, width, #instances] of np.float type, where each value
            with class_idxs of shape [#instances] and specifying the class for each instance.
          * Label image of shape [height, width] of integer type, where each value represents a class.
    rgb : np.ndarray
        The image to impose the results on to.
    class_idxs : np.ndarray, optional
        An integer array that specifies the class of each instance for the last dimension of the labels array.
    """
    
    output = label2rgb(label.argmax(-1), ARGS['postprocess_data'](image), bg_label=0)

    if class_idxs:
        pass
