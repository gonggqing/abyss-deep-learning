"""Module for visualizing various machine learning outputs.

Attributes
----------
COLOR_DICT : dict(str - > tuple(float, float, float))
    dict mapping color strings to RGB values in the range [0, 1].
DEFAULT_COLORS : list of str
    Default keys in COLOR_DICT to use.
"""
import numpy as np

from skimage._shared.utils import warn
from skimage.color import rgb_colors, rgb2gray, gray2rgb
from skimage.color.colorlabel import _rgb_vector, _match_label_with_color
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from abyss_deep_learning.utils import instance_to_categorical

__all__ = ['COLOR_DICT', 'label2rgb', 'DEFAULT_COLORS']


COLOR_DICT = {
    k: v for k, v in rgb_colors.__dict__.items()
    if isinstance(v, tuple)}

DEFAULT_COLORS = (
    'red', 'blue', 'yellow', 'magenta', 'green',
    'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')



def label2rgb(
        label, image=None, colors=None, alpha=0.3,
        gray_bg=False, contours='thick',
        bg_label=-1, bg_color=(0, 0, 0), image_alpha=1, kind='overlay'):
    """Return an RGB image where color-coded labels are painted over the image, and optionally contours are painted.
    Source: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorlabel.py

    Parameters
    ----------
    label : array, shape (M, N)
        Integer array of labels with the same shape as `image`.
    image : array, shape (M, N, 3), optional
        Image used as underlay for labels. If the input is an RGB image, it's
        converted to grayscale before coloring.
    colors : list, optional
        List of colors. If the number of labels exceeds the number of colors,
        then the colors are cycled.
    alpha : float [0, 1], optional
        Opacity of colorized labels. Ignored if image is `None`.
    gray_bg : bool, optional
        Set the background image to grayscale when mode='overlay'.
    contours : str, optional
        Description
    bg_label : int, optional
        Label that's treated as the background.
    bg_color : str or array, optional
        Background color. Must be a name in `COLOR_DICT` or RGB float values
        between [0, 1].
    image_alpha : float [0, 1], optional
        Opacity of the image.
    kind : string, one of {'overlay', 'avg'}
        The kind of color image desired. 'overlay' cycles over defined colors
        and overlays the colored labels over the original image. 'avg' replaces
        each labeled segment with its average color, for a stained-class or
        pastel painting appearance.
    contours, : string in {‘thick’, ‘inner’, ‘outer’, ‘subpixel’}, optional
        The mode for finding and drawing class spatial boundaries, use None to not draw contours.

    Returns
    -------
    result : array of float, shape (M, N, 3)
        The result of blending a cycling colormap (`colors`) for each distinct
        value in `label` with the image, at a certain alpha value.
    """
    if kind == 'overlay':
        return _label2rgb_overlay(label, image, colors, alpha, bg_label,
                                  bg_color, image_alpha, gray_bg=gray_bg, contours=contours)
    return _label2rgb_avg(label, image, bg_label, bg_color)

def _label2rgb_overlay(label, image=None, colors=None, alpha=0.3,
                       bg_label=-1, bg_color=None, image_alpha=1, gray_bg=False, contours=None):
    """Return an RGB image where color-coded labels are painted over the image.

    Parameters
    ----------
    label : array, shape (M, N)
        Integer array of labels with the same shape as `image`.
    image : array, shape (M, N, 3), optional
        Image used as underlay for labels. If the input is an RGB image, it's
        converted to grayscale before coloring.
    colors : list, optional
        List of colors. If the number of labels exceeds the number of colors,
        then the colors are cycled.
    alpha : float [0, 1], optional
        Opacity of colorized labels. Ignored if image is `None`.
    bg_label : int, optional
        Label that's treated as the background.
    bg_color : str or array, optional
        Background color. Must be a name in `COLOR_DICT` or RGB float values
        between [0, 1].
    image_alpha : float [0, 1], optional
        Opacity of the image.
    gray_bg : bool, optional
        Set the background image to grayscale when mode='overlay'.
    contours : None, optional
        Description
    contours, : string in {‘thick’, ‘inner’, ‘outer’, ‘subpixel’}, optional
        The mode for finding and drawing class spatial boundaries, use None to not draw contours.

    Returns
    -------
    result : array of float, shape (M, N, 3)
        The result of blending a cycling colormap (`colors`) for each distinct
        value in `label` with the image, at a certain alpha value.

    Raises
    ------
    ValueError
        When image and label are not the same shape.
    """
    if colors is None:
        colors = DEFAULT_COLORS
    colors = [_rgb_vector(c) for c in colors]

    if image is None:
        image = np.zeros(label.shape + (3,), dtype=np.float64)
        # Opacity doesn't make sense if no image exists.
        alpha = 1
    else:
        if not image.shape[:2] == label.shape:
            raise ValueError("`image` and `label` must be the same shape")

        if image.min() < 0:
            warn("Negative intensities in `image` are not supported")
        if gray_bg:
            image = img_as_float(rgb2gray(image))
            image = gray2rgb(image) * image_alpha + (1 - image_alpha)
        else:
            image = img_as_float(image)

    # Ensure that all labels are non-negative so we can index into
    # `label_to_color` correctly.
    offset = min(label.min(), bg_label)
    if offset != 0:
        label = label - offset  # Make sure you don't modify the input array.
        bg_label -= offset

    new_type = np.min_scalar_type(int(label.max()))
    if new_type == np.bool:
        new_type = np.uint8
    label = label.astype(new_type)

    mapped_labels_flat, color_cycle = _match_label_with_color(label, colors,
                                                              bg_label, bg_color)

    if len(mapped_labels_flat) == 0:
        return image

    dense_labels = range(max(mapped_labels_flat) + 1)

    label_to_color = np.array([c for i, c in zip(dense_labels, color_cycle)])

    mapped_labels = label
    mapped_labels.flat = mapped_labels_flat
    result = label_to_color[mapped_labels] * alpha + image * (1 - alpha)

    # Remove background label if its color was not specified.
    remove_background = 0 in mapped_labels_flat and bg_color is None
    if remove_background:
        result[label == bg_label] = image[label == bg_label]

    if contours:
        for label_idx in range(label_to_color.shape[0]):
            result = mark_boundaries(
                result, label == label_idx, color=label_to_color[label_idx], mode=contours)

    return result


def _label2rgb_avg(label_field, image, bg_label=0, bg_color=(0, 0, 0)):
    """Visualise each segment in `label_field` with its mean color in `image`.

    Parameters
    ----------
    label_field : array of int
        A segmentation of an image.
    image : array, shape ``label_field.shape + (3,)``
        A color image of the same spatial shape as `label_field`.
    bg_label : int, optional
        A value in `label_field` to be treated as background.
    bg_color : 3-tuple of int, optional
        The color for the background label

    Returns
    -------
    out : array, same shape and type as `image`
        The output visualization.
    """
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        color = image[mask].mean(axis=0)
        out[mask] = color

    return out


########   Generic View Functions    #########


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
