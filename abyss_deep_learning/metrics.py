"""Metrics for machine learning"""
from typing import List, Tuple

import numpy as np
import skimage
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon, polygon_perimeter, line

from abyss_deep_learning.utils import do_overlap


def bbox_iou_matrix(a, b):
    """ Calculate MxN matrix of IOU for two sets of bounding boxes, e.g: iou( predictions, ground_truth )

    parameters
    ----------
    a: numpy.array
       first Mx4 array of box begin/end coordinates as y1,x1,y2,x2
    b: numpy.array
       second Nx4 array of box begin/end coordinates as y1,x1,y2,x2

    returns
    -------
    numpy.array, MxN matrix of IOU values

    todo
    ----
    refactor, if too slow
    """
    intersections = get_bbox_intersection(a, b)
    ta = np.transpose(a)
    tb = np.transpose(b)
    aa = (ta[3] - ta[1]) * (ta[2] - ta[0])
    ba = (tb[3] - tb[1]) * (tb[2] - tb[0])
    a_areas = np.repeat([aa], len(b), axis=0)
    a_areas = np.transpose(a_areas)
    b_areas = np.repeat([ba], len(a), axis=0)
    return intersections / (a_areas + b_areas - intersections)


def get_bbox_intersection(bbox_a, bbox_b):
    ac = np.repeat([bbox_a], len(bbox_b), axis=0)
    ac = np.transpose(ac, axes=(2, 1, 0))
    bc = np.repeat([bbox_b], len(bbox_a), axis=0)
    bc = np.transpose(bc, axes=(2, 0, 1))
    minimums = np.minimum(ac, bc)
    maximums = np.maximum(ac, bc)
    intersections = np.maximum(minimums[3] - maximums[1], 0) * np.maximum(minimums[2] - maximums[0], 0)
    return intersections


# TODO: transition from sklearn contour operations to opencv contour operations
def poly_intersection_area(first: List[np.array], second: List[np.array], grid_max_x: int = None,
                           grid_max_y: int = None) -> Tuple[np.array, np.array, np.array]:
    """
    parameters
    ----------
        first: list of N x 2 np.array of co-ordinate points (x, y)
        second: list of N x 2 np.array of co-ordinate points (x, y)
        grid_max_y: max y value + 1 in list of np.array (can use 'bbox' field in annotation from COCO file)
        grid_max_x: max x value + 1 in list of np.array (can use 'bbox' field in annotation from COCO file)

    returns
    -------
        numpy.array, numpy.array, numpy.array: first polygon areas vector, second polygon areas vector, intersection areas matrix

    """

    precompute_first = []
    precompute_second = []
    if grid_max_x is None or grid_max_y is None:
        grid_max_x = 0
        grid_max_y = 0
        for array in first + second:
            # Get max value per row entry in the array
            upper_val_x, upper_val_y = array.max(axis=0)
            if upper_val_x > grid_max_x:
                grid_max_x = int(upper_val_x) + 1
            if upper_val_y > grid_max_y:
                grid_max_y = int(upper_val_y) + 1

    grid = np.zeros((grid_max_y, grid_max_x), dtype=np.uint8)
    first_areas = []
    second_areas = []
    first_bboxes = []
    second_bboxes = []

    def draw_grid(array_: np.array, bboxes: List[Tuple[int, int, int, int]], grid_: np.array) -> None:
        array_ = np.round(array_).astype(int)
        r = array_[:, 1]
        c = array_[:, 0]
        r[r < 0] = 0
        c[c < 0] = 0
        r[r >= grid_max_y] = grid_max_y - 1
        c[c >= grid_max_x] = grid_max_x - 1
        bboxes.append((min(c), min(r), max(c), max(r)))

        nx = len(np.unique(c))
        ny = len(np.unique(r))
        if nx > 2 or ny > 2:
            grid_[polygon_perimeter(r, c)] = 1
        elif nx == 2 or ny == 2:
            for i in range(len(c) - 1):
                if r[i] == r[i + 1] and c[i] == c[i + 1]:
                    grid_[r, c] = 1  # single pixel drawn
                else:
                    grid_[line(r[i], c[i], r[i + 1], c[i + 1])] = 1  # draw line
        elif nx == 1 and ny == 1:
            grid_[r, c] = 1
        else:
            raise ValueError(f"polygon malformed, unable to draw onto grid a polygon of shape {array_.shape}")

    for array in first:
        draw_grid(array, first_bboxes, grid)
        first_areas.append(np.count_nonzero(grid))
        precompute_first.append(np.array(grid))  # make a copy of grid
        grid[:] = 0
    for array in second:
        draw_grid(array, second_bboxes, grid)
        second_areas.append(np.count_nonzero(grid))
        precompute_second.append(np.array(grid))  # make a copy of grid
        grid[:] = 0

    intersections = []
    for i in range(len(precompute_first)):
        tmp = []
        first_bbox = first_bboxes[i]
        for j in range(len(precompute_second)):
            second_bbox = second_bboxes[j]
            tmp.append(
                np.count_nonzero(np.logical_and(precompute_first[i], precompute_second[j])) if do_overlap(first_bbox,
                                                                                                          second_bbox) else 0)
        intersections.append(tmp)
    return first_areas, second_areas, np.array(intersections, dtype=np.uint)


def poly_iou_matrix(predictions_array: List[np.array], truth_arrays: List[np.array],
                    grid_max_x: int = None, grid_max_y: int = None):
    """

    Args:
        grid_max_y: max y value + 1 in list of np.array (can use 'bbox' field in annotation from COCO file)
        grid_max_x: max x value + 1 in list of np.array (can use 'bbox' field in annotation from COCO file)
        predictions_array: list of N x 2 np.array of co-ordinate points (x, y)
        truth_arrays: list of N x 2 np.array of co-ordinate points (x, y)

    Returns:
        numpy.array: iou matrix with rows corresponding to the first input and columns to the second

    """
    first, second, intersections = poly_intersection_area(predictions_array, truth_arrays, grid_max_x, grid_max_y)
    len_first = len(first)
    len_second = len(second)
    first = np.transpose(np.repeat([first], len_second, axis=0))
    second = np.repeat([second], len_first, axis=0)
    # result = np.zeros( intersections.shape )
    unions = first + second - intersections
    iou = np.divide(intersections, unions, where=unions != 0)  # if union is zero, intersection will be zero, too
    return iou


def poly_iou_matrix_deprecated(predictions_array: List[np.array], truth_arrays: List[np.array], grid_max_x: int = None,
                               grid_max_y: int = None):
    """

    Args:
        grid_max_y: max y value + 1 in list of np.array (can use 'bbox' field in annotation from COCO file)
        grid_max_x: max x value + 1 in list of np.array (can use 'bbox' field in annotation from COCO file)
        predictions_array: list of N x 2 np.array of co-ordinate points (x, y)
        truth_arrays: list of N x 2 np.array of co-ordinate points (x, y)

    Returns:

    """
    precompute_predictions = []
    precompute_truths = []

    # Todo completely re-write me to take in to account lower offset by lower value?
    if grid_max_x is None or grid_max_y is None:
        grid_max_x = 0
        grid_max_y = 0
        for array in predictions_array + truth_arrays:
            # Get max value per row entry in the array
            upper_val_x, upper_val_y = array.max(axis=0)
            if upper_val_x > grid_max_x:
                grid_max_x = int(upper_val_x) + 1
            if upper_val_y > grid_max_y:
                grid_max_y = int(upper_val_y) + 1

    grid = np.zeros((grid_max_y, grid_max_x), dtype=np.uint8)
    for array in predictions_array:
        grid[array[:, 1], array[:, 0]] = 1
        grid[skimage.draw.polygon(array[:, 1], array[:, 0], grid.shape)] = 1
        precompute_predictions.append(np.array(grid))
        grid[:] = 0

    for array in truth_arrays:
        grid[array[:, 1], array[:, 0]] = 1
        grid[skimage.draw.polygon(array[:, 1], array[:, 0], grid.shape)] = 1
        precompute_truths.append(np.array(grid))
        grid[:] = 0

    result = []
    for grid_prediction in precompute_predictions:
        sub_result = []
        for grid_truth in precompute_truths:
            union = np.logical_or(grid_prediction, grid_truth)
            intersection = np.logical_and(grid_prediction, grid_truth)
            union_count = np.count_nonzero(union)
            if union_count:
                sub_result.append(np.count_nonzero(intersection) / union_count)
            else:
                sub_result.append(0)
        result.append(sub_result)
    return np.array(result)


def ious_to_sklearn_pred_true(ious, labels_true, labels_pred, iou_threshold=0., blank_id=0):
    """ Convert labelled bboxes to y_true and y_pred that could be consumed directly by sklearn.metrics functions

    example
    -------
    p, t = bbox_to_sklearn_pred_true( bbox_iou_matrix( a, b ), a_labels, b_labels, 0.5 )
    print( sklearn.metrics.confusion_matrix( p, t ) )

    parameters
    ----------
    ious: numpy.array, iou matrix as returned by e.g. by bbox_iou_matrix( a, b )
    labels_true: numpy.array, vector of ground truth bbox labels
    labels_pred: numpy.array, vector of predicted bbox labels
    iou_threshold: float, iou threshold

    returns
    -------
    numpy.array, numpy.array: y_true, y_pred as in sklearn.metrics
    """
    m = (ious > iou_threshold) * 1
    i = np.nonzero(m)
    fp = np.nonzero(np.max(m, axis=1) == 0)[0]
    fn = np.nonzero(np.max(m, axis=0) == 0)[0]
    y_true = np.concatenate((np.array(labels_true)[i[1]], [blank_id] * len(fp), np.array(labels_true)[fn]))
    y_pred = np.concatenate((np.array(labels_pred)[i[0]], np.array(labels_pred)[fp], [blank_id] * len(fn)))
    return y_true, y_pred


def one_to_one(ious, iou_threshold=None):
    """ Match two sets of boxes one to one by IOU on given treshold

    parameters
    ----------
    ious: numpy.array, iou matrix as returned by e.g. by bbox_iou_matrix( a, b )

    returns
    -------
    numpy.array, MxN matrix of matched IOU values
    """
    ious *= (ious >= (0. if iou_threshold is None else iou_threshold))
    rows, cols = linear_sum_assignment(1 - ious)
    result = np.zeros(ious.shape)
    result[rows, cols] = ious[rows, cols] * (ious[rows, cols] > 0)
    return result


def one_to_many(ious, iou_threshold=None):
    """ Match two sets of boxes one to many on max IOU by IOU on given treshold

    parameters
    ----------
    ious: numpy.array, iou matrix as returned by e.g. by bbox_iou_matrix( a, b )
    iou_threshold: float, iou threshold

    returns
    -------
    numpy.array, MxN matrix of matched IOU values
    """
    ious *= (ious >= (0. if iou_threshold is None else iou_threshold))
    flags = np.zeros(ious.shape)
    flags[np.argmax(ious, axis=0), [*range(ious.shape[1])]] = 1
    return ious * flags


def many_to_one(ious, iou_threshold=None):
    """ Match two sets of boxes many to one on max IOU by IOU on given treshold

    parameters
    ----------
    ious: numpy.array, iou matrix as returned by e.g. by bbox_iou_matrix( a, b )
    iou_threshold: scalar iou threshold

    returns
    -------
    numpy.array, MxN matrix of matched IOU values
    """
    return np.transpose(one_to_many(np.transpose(ious), iou_threshold))


def tp_fp_tn_fn(predictions, truth, threshold=None, match=one_to_one, iou_matrix=bbox_iou_matrix):
    """ Return TP, FP, TN, FN

    parameters
    ----------
    predictions: numpy.array
                 predictions as Mx4 array of box begin/end coordinates as y1,x1,y2,x2
    truth: numpy.array
           ground truth as Nx4 array of box begin/end coordinates as y1,x1,y2,x2
    threshold: float, IOU threshold
    match: function, how to match
    iou_matrix: function, how to produce iou matrix

    returns
    -------
    TP, FP, TN, FN
    TP: numpy.array, 2d; indices of TP in predictions matched with indices in truth
    FP: numpy.array, 1d; indices of FP in a
    TN: numpy.array, 1d; indices of TN, empty for bounding boxes, since it does not make sense
    FN: numpy.array, 1d; indices of FN in b
    """
    if len(predictions) == 0:
        return [], [], [], [*range(len(*truth))]
    ious = iou_matrix(predictions, truth)
    matched = (match(ious, threshold) > 0) * 1
    thresholded = ((ious > (0. if threshold is None else threshold)) > 0) * 1
    tpi = np.nonzero(matched)
    fpi = np.nonzero(np.max(thresholded, axis=1) == 0)[0]
    tni = []
    fni = np.nonzero(np.max(thresholded, axis=0) == 0)[0]
    return tpi, fpi, tni, fni
