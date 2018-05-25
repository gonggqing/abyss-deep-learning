# Metrics for machine learning
import os
import sys

from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from abyss_maskrcnn.utils import compute_overlaps

def _flat_np(df):
    return df.apply(pd.Series).as_matrix()

def result_to_series(result):
    num_instances = len(result['rois'])
    frame = []
    for idx in range(num_instances):
        frame.append({
            'roi': result['rois'][idx, :],
            'score': result['scores'][idx].astype(float) if 'scores' in result else 0.0,
            'mask': result['masks'][..., idx],
            'class_id': result['class_ids'][idx].astype(int)
        })
    return pd.DataFrame(frame, columns=frame[0].keys())

def calculate_tfpn(predicted, object_gts, iou_thresh=0.5, bipartite=False):
    '''Calculate True Positive, False Positive and False Negatives given two objects of the same class
       Definitions:
        TP = number of detections with IoU>0.5
        FP = number of detections with IoU<=0.5 (when bipartite=False)
        FP = number of detections with IoU<=0.5 or detected more than once (when bipartite=True)
        FN = number of objects that not detected or detected with IoU<=0.5
        input is pandas.DataFrame as returned by result_to_series
    '''
    # ious index is [pred_idx, gt_idx]
    ious = compute_overlaps(_flat_np(predicted['roi']), _flat_np(object_gts['roi']))
    if bipartite:
        # Data association (bipartite matching: one prediction to one ground truth)
        TP = np.zeros(len(predicted), dtype=np.bool)
        FP = np.zeros(len(predicted), dtype=np.bool)
        FN = np.product(ious <= iou_thresh, axis=0) > 0
        nulls = np.prod(ious <= 1e-6, axis=1) > 0  #Need this so linear_sum_assignment works
        
        pred_idxs = np.arange(len(predicted))[~nulls]
        pred_idx_ptr, gt_idx = linear_sum_assignment(1 - ious[~nulls, :])
        matched_ious = ious[pred_idxs[pred_idx_ptr], gt_idx]
        matched_idxs = pred_idxs[pred_idx_ptr]
        
        TP[matched_idxs] = matched_ious > iou_thresh
        FP[matched_idxs] = (matched_ious <= iou_thresh)
        FP |= nulls
    else:
        # Data association (many predictions to many ground truths) 
        TP = np.sum(ious > iou_thresh, axis=1) > 0
        FP = np.product(ious <= iou_thresh, axis=1) > 0
        FN = np.product(ious <= iou_thresh, axis=0) > 0
    return (TP, FP, FN)

def calc_image_stats(predicted, object_gts):
    image_stats = {}
    for class_id in np.unique(predicted['class_id']):
        (TP, FP, FN) = calculate_tfpn(
        predicted[predicted['class_id'] == class_id],
        object_gts[predicted['class_id'] == class_id],
        bipartite=True)
        image_stats[class_id] = {
            'TP': TP,
            'FP': FP,
            'FN': FN,
        }
    return image_stats
