'''Metrics for machine learning'''
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
    if not frame:
        return None
    return pd.DataFrame(frame, columns=frame[0].keys())

def calculate_tfpn(predicted, object_gts, matching='one-to-one', iou_thresh=0.5):#, scores=None, score_thresh=None):
    '''Calculate True Positive, False Positive and False Negatives given two objects of the same class
        inputs are pandas.DataFrame as returned by result_to_series
    '''
    def raise_(ex):
        raise ex
    # ious index is [pred_idx, gt_idx]
    
    predicted = None if not predicted.size else _flat_np(predicted['roi'])
    object_gts = None if not object_gts.size else _flat_np(object_gts['roi'])
    if predicted is not None and object_gts is not None:
        ious = compute_overlaps(predicted, object_gts)
        TP, FP, FN = {
            'one-to-one': matching_one_to_one,
            'one-to-many': lambda _, __: raise_(NotImplementedError("one-to-many not yet implemented")),
            'many-to-one': matching_many_to_one,
            'many-to-many': matching_many_to_many
        }[matching](ious, iou_thresh) #, scores=scores, score_thresh=score_thresh)
    elif predicted is None and object_gts is not None:
        TP, FP, FN = 0, 0, len(object_gts)
    elif predicted is not None and object_gts is None:
        TP, FP, FN = 0, len(predicted), 0
    elif predicted is None and object_gts is None:
        TP, FP, FN = 0, 0, 0
    return TP, FP, FN

def calc_image_stats(predicted, object_gts, matching, iou_thresh=0.5, score_thresh=None):
    unique_classes = set()
    if predicted is not None:
        unique_classes |= set(np.unique(predicted['class_id']).tolist())
    if object_gts is not None:
        unique_classes |= set(np.unique(object_gts['class_id']).tolist())
    image_stats = []
    for class_id in unique_classes:
        num_gt = 0 if object_gts is None else int(np.sum(object_gts['class_id'] == class_id))
        num_pred = 0 if predicted is None else int(np.sum(predicted['class_id'] == class_id))
        if predicted is None and object_gts is not None:
            ret = {
                'match': matching,
                'class_id': class_id,
                'iou_thresh': iou_thresh,
                'TP': 0,
                'FP': 0,
                'FN': num_gt,
                'num_gt': num_gt,
                'num_pred': num_pred,
            }
        elif predicted is not None and object_gts is None:
            ret = {
                'match': matching,
                'class_id': class_id,
                'iou_thresh': iou_thresh,
                'TP': 0,
                'FP': num_pred,
                'FN': 0,
                'num_gt': num_gt,
                'num_pred': num_pred,
            }
        elif predicted is None and object_gts is None:
            continue
        else:
            class_mask_pred = predicted['class_id'] == class_id
            (TP, FP, FN) = calculate_tfpn(
                predicted[class_mask_pred],
                object_gts[object_gts['class_id'] == class_id],
                iou_thresh=iou_thresh, matching=matching)
                # scores=predicted[class_mask_pred]['score'],
                # score_thresh=score_thresh)
            ret = {
                'match': matching,
                'class_id': class_id,
                'iou_thresh': iou_thresh,
                'TP': int(np.sum(TP)),
                'FP': int(np.sum(FP)),
                'FN': int(np.sum(FN)),
                'num_gt': num_gt,
                'num_pred': num_pred,
            }
        image_stats.append(ret)
    return image_stats

def matching_one_to_one(ious, iou_thresh):
    #, scores=None, score_thresh=None):
    '''Definitions (one-to-one):
        TP = number of detections with (IoU > 0.5)
        FP = number of detections with (IoU <= 0.5 or detected more than once)
        FN = number of objects that (not detected or detected with IoU<=0.5)'''
    assert ious.size > 0 and len(ious.shape) == 2, "matching_one_to_one got invalid IOUs matrix"
    TP = np.zeros(ious.shape[0], dtype=np.bool)
    FP = np.zeros(ious.shape[0], dtype=np.bool)
    FN = np.zeros(ious.shape[1], dtype=np.bool)
    # print(ious)
    invalid = ious <= iou_thresh
    # if score_thresh:
    #     invalid[scores <= score_thresh, :] = True
    ious[invalid] = 0
    # print(ious)
    pred_idx_ptr, gt_idx = linear_sum_assignment(1 - ious)
    pred_idx = np.arange(len(TP))[pred_idx_ptr]
    match_costs = ious[pred_idx, gt_idx]
    valid = match_costs > iou_thresh
    # if scores is not None:
    #     valid &= scores[pred_idx] > score_thresh
    pred_idx = pred_idx[valid]
    gt_idx = gt_idx[valid]
    # pred_idx_unmatched = np.array(list(set(range(ious.shape[0])) - set(pred_idx.tolist())))
    gt_idx_unmatched = np.array(list(set(range(ious.shape[1])) - set(gt_idx.tolist())))

    TP[pred_idx] = True
    FP |= ~TP
    if gt_idx_unmatched.size:
        FN[gt_idx_unmatched] = True
    # printTPFPFN(TP, FP, FN)
    return TP, FP, FN

def printTPFPFN(TP, FP, FN):
    print("TP" , TP)
    print("FP" , FP)
    print("FN" , FN)

def matching_many_to_one(ious, iou_thresh):
    #, scores=None, score_thresh=None):
    '''Definitions (many-to-one):
        TP = number of detections with IoU > iou_thresh
        FP = number of detections with IoU <= iou_thresh
        FN = number of objects with IoU <= iou_thresh'''
    assert ious.size > 0 and len(ious.shape) == 2, "matching_many_to_one got invalid IOUs matrix"
    ious[ious <= iou_thresh] = 0
    # print(ious)
    # if score_thresh:
    #     ious[scores <= score_thresh, :] = 0
    TP = np.sum(ious, axis=1) != 0
    FP = ~TP
    FN = np.zeros(ious.shape[1], dtype=np.bool)
    gt_idx = np.arange(len(FN))
    gt_idx_ptr = np.argmax(ious[TP, :], axis=1)
    gt_idx = gt_idx[gt_idx_ptr]
    gt_idx_unmatched = np.array(list(set(range(ious.shape[1])) - set(gt_idx.tolist())))
    if gt_idx_unmatched.size:
        FN[gt_idx_unmatched] = True
    # printTPFPFN(TP, FP, FN)
    return TP, FP, FN

def matching_many_to_many(ious, iou_thresh):
    # , scores=None, score_thresh=None):
    '''Definitions (many-to-many):
        TP = number of detections with IoU > iou_thresh
        FP = number of detections with IoU <= iou_thresh
        FN = number of objects with IoU <= iou_thresh'''
    assert ious.size > 0 and len(ious.shape) == 2, "matching_many_to_many got invalid IOUs matrix"
    ious[ious <= iou_thresh] = 0
    # print(ious)
    # if score_thresh:
    #     ious[scores <= score_thresh, :] = 0
    FP = np.zeros(ious.shape[0], dtype=np.bool)
    TP = np.zeros(ious.shape[0], dtype=np.bool)
    FN = np.ones(ious.shape[1], dtype=np.bool)
    conditions = ious > 0
    pred_idx, gt_idx = np.nonzero(conditions)
    TP[np.unique(pred_idx)] = True
    FN[np.unique(gt_idx)] = False
    FP = ~TP
    # printTPFPFN(TP, FP, FN)
    return TP, FP, FN
