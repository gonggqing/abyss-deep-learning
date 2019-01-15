'''Metrics for machine learning'''
import numpy as np
from scipy.optimize import linear_sum_assignment

def bbox_iou_matrix( a, b ):
    ''' Calculate MxN matrix of IOU for two sets of bounding boxes, e.g: iou( predictions, ground_truth )
    
    parameters
    ----------
    a: numpy.array
       first Mx4 array of box begin/end coordinates as y1,x1,y2,x2
    b: numpy.array
       second Mx4 array of box begin/end coordinates as y1,x1,y2,x2
    
    returns
    -------
    numpy.array, MxN matrix of IOU values
    
    todo
    ----
    refactor, if too slow
    '''
    ac = np.repeat( [ a ], len( b ), axis = 0 )
    ac = np.transpose( ac, axes = ( 2, 1, 0 ) )
    bc = np.repeat( [ b ], len( a ), axis = 0 )
    bc = np.transpose( bc, axes = ( 2, 0, 1 ) )
    minimums = np.minimum( ac, bc )
    maximums = np.maximum( ac, bc )
    intersections = np.maximum( minimums[3] - maximums[1], 0 ) * np.maximum( minimums[2] - maximums[0], 0 )
    ta = np.transpose( a )
    tb = np.transpose( b )
    aa = ( ta[3] - ta[1] ) * ( ta[2] - ta[0] )
    ba = ( tb[3] - tb[1] ) * ( tb[2] - tb[0] )
    a_areas = np.repeat( [ aa ], len( b ), axis = 0 )
    a_areas = np.transpose( a_areas )
    b_areas = np.repeat( [ ba ], len( a ), axis = 0 )
    return intersections / ( a_areas + b_areas - intersections )

def one_to_one( ious, iou_threshold = None, iou_matrix = bbox_iou_matrix ):
    ''' Match two sets of boxes one to one by IOU on given treshold
    
    parameters
    ----------
    ious: numpy.array, iou matrix as returned by e.g. by bbox_iou_matrix( a, b )
    
    returns
    -------
    numpy.array, MxN matrix of IOU values
    '''
    ious *= ( ious >= ( 0. if iou_threshold is None else iou_threshold ) )
    rows, cols = linear_sum_assignment( 1 - ious )
    result = np.zeros( ious.shape )
    result[rows, cols] = ious[rows, cols] * ( ious[rows, cols] > 0 )
    return result

def one_to_many( ious, iou_threshold = None ):
    ''' Match two sets of boxes one to many on max IOU by IOU on given treshold
    
    parameters
    ----------
    ious: numpy.array, iou matrix as returned by e.g. by bbox_iou_matrix( a, b )
    
    returns
    -------
    numpy.array, MxN matrix of IOU values
    '''
    ious *= ( ious >= ( 0. if iou_threshold is None else iou_threshold ) )
    flags = np.zeros( ious.shape )
    flags[ np.argmax( ious, axis = 0 ), [*range( ious.shape[1] )] ] = 1
    return ious * flags

def tp_fp_tn_fn( predictions, truth, threshold = None, match = one_to_one, iou_matrix = bbox_iou_matrix ):
    ''' Return TP, FP, TN, FN
    
    parameters
    ----------
    a: numpy.array
       for bounding boxes, first Mx4 array of box begin/end coordinates as y1,x1,y2,x2
    b: numpy.array
       for bounding boxes, second Mx4 array of box begin/end coordinates as y1,x1,y2,x2
    
    returns
    -------
    TP, FP, TN, FN
    TP: numpy.array of size M, indices of TP in a
    FP: numpy.array of size M, indices of FP in a
    TN: numpy.array of variable size, indices of TN, empty for bounding boxes, since it does not make sense
    FN: numpy.array of size N, indices of FN in b
    '''
    if len( predictions ) == 0: return [], [], [], truth
    ious = iou_matrix( predictions, truth )
    matched = ( match( ious, threshold ) > 0 ) * 1
    thresholded = ( ( ious > ( 0. if threshold is None else threshold ) ) > 0 ) * 1
    tpi = np.nonzero( matched )[0]
    fpi = np.nonzero( np.max( thresholded, axis = 1 ) == 0 )[0]
    tni = []
    fni = np.nonzero( np.max( thresholded, axis = 0 ) == 0 )[0]
    return tpi, fpi, tni, fni



#def _flat_np(df):
    #return df.apply(pd.Series).as_matrix()

#def result_to_series(result):
    #num_instances = len(result['rois'])
    #frame = []
    #for idx in range(num_instances):
        #frame.append({
            #'roi': result['rois'][idx, :],
            #'score': result['scores'][idx].astype(float) if 'scores' in result else 0.0,
            #'mask': result['masks'][..., idx],
            #'class_id': result['class_ids'][idx].astype(int)
        #})
    #if not frame:
        #return None
    #return pd.DataFrame(frame, columns=frame[0].keys())


#def compute_iou(box, boxes, box_area, boxes_area):
    #"""Calculates IoU of the given box with the array of the given boxes.
    #box: 1D vector [y1, x1, y2, x2]
    #boxes: [boxes_count, (y1, x1, y2, x2)]
    #box_area: float. the area of 'box'
    #boxes_area: array of length boxes_count.
    #Note: the areas are passed in rather than calculated here for
          #efficency. Calculate once in the caller to avoid duplicate work.
    
    #*Coppied from maskrcnn.utils
    #"""
    

    ## Calculate intersection areas
    #y1 = np.maximum(box[0], boxes[:, 0])
    #y2 = np.minimum(box[2], boxes[:, 2])
    #x1 = np.maximum(box[1], boxes[:, 1])
    #x2 = np.minimum(box[3], boxes[:, 3])
    #intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    #union = box_area + boxes_area[:] - intersection[:]
    #iou = intersection / union
    #return iou


#def compute_overlaps(boxes1, boxes2):
    #"""Computes IoU overlaps between two sets of boxes.
    #boxes1, boxes2: [N, (y1, x1, y2, x2)].
    #For better performance, pass the largest set first and the smaller second.
    
    #*Coppied from maskrcnn.utils
    #"""

    ## Areas of anchors and GT boxes
    #area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    #area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    ## Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    ## Each cell contains the IoU value.
    #overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    #for i in range(overlaps.shape[1]):
        #box2 = boxes2[i]
        #overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    #return overlaps


#def calculate_tfpn(predicted, object_gts, matching='one-to-one', iou_thresh=0.5):#, score_thresh=None):
    #'''Calculate True Positive, False Positive and False Negatives given two objects of the same class
        #inputs are pandas.DataFrame as returned by result_to_series
    #'''
    #def raise_(ex):
        #raise ex
    ## ious index is [pred_idx, gt_idx]
    
    #predicted = None if not predicted.size else _flat_np(predicted['roi'])
    #object_gts = None if not object_gts.size else _flat_np(object_gts['roi'])
    #if predicted is not None and object_gts is not None:
        #ious = compute_overlaps(predicted, object_gts)
        #TP, FP, FN = {
            #'one-to-one': matching_one_to_one,
            #'one-to-many': lambda _, __: raise_(NotImplementedError("one-to-many not yet implemented")),
            #'many-to-one': matching_many_to_one,
            #'many-to-many': matching_many_to_many
        #}[matching](ious, iou_thresh) #, scores=scores, score_thresh=score_thresh)
        ## scores = predicted['scores']
    #elif predicted is None and object_gts is not None:
        #TP, FP, FN = 0, 0, len(object_gts)
    #elif predicted is not None and object_gts is None:
        #TP, FP, FN = 0, len(predicted), 0
    #elif predicted is None and object_gts is None:
        #TP, FP, FN = 0, 0, 0
    #return TP, FP, FN

#def calc_image_stats(predicted, object_gts, matching, iou_thresh=0.5, score_thresh=None):
    #unique_classes = set()
    #if predicted is not None:
        #unique_classes |= set(np.unique(predicted['class_id']).tolist())
    #if object_gts is not None:
        #unique_classes |= set(np.unique(object_gts['class_id']).tolist())
    #image_stats = []
    #for class_id in unique_classes:
        #num_gt = 0 if object_gts is None else int(np.sum(object_gts['class_id'] == class_id))
        #num_pred = 0 if predicted is None else int(np.sum(predicted['class_id'] == class_id))
        #if predicted is None and object_gts is not None:
            #ret = {
                #'match': matching,
                #'class_id': class_id,
                #'iou_thresh': iou_thresh,
                #'TP': 0,
                #'FP': 0,
                #'FN': num_gt,
                #'num_gt': num_gt,
                #'num_pred': num_pred,
            #}
        #elif predicted is not None and object_gts is None:
            #ret = {
                #'match': matching,
                #'class_id': class_id,
                #'iou_thresh': iou_thresh,
                #'TP': 0,
                #'FP': num_pred,
                #'FN': 0,
                #'num_gt': num_gt,
                #'num_pred': num_pred,
            #}
        #elif predicted is None and object_gts is None:
            #continue
        #else:
            #class_mask_pred = predicted['class_id'] == class_id
            #(TP, FP, FN) = calculate_tfpn(
                #predicted[class_mask_pred],
                #object_gts[object_gts['class_id'] == class_id],
                #iou_thresh=iou_thresh, matching=matching)
                ## scores=predicted[class_mask_pred]['score'],
                ## score_thresh=score_thresh)
            #ret = {
                #'match': matching,
                #'class_id': class_id,
                #'iou_thresh': iou_thresh,
                #'TP': int(np.sum(TP)),
                #'FP': int(np.sum(FP)),
                #'FN': int(np.sum(FN)),
                #'num_gt': num_gt,
                #'num_pred': num_pred,
            #}
        #image_stats.append(ret)
    #return image_stats

#def matching_one_to_one(ious, iou_thresh):
    #'''Definitions (one-to-one):
        #TP = number of detections with (IoU > 0.5)
        #FP = number of detections with (IoU <= 0.5 or detected more than once)
        #FN = number of objects that (not detected or detected with IoU<=0.5)'''
    #assert ious.size > 0 and len(ious.shape) == 2, "matching_one_to_one got invalid IOUs matrix"
    #TP = np.zeros(ious.shape[0], dtype=np.bool)
    #FP = np.zeros(ious.shape[0], dtype=np.bool)
    #FN = np.zeros(ious.shape[1], dtype=np.bool)
    ## print(ious)
    #invalid = ious <= iou_thresh
    #ious[invalid] = 0
    ## print(ious)
    #pred_idx_ptr, gt_idx = linear_sum_assignment(1 - ious)
    #pred_idx = np.arange(len(TP))[pred_idx_ptr]
    #match_costs = ious[pred_idx, gt_idx]
    #valid = match_costs > iou_thresh
    #pred_idx = pred_idx[valid]
    #gt_idx = gt_idx[valid]
    ## pred_idx_unmatched = np.array(list(set(range(ious.shape[0])) - set(pred_idx.tolist())))
    #gt_idx_unmatched = np.array(list(set(range(ious.shape[1])) - set(gt_idx.tolist())))

    #TP[pred_idx] = True
    #FP |= ~TP
    #if gt_idx_unmatched.size:
        #FN[gt_idx_unmatched] = True
    ## printTPFPFN(TP, FP, FN)
    ## Y_true = 
    #return TP, FP, FN

#def printTPFPFN(TP, FP, FN):
    #print("TP" , TP)
    #print("FP" , FP)
    #print("FN" , FN)

#def matching_many_to_one(ious, iou_thresh):
    ##, scores=None, score_thresh=None):
    #'''Definitions (many-to-one):
        #TP = number of detections with IoU > iou_thresh
        #FP = number of detections with IoU <= iou_thresh
        #FN = number of objects with IoU <= iou_thresh'''
    #assert ious.size > 0 and len(ious.shape) == 2, "matching_many_to_one got invalid IOUs matrix"
    #ious[ious <= iou_thresh] = 0
    ## print(ious)
    ## if score_thresh:
    ##     ious[scores <= score_thresh, :] = 0
    #TP = np.sum(ious, axis=1) != 0
    #FP = ~TP
    #FN = np.zeros(ious.shape[1], dtype=np.bool)
    #gt_idx = np.arange(len(FN))
    #gt_idx_ptr = np.argmax(ious[TP, :], axis=1)
    #gt_idx = gt_idx[gt_idx_ptr]
    #gt_idx_unmatched = np.array(list(set(range(ious.shape[1])) - set(gt_idx.tolist())))
    #if gt_idx_unmatched.size:
        #FN[gt_idx_unmatched] = True
    ## printTPFPFN(TP, FP, FN)
    #return TP, FP, FN

#def matching_many_to_many(ious, iou_thresh):
    ## , scores=None, score_thresh=None):
    #'''Definitions (many-to-many):
        #TP = number of detections with IoU > iou_thresh
        #FP = number of detections with IoU <= iou_thresh
        #FN = number of objects with IoU <= iou_thresh'''
    #assert ious.size > 0 and len(ious.shape) == 2, "matching_many_to_many got invalid IOUs matrix"
    #ious[ious <= iou_thresh] = 0
    ## print(ious)
    ## if score_thresh:
    ##     ious[scores <= score_thresh, :] = 0
    #FP = np.zeros(ious.shape[0], dtype=np.bool)
    #TP = np.zeros(ious.shape[0], dtype=np.bool)
    #FN = np.ones(ious.shape[1], dtype=np.bool)
    #conditions = ious > 0
    #pred_idx, gt_idx = np.nonzero(conditions)
    #TP[np.unique(pred_idx)] = True
    #FN[np.unique(gt_idx)] = False
    #FP = ~TP
    ## printTPFPFN(TP, FP, FN)
    #return TP, FP, FN
