import tensorflow as tf


def auc_factory(auc_type, weights=None):
    '''Returns the area under the curve for either ROC or PR curve.
       auc_type: (string) Either 'roc' or 'pr'.
       weights is a class weight vector broadcastable into the labels.'''
    metric_name = 'auc_{:s}'.format(auc_type)

    def metric(y_true, y_pred):
        # any tensorflow metric

        value, update_op = tf.metrics.auc(
            y_true, y_pred, weights,
            curve=auc_type,
            name=metric_name,
            summation_method='careful_interpolation')

        # find all variables created for this metric
        metric_vars = [
            i for i in tf.local_variables()
            if metric_name in i.name.split('/')[1]]

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value

    metric.__name__ = metric_name
    return metric


def mpca_factory(num_classes, weights=None):
    '''Returns the mean per-class average accuracy.
       num_classes: (int) number of classes in labels.
       weights is a class weight vector broadcastable into the labels.'''
    metric_name = 'mpca'

    def metric(y_true, y_pred):
        value, update_op = tf.metrics.mean_per_class_accuracy(
            y_true, y_pred, num_classes, weights=weights)
        # find all variables created for this metric
        metric_vars = [i for i in tf.local_variables(
        ) if metric_name in i.name.split('/')[1]]
        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
        # force to update metric values
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value

    metric.__name__ = metric_name
    return metric
