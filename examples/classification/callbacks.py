import numpy as np
import keras
import keras.backend as K
import keras.callbacks
import os
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score
"""
These callbacks are designed to be used with the ImageClassifier class
"""
class SaveModelCallback(keras.callbacks.Callback):
    """
    Saves the model at the end of an epoch. To be used with an ImageClassifier class.

    Usage:
        callbacks = [SaveModelCallback(classifier.save, 'models')]
    """
    def __init__(self, save_fn, model_dir, save_interval=1):
        self.model_dir = model_dir
        self.save_fn = save_fn
        self.save_interval = save_interval
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.save_interval == 0:
            self.save_fn(os.path.join(self.model_dir, 'model_%d.h5'%epoch))
        return
class PrecisionRecallF1Callback(keras.callbacks.Callback):
    """
    Calculates Precision, Recall and F1 Score
    """
    def __init__(self, validation_data):
        """
        Initialises the validation data
        Args:
            validation_data: ( tuple(np.ndarray, np.ndarray) ) Currently has to be a tuple of x,y. # TODO implement generator method
        """
        # if isinstance(validation_data, tuple) and isinstance(validation_data[0], np.ndarray) and isinstance(validation_data[1], np.ndarray):
        #     raise ValueError("validation data needs to be a (np.ndarray, np.ndarray) tuple - cannot work with generators yet")
        self.validation_data = validation_data
    def on_epoch_end(self, epoch, logs={}):
        """
        On the end of every epoch, calculate Precision, Recall, F1 score
        """
        predictions = self.model._feed_outputs[0]
        labels = tf.cast(self.model._feed_targets[0], tf.bool)
        _, precision = tf.metrics.precision(labels, predictions)
        _, recall = tf.metrics.recall(labels, predictions)

        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_pred = y_pred.round().astype(int)
        y_true = self.validation_data[1].astype(int)

        precision = precision_score(y_true, y_pred, average='micro')  # What average method to use???
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')

        logs['precision'] = precision
        logs['recall'] = recall
        logs['f1'] = f1
        return
class PrecisionRecallF1Callback_ERIC(keras.callbacks.Callback):
    '''
    compute and print out the f1 score, recall, and precision at the end of each epoch, using the whole validation data.
    Source and discussion here:
    https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    '''
    def __init__(self, generator, labels):
        self.data_in = generator
        self.labels = labels
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.rint(self.model.predict_generator(self.data_in, steps=len(self.labels))).argmax(axis=-1)
        val_targ = self.labels

        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return

from keras.callbacks import TensorBoard
from tensorboard.plugins.pr_curve import summary as pr_summary

# Check complete example in:
# https://github.com/akionakamura/pr-tensorboard-keras-example
class PRTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        # One extra argument to indicate whether or not to use the PR curve summary.
        self.pr_curve = kwargs.pop('pr_curve', True)
        super(PRTensorBoard, self).__init__(*args, **kwargs)

        global tf
        import tensorflow as tf

    def set_model(self, model):
        super(PRTensorBoard, self).set_model(model)

        if self.pr_curve:
            # Get the prediction and label tensor placeholders.
            predictions = self.model._feed_outputs[0]
            labels = tf.cast(self.model._feed_targets[0], tf.bool)
            # Create the PR summary OP.
            self.pr_summary = pr_summary.op(tag='pr_curve',
                                            predictions=predictions,
                                            labels=labels,
                                            display_name='Precision-Recall Curve')

    def on_epoch_end(self, epoch, logs=None):
        super(PRTensorBoard, self).on_epoch_end(epoch, logs)

        if self.pr_curve and self.validation_data:
            # Get the tensors again.
            tensors = self.model._feed_targets + self.model._feed_outputs
            # Predict the output.
            predictions = self.model.predict(self.validation_data[:-2])
            # Build the dictionary mapping the tensor to the data.
            val_data = [self.validation_data[-2], predictions]
            feed_dict = dict(zip(tensors, val_data))
            # Run and add summary.
            result = self.sess.run([self.pr_summary], feed_dict=feed_dict)
            self.writer.add_summary(result[0], epoch)
        self.writer.flush()