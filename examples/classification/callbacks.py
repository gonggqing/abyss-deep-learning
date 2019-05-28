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
