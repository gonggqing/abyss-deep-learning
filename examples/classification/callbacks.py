import numpy as np
import keras
import keras.backend as K
import keras.callbacks
import os
import tensorflow as tf
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, Callback, TerminateOnNaN, LearningRateScheduler
import copy
import math

from sklearn.metrics import precision_score, recall_score, f1_score
"""
These callbacks are designed to be used with the ImageClassifier class
"""


class DEPPrecisionRecallF1Callback(keras.callbacks.Callback):
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


class PrecisionRecallF1Callback(keras.callbacks.Callback):
    def __init__(self, generator, val_steps):
        super(PrecisionRecallF1Callback, self).__init__()
        self.generator = generator
        self.batch_size = 1
        self.val_steps = val_steps

    def on_epoch_end(self, epoch, logs=None):
        y_preds = []
        y_true = []
        gen = self.generator
        for batch, (val_x, val_y) in enumerate(gen):
            if batch >= self.val_steps:
                break
            preds = np.squeeze(self.model.predict(val_x),axis=0)  # Predict - relies on batch_size=1
            preds = preds.round().astype(int)

            y_preds.append(preds)
            y_true.append(np.squeeze(val_y, axis=0).round().astype(int))

        if len(y_preds) != len(y_true) or len(y_preds) == 0 or len(y_true) == 0:
            return
        y_preds = np.asarray(y_preds)
        y_true = np.asarray(y_true)

        precision = precision_score(y_true, y_preds, average='micro')  # What average method to use???
        recall = recall_score(y_true, y_preds, average='micro')
        f1 = f1_score(y_true, y_preds, average='micro')

        logs['precision'] = precision
        logs['recall'] = recall
        logs['f1'] = f1

        del gen

        return



class SaveModelCallback(Callback):
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
    def on_train_end(self, epoch):
        self.save_fn(os.path.join(self.model_dir, 'best_model.h5'))


class TrainValTensorBoard(TensorBoard):
    '''
    Additional and improved logging parameters with tensorboard. Wraps around tensorboard callback.
    adapted from:
    https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
    '''
    def __init__(self, log_dir='./logs', update_freq='epoch', update_step=1, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        if update_freq not in ['epoch', 'batch']:
            raise ValueError("TrainValTensorBoard callback update frequency is invalid. "
                             "Select either: update_freq = epoch or batch.")
        self.update_freq = update_freq
        self.update_step = update_step
        self.losses = [] # this is here accumulate metrics every batch, rather than the default on_epoch_end
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)
    def handle_validation(self, count, logs=None):
        '''
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        takes the logs, and processing any validation ones so they will be plotted on the same tensorboard graph as
        their training counterparts. Any remaining, no processed logs are returned.

        '''
        logs = logs or {}
        if self.update_freq is 'batch':
            val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, count)
            self.val_writer.flush()
        return {k: v for k, v in logs.items() if not k.startswith('val_')}
    def on_epoch_end(self, epoch, logs=None):
            logs = self.handle_validation(epoch, logs)
            logs.update({'learning_rate': K.eval(self.model.optimizer.lr)}) # additioanlly log learning rate
            super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)
    def on_batch_end(self, batch, logs=None):
        if batch % self.update_step is 0 or self.update_freq is 'epoch':
            logs = self.handle_validation(batch, logs)
            # Pass the remaining logs to `TensorBoard.on_epoch_end`
            super(TrainValTensorBoard, self).on_epoch_end(epoch=batch, logs=logs)
    def on_train_end(self, logs=None):
        self.val_writer.close()



class TrainsCallback(keras.callbacks.Callback):
    """
    Publishes scalar logs to the Trains server
    """
    def __init__(self, logger):
        """
        Initialises the callback
        Args:
            logger: (Task Logger). The logger to report scalars to. Initialise with task.get_logger() where task is the trains.Task class.
        """
        super(TrainsCallback, self).__init__()
        self.logger = logger
        self.epoch_count = 0

    def on_batch_end(self, batch, logs=None):
        """
        Publishes batchwise information

        Args:
            batch: (int) the current batch
            logs: (dict) the logs

        Returns:
            None
        """
        for k,v in logs.items():
            self.logger.report_scalar("batch_" + k, "series A", iteration=self.epoch_count+batch, value=v)

    def on_epoch_end(self, epoch, logs=None):
        """
        Publishes information at the end of each epoch
        Args:
            epoch: (int) the epoch
            logs: (dict) the logs

        Returns:
            None
        """
        for k,v in logs.items():
            self.logger.report_scalar(k, "series A", iteration=epoch, value=v)
        self.epoch_count += 1


def create_lr_schedule_callback(lr_schedule, lr, lr_schedule_params=None):
    """
    Creates the lr schedule callback, given a lr_schedule type and the parameters.

    Args:
        lr_schedule: (str) The type of learning rate schedule to use. Options are {step,exp,plateau,cyclic}
        lr: (float) The base learning rate
        lr_schedule_params: (dict) The parameters to initialise the models.
    Returns:
        (keras.callbacks.Callback) A callback that alters the learning rate.

    """
    if lr_schedule == "step":
        if lr_schedule_params is None:
            lr_schedule_params = {
                'drop': 0.1,
                'steps': 10
            }
        def step_decay(epoch):
            initial_lrate = lr
            drop = lr_schedule_params['drop']
            epochs_drop = lr_schedule_params['steps']
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate
        lr_schedule_callback = LearningRateScheduler(schedule=step_decay, verbose=True)

    elif lr_schedule == "exp":  # This is per epoch, not batch!
        if lr_schedule_params is None:
            lr_schedule_params = {
                'rate': 0.95
            }
        def exp_decay(epoch, curr_lr):
            new_lr = curr_lr * lr_schedule_params['rate']
            return new_lr
        lr_schedule_callback = LearningRateScheduler(schedule=exp_decay, verbose=True) # This is per epoch, not batch!

    elif lr_schedule == "cyclic":
        from keras_contrib.callbacks.cyclical_learning_rate import CyclicLR
        if lr_schedule_params is None:
            lr_schedule_params = {
                'base_lr': lr,
                'max_lr': lr*6
            }
        else:
            lr_schedule_params['base_lr'] = lr
        lr_schedule_callback = CyclicLR(**lr_schedule_params)
    elif lr_schedule == "plateau":
        if lr_schedule_params is None:
            lr_schedule_params = {}
        lr_schedule_callback = ReduceLROnPlateau(**lr_schedule_params)
    else:
        raise NotImplementedError("LR Schedule type %s not implemented")
    return lr_schedule_callback
