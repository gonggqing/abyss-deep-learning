"""Class and support functions for an improved keras tensorboard callback.
"""
# import os
# from tensorflow.contrib.tensorboard.plugins import projector
from keras.backend import tf
from keras.callbacks import TensorBoard, Callback
from keras.models import Model
from tensorboard import summary
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.pr_curve import summary as pr_summary
import keras.backend as K

class ImprovedTensorBoard(TensorBoard):

    """An improved keras tensorboard callback including easy interface support for
    adding scalar summaries and the Custom Scalars, PR Curve and Embeddings plugins.
    """
    
    def __init__(self, scalars=None, groups=None, pr_curve=None, num_classes=None, **kwargs):
        """Constructor
        
        Args:
            scalars:
                A dict mapping strings to tensors.
                These tensors will be evaluated and show up as a scalar summary.
            groups:
                A dict that defines groups of scalars and the op names that they group.
                Accepts regex for op names.
                Example: {'category A': {'chart A1': ['op_name_1', r'.*acc.*']}}
            pr_curve:
                Evaluate the precision-recall curve.
            num_classes:
                The number of classes (dimension 1 of the data).
            log_dir: the path of the directory where to save the log
                files to be parsed by TensorBoard.
            histogram_freq: frequency (in epochs) at which to compute activation
                and weight histograms for the layers of the model. If set to 0,
                histograms won't be computed. Validation data (or split) must be
                specified for histogram visualizations.
            write_graph: whether to visualize the graph in TensorBoard.
                The log file can become quite large when
                write_graph is set to True.
            write_grads: whether to visualize gradient histograms in TensorBoard.
                `histogram_freq` must be greater than 0.
            batch_size: size of batch of inputs to feed to the network
                for histograms computation.
            write_images: whether to write model weights to visualize as
                image in TensorBoard.
            embeddings_freq: frequency (in epochs) at which selected embedding
                layers will be saved. If set to 0, embeddings won't be computed.
                Data to be visualized in TensorBoard's Embedding tab must be passed
                as `embeddings_data`.
            embeddings_layer_names: a list of names of layers to keep eye on. If
                None or empty list all the embedding layer will be watched.
            embeddings_metadata: a dictionary which maps layer name to a file name
                in which metadata for this embedding layer is saved. See the
                [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
                about metadata files format. In case if the same metadata file is
                used for all embedding layers, string can be passed.
            embeddings_data: data to be embedded at layers specified in
                `embeddings_layer_names`. Numpy array (if the model has a single
                input) or list of Numpy arrays (if the model has multiple inputs).
                Learn [more about embeddings](https://www.tensorflow.org/programmers_guide/embedding)
            **kwargs:
                All keyword arguments are passed to the standard TensorBoard plugin.
        """
        super().__init__(**kwargs)

        if not isinstance(scalars, (dict, type(None))):
            raise ValueError("scalars must be a dict mapping Strings to Tensors")
        self.scalars = scalars

        if not isinstance(groups, (dict, type(None))):
            raise ValueError("groups must be a dict like {'category A': {'chart A1': ['op_name_1', 'op_name_2']}}")
        self.groups = groups

        if pr_curve and num_classes is None:
            raise ValueError("pr_curve requires num_classes to be set.")

        self.pr_curve = pr_curve
        self.pr_summary = []
        self.num_classes = num_classes

        self.layout_summary = None
        if groups:
            categories = []
            for category_name, chart in groups.items():
                chart_list = []
                for chart_name, op_list in chart.items():
                    chart_list.append(
                        layout_pb2.Chart(
                            title=chart_name,
                            multiline=layout_pb2.MultilineChartContent(tag=op_list)))
                categories.append(
                    layout_pb2.Category(title=category_name, chart=chart_list))
            self.layout_summary = summary.custom_scalar_pb(
                layout_pb2.Layout(category=categories))


    def set_model(self, model):
        super().set_model(model)
        if self.layout_summary:
            self.writer.add_summary(self.layout_summary)

        if self.pr_curve:
            for class_idx in range(self.num_classes):
                predictions = self.model._feed_outputs[0][..., class_idx]
                labels = tf.cast(self.model._feed_targets[0][...,  class_idx], tf.bool)
                summary_op = pr_summary.op(
                    name='pr_curve_class_' + str(class_idx),
                    predictions=predictions,
                    labels=labels,
                    display_name='Precision-Recall (Class ' + str(class_idx) + ')')
                self.pr_summary.append(summary_op)
        self.merged = tf.summary.merge_all()

    def on_epoch_end(self, epoch, logs=None):
        if self.scalars:
            for name, value in self.scalars.items():
                try:
                    logs[name] = K.eval(value)
                except tf.errors.InvalidArgumentError:
                    if not self.validation_data:
                        raise ValueError("You must supply ImprovedTensorBoard with validation_data.")
                    tensors = (self.model.inputs + self.model.targets + self.model.sample_weights)
                    feed_dict = dict(zip(tensors, self.validation_data))
                    logs[name] = K.get_session().run([value], feed_dict=feed_dict)[0]

        if self.pr_curve and self.validation_data:
            tensors = self.model._feed_targets + self.model._feed_outputs
            predictions = self.model.predict(self.validation_data[0])
            feed_dict = dict(zip(tensors, [self.validation_data[1], predictions]))
            results = self.sess.run(self.pr_summary, feed_dict=feed_dict)
        
        super().on_epoch_end(epoch, logs)


def procuce_embeddings_tsv(path, headers, labels):
    """Produce the tab separated values required for visualizing the classes on the embeddings.
    
    Args:
        path (str):
            The path to save the tsv to. Should match the embeddings_metadata argument to TensorBoard.
        headers (dict of strings): The header to write to the tsv. Ignored if only 1 column in labels.
        labels (np.ndarray): The labels to write. Number of columns has to match headers.
    """
    assert labels.shape[1] == len(headers), \
        "labels.shape[1] is {:d} but len(headers) is {:d}".format(labels.shape[1], len(headers))
    with open(path, 'w') as file:
        if labels.shape[1] > 1:
            file.write("\t".join(headers) + "\n")
        for label in labels:
            file.write("\t".join([str(i) for i in label]) + "\n")



def kernel_sparsity(model, min_value=1e-6):
    '''Returns a tensor with the fraction of approximately zero valued weights in the model.
    
    Args:
        model (keras.Model): The model to monitor.
        min_value (float, optional): The smallest value at which to consider a weight as being sparse.
    
    Returns:
        tf.Tensor: A tensor containing the approximate kernel sparsity of the model.
    '''
    num = tf.zeros(1)
    den = tf.zeros(1)
    for weight in model.trainable_weights:
        if 'kernel' not in weight.name:
            continue
        size = tf.cast(tf.size(weight), tf.float32)
        zeros = size - tf.cast(tf.count_nonzero(tf.greater(weight, min_value)), tf.float32)
        num += zeros
        den += size
    return num / den

def avg_update_ratio(model, weight):
    '''Returns the average update-to-weight ratio for the given weight and model.
    This should be in the realm of 1e-3 for a healthy network.
    
    Args:
        model (keras.Model): The model the weight belongs to.
        weight (tf.Tensor): The weight tensor to monitor.
    
    Returns:
        tf.Tensor: A tensor containing the average update to weight ratio.
    '''
    grads = model.optimizer.get_gradients(model.total_loss, [weight])[0]
    return tf.norm(grads) * model.optimizer.lr / tf.norm(weight)
