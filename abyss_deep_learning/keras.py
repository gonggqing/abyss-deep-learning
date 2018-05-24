import numpy as np
from itertools import product
from numpy.lib.stride_tricks import as_strided as ast
from skimage.color import rgb2gray
import keras.backend as K


def remove_mean_gen(gen, mean):
    for inputs, targets in gen:
        yield inputs - mean, targets


def rgb2gray_gen(gen):
    for inputs, targets in gen:
        inputs_2 = np.zeros(inputs.shape[0:-1] + (1,))
        for i in range(inputs.shape[0]):
            inputs_2[i] = rgb2gray(inputs)
        yield inputs_2, targets


def binary_targets_gen(gen):
    for inputs, targets in gen:
        for i in range(targets.shape[0]):
            targets[i, ..., 0] = np.logical_or.reduce(
                targets[i, ..., 1:], axis=-1)
        yield inputs, targets[..., 0][..., np.newaxis]


def transpose_image_gen(gen):
    for inputs, targets in gen:
        yield np.transpose(inputs, (0, 3, 1, 2)), targets


def onehot_generator(gen):
    for inputs, targets in gen:
        yield inputs, targets.reshape((targets.shape[0], -1, targets.shape[-1]))


def resize_generator(gen, divide):
    for inputs, targets in gen:
        targets_shape = list(targets.shape)
        targets_shape[1] //= divide
        targets_shape[2] //= divide
        targets_small = np.ones(targets_shape, dtype=targets.dtype)
        for i in range(targets_shape[0]):
            targets_small[i, ...] = targets[i, ::divide, ::divide]
        yield inputs, targets_small


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def bilinear_upsample_weights(filter_size, number_of_classes):
    def upsample_filt(size):
        """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def initialize_conv_transpose2d(model, layer_names, trainable=True):
    for name in layer_names:
        layer = model.get_layer(name=name)
        v = layer.get_weights()
        if len(layer.weights) > 1:
            layer.set_weights(
                [bilinear_upsample_weights(v[0].shape[0], v[0].shape[2]), v[1]])
        else:
            layer.set_weights(
                [bilinear_upsample_weights(v[0].shape[0], v[0].shape[2])])
        layer.trainable = trainable
