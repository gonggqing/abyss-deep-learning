import numpy as np
from itertools import product
from numpy.lib.stride_tricks import as_strided as ast
from skimage.color import rgb2gray


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
