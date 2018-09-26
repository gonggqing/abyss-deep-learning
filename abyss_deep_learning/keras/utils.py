from collections import Counter

import numpy as np
from itertools import cycle
# from numpy.lib.stride_tricks import as_strided as ast
from skimage.color import rgb2gray
import keras.backend as K

######### Model utilities #########

def reset_weights(model):
    '''Reinitialize all weights in a Model'''
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def bilinear_upsample_weights(filter_size):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
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
    weights = np.zeros(filter_size, dtype=np.float32)
    weights[:] = upsample_filt(filter_size[0])[..., np.newaxis, np.newaxis]
    return weights

def initialize_conv_transpose2d(model, layer_names, trainable=True):
    """Initialize Conv_Transpose2D layers given by name in a model.
    The weights are initialised to a noisy bilinear upsampling."""
    session = K.get_session()
    for name in layer_names:
        layer = model.get_layer(name=name)
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        # if hasattr(layer, 'bias_initializer')
        #     layer.bias.initializer.run(session=session)
        layer.trainable = trainable
        weights = layer.weights
        values = layer.get_weights()
        for i, (weight, value) in enumerate(zip(weights, values)):
            if 'kernel' in weight.name.lower():
                values[i] += bilinear_upsample_weights(value.shape)
        layer.set_weights(values)


######### Common generators #########


def lambda_gen(gen, func):
    for data in gen:
        yield func(*data)

def rgb2gray_gen(gen):
    """Convert RGB image to grayscale"""
    for inputs, targets in gen:
        yield rgb2gray(inputs), targets

def swap_channels_gen(gen, reverse=False):
    """
    Swaps image dim order from [channels, height, width] to [height, width, channel].
    Use reverse=True to go backwards.
    """
    for inputs, targets in gen:
        if reverse:
            yield np.transpose(inputs, (2, 0, 1)), targets
        else:
            yield np.transpose(inputs, (1, 2, 0)), targets

def array_gen(X, Y, batch_size=1):
    it = zip(cycle(X), cycle(Y))
    batch = []
    for x, y in it:
        batch.append((x, y))
        if len(batch) >= batch_size:
            yield tuple(map(np.array, tuple(map(tuple, zip(*batch)))))
            batch = []

def preprocess_gen(gen, preprocess_fn, input_shape):
    '''Runs the function preprocess_fn(input, target)'''
    for rgb, masks in gen:
        yield (
            preprocess_fn(rgb, input_shape, mask=False),
            preprocess_fn(masks, input_shape[0:2] + (masks.shape[-1],), mask=True))

       
def head_gen(gen, first=1):
    '''stop generator after N records.'''
    for i, item in enumerate(gen):
        assert isinstance(item, tuple), "head_gen inputs a tuple, even if it is size (1,)"
        yield item
        if i + 1 == first:
            return

def batching_gen(gen, batch_size=1):
    '''Read an unbatched generator and batch it to the given size.
       Using batch_size=0 will perform unbatching.'''
    if not batch_size:
        for item in gen: # item is a tuple
            batch_size = item[0].shape[0]
            for batch_idx in range(batch_size):
                yield tuple(field[batch_idx, ...] for field in item)
    else:
        num_items = None
        for row in gen: # Must be tuple or list
            assert isinstance(row, tuple), "batching_gen inputs a tuple, even if it is size (1,)"
            if not num_items:
                num_items = len(row)
                batches = [[] for i in range(num_items)]
            for i, item in enumerate(row):
                batches[i].append(item)
            if len(batches[0]) >= batch_size:
                yield tuple(np.array(item) for item in batches)
                batches = [[] for i in range(num_items)]

def gen_dump_data(gen, num_images):
    data = [[], []]
    for i, (image, caption) in enumerate(gen):
        if i >= num_images:
            break
        data[0].append(image)
        data[1].append(caption)
    data = (
        np.concatenate([i[np.newaxis, ...] for i in data[0]], axis=0),
        np.concatenate([np.asarray(i)[np.newaxis, ...] for i in data[1]], axis=0)
    )
    return data

def count_labels_single(data):
    return Counter([int(j) for i in data[1] for j in np.argwhere(i)])

def count_labels_multi(data):
    values = np.sum(data[1], axis=0).tolist()
    keys = np.arange(len(values))
    return dict(zip(keys, values))

def calc_class_weights(data, caption_type='single'):
    '''Calculate the class weights required to get a balanced optimizer.
    Data is a keras-generator-style tuple (inputs[], targets[]).
    To use a generator use data=gen_dump_data(gen, num_images).'''
    count_function = count_labels_single if caption_type == "single" else count_labels_multi
    counts = count_function(data)
    class_weights =  np.array([j for i, j in sorted(counts.items(), key=lambda x: x[0])], dtype=np.float64)
    class_weights = np.max(class_weights) / class_weights
    class_weights = dict(zip(sorted(counts.keys()), class_weights.tolist()))
    return class_weights

#### Generic Algorithms

class LRSearch(object):
    def __init__(self, model, x, y=None, batch_size=None):
        self.model = model
        self.results = dict()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        if not model.built:
            raise ValueError("Model must be compiled first.")
        self.weights = self.model.get_weights()
        
    def fit(self, n_lrs=10, n_epochs=1, lr_power_range=(-5, -2)):
        from types import GeneratorType
        from keras.callbacks import TerminateOnNaN
        
        common = dict(callbacks=[TerminateOnNaN()])
        for lr in 10 ** np.random.uniform(lr_power_range[0], lr_power_range[1], n_lrs):
            self.model.reset_states()
            self.model.set_weights(self.weights)
            if isinstance(self.x, GeneratorType):
                result = self.model.fit_generator(self.x, self.y, epochs=n_epochs, **common)
            else:
                result = self.model.fit(self.x, self.y, batch_size=self.batch_size, epochs=n_epochs, **common)
            self.results[float(lr)] = result.history['loss'][-1]
    
    def plot(self):
        x, y = list(self.results.keys()), list(self.results.values())
        plt.figure()
        plt.semilogx(x, y, '.')
        