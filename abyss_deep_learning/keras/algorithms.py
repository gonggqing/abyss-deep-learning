from collections import Counter

import numpy as np
from itertools import cycle
from skimage.color import rgb2gray
import keras.backend as K
from abyss_deep_learning.utils import tile_gen, detile
import matplotlib.pyplot as plt

#### Generic Algorithms

class LRSearch(object):
    def __init__(self, model, x, y=None, batch_size=None):
        self.model = model
        self.results = {'final_loss': dict(), 'improvement': dict(), 'history': dict(), 'epochs': dict()}
        self.inputs = x
        self.targets = y
        self.batch_size = batch_size
        if not model.built:
            raise ValueError("Model must be compiled first.")
        self.weights = self.model.get_weights()
        
    def fit(self, n_lrs=10, n_epochs=1, lr_power_range=(-5, -2), **kwargs):
        from types import GeneratorType
        from keras.callbacks import TerminateOnNaN
        kwargs['callbacks'] = [TerminateOnNaN()]
        for learning_rate in 10 ** np.random.uniform(lr_power_range[0], lr_power_range[1], n_lrs):
            print("Starting LR {:e}".format(learning_rate))
            self.model.reset_states()
            self.model.set_weights(self.weights)
            K.set_value(self.model.optimizer.lr, learning_rate)
            if isinstance(self.inputs, GeneratorType):
                result = self.model.fit_generator(
                    self.inputs, epochs=n_epochs, verbose=0, **kwargs)
            else:
                result = self.model.fit(
                    self.inputs, self.targets, batch_size=self.batch_size, epochs=n_epochs, verbose=0, **kwargs)
            self.results['final_loss'][float(learning_rate)] = result.history['loss'][-1]
            self.results['improvement'][float(learning_rate)] = result.history['loss'][-1] - result.history['loss'][0]
            self.results['history'][float(learning_rate)] = result.history['loss']
            self.results['epochs'][float(learning_rate)] = result.epoch
            
    
    def plot(self):
        if self.model.optimizer == 'categorical_crossentropy'
        rand_loss = -np.log(1 / y_gt.shape[1])

        x, y = list(self.results['final_loss'].keys()), list(self.results['final_loss'].values())
        plt.figure()
        fig, ax1 = plt.subplots()
        ax1.semilogx(x, 100 * (1 - y / rand_loss), 'x')

        ax1.grid(True)
        ax1.set_ylabel('final loss (% better than rand)')
        ax1.tick_params('y')
        ax1.set_xlabel("learning rate")
        ax1.set_ylim([-50, 100])


