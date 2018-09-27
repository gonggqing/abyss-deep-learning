import numpy as np
from abyss_deep_learning.base.datasets import DatasetTaskBase, DatasetTypeBase
from abyss_deep_learning.keras.utils import gen_dump_data

class CachedGenClassificationDataset(DatasetTypeBase, DatasetTaskBase):
    '''Dumps data from a generator and caches it in a Dataset object suitable for image classification.'''
    def __init__(self, gen, n_samples, **kwargs):
        self.x_, self.y_ = gen_dump_data(gen, n_samples)
        self.data_ids = np.arange(n_samples)
        
    def load_data(self, data_id, **kwargs):
        return self.x_[data_id]
        
    def load_targets(self, data_id, **kwargs):
        return self.y_[data_id]

    def sample(self, image_id=None, **kwargs):
        if not image_id:
            image_id = np.random.choice(self.data_ids)
        return (self.load_data(image_id, **kwargs), self.load_targets(image_id, **kwargs))

    def generator(self, data_ids=None, shuffle_ids=False, endless=False, **kwargs):
        from itertools import cycle
        from random import shuffle
        if not data_ids:
            data_ids = list(self.data_ids)
        if shuffle_ids:
            shuffle(data_ids)
        iterator = cycle if endless else iter
        for data_id in iterator(data_ids):
            yield self.load_data(data_id, **kwargs), self.load_targets(data_id, **kwargs)
