'''Base classes for supervised and unsupervised datasets.
'''
from abc import ABCMeta, abstractmethod
from itertools import cycle
from random import shuffle

from pycocotools.coco import COCO
from skimage.color import gray2rgb
from skimage.io import imread
import numpy as np

############################# Abstract Base Classes #############################

class DatasetTypeBase(metaclass=ABCMeta):
    '''Must subclass methods:
        load_data
    '''
    @abstractmethod
    def load_data(self, data_id, **kwargs):
        pass

class DatasetTaskBase(metaclass=ABCMeta):
    '''Must subclass methods:
        load_targets
    '''
    @abstractmethod
    def load_targets(self, data_id, **kwargs):
        pass
    
# class DatasetBase(object):
#     '''A dataset consists of a DatasetType and a DatasetTask to define its format.'''
#     def __init__(self, dataset_type, dataset_task, **kwargs):
#         assert issubclass(dataset_type, DatasetTypeBase)
#         assert issubclass(dataset_task, DatasetTaskBase)
#         self.dataset_type = dataset_type
#         self.dataset_task = dataset_task
        
        
