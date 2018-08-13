# from abc import ABCMeta, abstractmethod
from contextlib import redirect_stdout
from sys import stderr
import concurrent.futures
import itertools
import random
from collections import Counter
        
from sklearn.utils.class_weight import compute_class_weight
from skimage.io import imread
from pycocotools.coco import COCO
import numpy as np

from abyss_deep_learning.datasets.base import DatasetTypeBase, DatasetTaskBase
from abyss_deep_learning.datasets.translators import AnnotationTranslator

######################## Abstract Classes with COCO data format ########################
class CocoInterface(object):
    @property
    def coco(self):
        return self._coco

    def __init__(self, coco, **kwargs):
        self._coco = coco

class CocoDataset(CocoInterface):
    '''An dataset that fits the COCO JSON model.'''
    def __init__(self, json_path, **kwargs):
        '''Base type for datasets using the COCO JSON data model.'''
        self.json_path = json_path
        with redirect_stdout(stderr):
            self._coco = COCO(json_path)
        CocoInterface.__init__(self, self.coco, **kwargs)
        self.data_ids = [image['id'] for image in self.coco.imgs.values()]



########### COCO Dataset Types #################
def _noop(*args):
    return args if len(args) > 1 else args[0]

class ImageDatatype(CocoInterface, DatasetTypeBase):
    '''Do not reply on the presence of categories, as this implys it is an object dataset.
    The only precondition on this is that it involves image data.

    kwargs:
      * cached: (Boolean)
          Cache the data in memory instead of loading it every time
      * preprocess_data: (Callable)
          After loading the image run it through this translator function.
          The signature is lambda image: transform(image).
          Should be used for example, downsampling images prior to caching them.
    '''
    def __init__(self, coco, **kwargs):
        CocoInterface.__init__(self, coco, **kwargs)
        self._data = dict()
        self._preprocess_data = kwargs.get('preprocess_data', _noop)

        if kwargs.get('cached', False):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for image_id, image in zip(
                        self.data_ids,
                        executor.map(self.load_data, self.data_ids)):
                    self._data[image_id] = image

    def load_data(self, image_id, **kwargs):
        if image_id in self._data:
            return self._data[image_id]
        return self._preprocess_data(
            imread(self.coco.imgs[image_id]['path']))

    def sample(self, data_id=None, **kwargs):
        if not data_id:
            data_id = random.choice(self.data_ids)
        return (self.load_data(data_id, **kwargs),)

    def generator(self, data_ids=None, shuffle_ids=False, endless=False, **kwargs):
        if not data_ids:
            data_ids = list(self.data_ids)
        if shuffle_ids:
            random.shuffle(data_ids)
        iterator = itertools.cycle if endless else iter
        for data_id in iterator(data_ids):
            yield self.load_data(data_id, **kwargs)

########### COCO Task Types #################

class ClassificationTask(CocoInterface, DatasetTaskBase):
    def __init__(self, coco, translator=None, **kwargs):
        '''Assumes that the data can be anything, but each data has 0 or more targets.

        kwargs:
          * cached: (Boolean)
              Cache the targets in memory instead of loading it every time
          * force_balance: (Boolean) #TODO
              Sample ids such that all classes are balanced to the smallest class.
          * translator: (Callable)
              After loading the data target run it through this translator function.
              Must be an instance of a subclass of abyss.datasets.translators.AnnotationTranslator.
              Should be used for example remapping captions.
        '''
        CocoInterface.__init__(self, coco, **kwargs)
        assert isinstance(translator, (AnnotationTranslator, type(None)))
        self.translator = translator or AnnotationTranslator()
        self.captions = set(sorted([
            caption
            for annotation in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[]))
            for caption in self.translator.translate(annotation)
            if self.translator.filter(annotation)]))
        self.num_classes = len(self.captions)
        self.stats = dict()
        self._targets = dict()

        self._preprocess_targets = kwargs.get('preprocess_targets', _noop)

        if kwargs.get('cached', False):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for data_id, targets in zip(
                        self.data_ids, executor.map(self.load_targets, self.data_ids)):
                    self._targets[data_id] = targets

        self._calc_class_stats()

    def load_caption(self, data_id, **kwargs):
        caps = [
            self.translator.translate(annotation)
            for annotation in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[data_id]))
            if self.translator.filter(annotation)]
        return set([i for f in caps for i in f])

    def load_targets(self, data_id, **kwargs):
        if data_id in self._targets:
            return self._targets[data_id]
        return self.load_caption(data_id, **kwargs)

    def _calc_class_stats(self):
        if not self.stats:
            targets = [self.load_targets(data_id) for data_id in self.data_ids]
            unlabeled = sum([1 for target in targets if not target])
            self.stats['unlabeled'] = unlabeled / len(self.data_ids)
            targets = [caption 
                for captions in targets
                for caption in captions]
            self.stats['images_per_class'] = dict(sorted(Counter(targets).items(), key=lambda x: x[0]))
            class_weights = compute_class_weight('balanced', list(self.captions), targets)
            class_weights = {i: float(np.round(v, 3)) for i, v in enumerate(class_weights)}
            self.stats['class_weights'] = class_weights
            a = np.array(list(class_weights.values()))
            self.stats['trivial_accuracy'] = np.mean(a / np.max(a))

    @property
    def class_weights(self):
        '''Returns the class weights that will balance the backprop update over the class distribution.'''
        return self.stats['class_weights']

    def print_class_stats(self):
        '''Prints statistics about the class/image distribution.'''
        self._calc_class_stats()
        print("{:s} class stats {:s}".format('=' * 8, '=' * 8))
        print("data count per class:")
        print(" ", self.stats['images_per_class'])
        print("class weights:")
        print(" ", self.class_weights)
        print("trivial result accuracy:\n  {:.2f} or {:.2f}".format(
            self.stats['trivial_accuracy'], 1 - self.stats['trivial_accuracy']))

#### COCO Realisations ########

class ImageClassificationDataset(CocoDataset, ImageDatatype, ClassificationTask):
    # TODO: 
    #   *  Class statistics readout
    #   *  Support for computing class weights given current dataset config
    #   *  Support for forcing class balance by selecting IDs evenly
    #   *  Generator data order optimization
    #   *  Support for visualising data sample or prediction with same format
    def __init__(self, json_path, **kwargs):
        CocoDataset.__init__(self, json_path, **kwargs)
        ImageDatatype.__init__(self, self.coco, **kwargs)
        ClassificationTask.__init__(self, self.coco, **kwargs)
        
    def sample(self, image_id=None, **kwargs):
        if not image_id:
            image_id = random.choice(self.data_ids)
        return (self.load_data(image_id, **kwargs), self.load_targets(image_id, **kwargs))
            
    def generator(self, data_ids=None, shuffle_ids=False, endless=False, **kwargs):
        if not data_ids:
            data_ids = list(self.data_ids)
        if shuffle_ids:
            random.shuffle(data_ids)
        iterator = itertools.cycle if endless else iter
        for data_id in iterator(data_ids):
            yield self.load_data(data_id, **kwargs), self.load_targets(data_id, **kwargs)

