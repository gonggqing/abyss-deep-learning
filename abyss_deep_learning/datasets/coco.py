# from abc import ABCMeta, abstractmethod
import json
from collections import Counter
from contextlib import redirect_stdout
from sys import stderr
import concurrent.futures
import itertools
import os
import random
import sys

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from abyss_deep_learning.base.datasets import DatasetTaskBase, DatasetTypeBase
from abyss_deep_learning.datasets.translators import AnnotationTranslator


######################## Abstract Classes with COCO data format ########################
class CocoInterface(object):
    @property
    def coco(self):
        return self._coco

    def __init__(self, coco, **kwargs):
        self._coco = coco


class CocoDataset(CocoInterface):
    """An dataset that fits the COCO JSON model."""

    def __init__(self, json_path_or_string, **kwargs):
        '''Base type for datasets using the COCO JSON data model.'''
        self.json_path = json_path_or_string
        self.image_dir = kwargs.get('image_dir', None)
        with redirect_stdout(stderr):
            if os.path.exists(self.json_path):
                self._coco = COCO(json_path_or_string)
            else:
                self._coco = COCO()
                try:
                    json_dict = json.loads(json_path_or_string)
                    if 'images' in json_dict and 'annotations' in json_dict:
                        self._coco.dataset = json_dict
                        self._coco.createIndex()
                    else:
                        raise SyntaxError("Invalid json data format - JSON not in COCO format")
                except json.decoder.JSONDecodeError:
                    raise SyntaxError("Invalid json data format - JSON malformed")


        CocoInterface.__init__(self, self.coco, **kwargs)
        self.data_ids = kwargs.pop('data_ids', [image['id'] for image in self.coco.imgs.values()])

    def split(self, ratios):
        with redirect_stdout(sys.stderr):
            ds = self._coco
            dss = []
            for r in ratios:
                dss.append(COCO())

        # sys.stdout = sys.__stdout__

        sample_nums = [r * len(ds.imgs) for r in ratios]

        shuffled_imgs = random.shuffle(ds.imgs)

        last_index = 0

        for i, sn in enumerate(sample_nums):
            dss[i].imgs = {img['id']: img for img in shuffled_imgs[last_index:last_index + sn]}
            last_index += sn

            im_ids = dss[i].getImgIds()
            ann_ids = dss[i].getAnnIds(imgIds=im_ids)

            dss[i].anns = ds.loadAnns(ids=ann_ids)

        return dss


########### COCO Dataset Types #################
def _noop(*args):
    return args if len(args) > 1 else args[0]


class ImageDatatype(CocoInterface, DatasetTypeBase):
    '''Do not rely on the presence of categories, as this implys it is an object dataset.
    The only precondition on this is that it involves image data.

    kwargs:
      * cached: (Boolean)
          Cache the data in memory instead of loading it every time
      * preprocess_data: (Callable)
          After loading the image run it through this translator function.
          The signature is lambda image: transform(image).
          Should be used for example, downsampling images prior to caching them.
      * image_dir: (string)
          The image dir to use if no path is given in a dataset image.
    '''

    def __init__(self, coco, **kwargs):
        CocoInterface.__init__(self, coco, **kwargs)
        self._data = dict()
        self._preprocess_data = kwargs.get('preprocess_data', _noop)
        self.dtype_image = np.uint8

        if kwargs.get('cached', False):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for image_id, image in zip(
                        self.data_ids,
                        executor.map(self.load_data, self.data_ids)):
                    self._data[image_id] = image

    def load_data(self, image_id, **kwargs):
        if image_id in self._data:
            return self._data[image_id]

        if 'path' in self.coco.imgs[image_id]:
            path = self.coco.imgs[image_id]['path']
        else:
            assert self.image_dir, "Dataset image dir must be set as no path is provided in database."
            path = os.path.join(self.image_dir, self.coco.imgs[image_id]['file_name'])

        return self._preprocess_data(imread(path, plugin='imageio'))

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
    def __init__(self, coco, translator=None, use_captions=False, **kwargs):
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
        self.translator = translator or AnnotationTranslator()
        assert isinstance(translator, (AnnotationTranslator, type(None)))
        # TODO - self.captions implemented like this doesn't allow full use of translators
        # TODO - Get captions from categories instead
        self.use_captions = use_captions
        if self.use_captions:
            self.captions = set(sorted([
                caption
                for annotation in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[]))
                if self.translator.filter(annotation)
                for caption in self.translator.translate(annotation)
            ]))
            self.num_classes = len(self.captions)
        else:  # Categories and captions are now interchangeable
            self.categories = set(sorted([
                caption
                for annotation in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[]))
                if self.translator.filter(annotation)
                for caption in self.translator.translate(annotation)
            ]))
            self.captions = self.categories
            self.num_classes = len(self.captions)
        self.stats = dict()
        self._targets = dict()

        self._preprocess_targets = kwargs.get('preprocess_targets', _noop)

        if kwargs.get('cached', False):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for data_id, targets in zip(
                        self.data_ids, executor.map(self.load_targets, self.data_ids)):
                    self._targets[data_id] = targets

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

    def __len__(self):
        """
        Denotes the number of images in the dataset
        Returns:
            (int): number of images in the dataset
        """
        return len(self.coco.loadImgs(self.coco.getImgIds()))

    def _calc_class_stats(self):
        if not self.stats:
            targets = [self.load_targets(data_id) for data_id in self.data_ids]
            unlabeled = sum([1 for target in targets if not target])
            self.stats['unlabeled'] = unlabeled / len(self.data_ids)
            targets = [
                caption
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
        if not self.stats:
            self._calc_class_stats()
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


class SemanticSegmentationTask(CocoInterface, DatasetTaskBase):
    def __init__(self, coco, translator=None, num_classes=None, **kwargs):
        """
        Segmentation arguments:
            coco (pycocotools.COCO): The COCO object to read the targes from
            translator (AnnotationTranslator, optional): An instance of an abyss_deep_learning.datasets.translators.AnnotationTranslator
            num_classes (int, optional): The number of classes to generate data for; if None then infer from coco.cats
            cached (bool, optional): Whether to cache the entire dataset into memory.
        """
        CocoInterface.__init__(self, coco, **kwargs)
        assert isinstance(translator, (AnnotationTranslator, type(None)))
        self.translator = translator or AnnotationTranslator()
        self.num_classes = num_classes if num_classes else len(self.coco.cats) + 1
        self.stats = dict()
        self._targets = dict()

        self._preprocess_targets = kwargs.get('preprocess_targets', _noop)

        if kwargs.get('cached', False):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for data_id, targets in zip(
                        self.data_ids, executor.map(self.load_targets, self.data_ids)):
                    self._targets[data_id] = targets
        # self._calc_class_stats()

    def load_targets(self, data_id, **kwargs):
        # assert np.issubdtype(type(data_id), np.integer), "Must pass exactly one ID"
        if data_id in self._targets:
            return self._targets[data_id]
        img = self.coco.loadImgs(ids=[data_id])[0]
        anns = [self.translator.translate(ann) for ann in self.coco.loadAnns(
            self.coco.getAnnIds([data_id])) if self.translator.filter(ann)]
        if anns:
            masks = np.array([self.coco.annToMask(ann) for ann in anns]).transpose((1, 2, 0))
            class_ids = np.array([ann['category_id'] for ann in anns])
            return self._preprocess_targets(
                SemanticSegmentationTask._pack_masks(
                    masks, class_ids, self.num_classes, dtype=self.dtype_image))
        masks = np.zeros((img['height'], img['width'], self.num_classes), dtype=self.dtype_image)
        masks[..., 0] = 1
        return self._preprocess_targets(masks)

    def _calc_class_stats(self):
        if not self.stats:
            self.stats = dict()
            class_count = dict()
            for data_id in self.data_ids:
                target = self.load_targets(data_id).argmax(-1)
                for key, val in Counter(target.ravel().tolist()).items():
                    class_count[key] = class_count.get(key, 0) + val

            self.stats['class_weights'] = np.array(
                [class_count.get(key, 0) for key in range(self.num_classes)], dtype=np.float64)
            self.stats['class_weights'] **= -1.0
            self.stats['class_weights'] /= self.stats['class_weights'].min()

    @staticmethod
    def _pack_masks(masks, mask_classes, num_classes, dtype=np.uint8):
        '''Pack a list of instance masks into a categorical mask.
        Expects masks to be shape [height, width, num_instances] and mask_classes to be [num_instances].'''
        num_shapes = len(mask_classes)
        shape = masks.shape
        packed = np.zeros(shape[0:2] + (num_classes,), dtype=dtype)
        packed[..., 0] = 1
        for i in range(num_shapes):
            class_id = mask_classes[i]
            mask = masks[..., i]
            packed[..., class_id] |= mask
            packed[..., 0] &= ~mask
        return packed

    @property
    def class_weights(self):
        '''Returns the class weights that will balance the backprop update over the class distribution.'''
        return self.stats['class_weights']

    def print_class_stats(self):
        '''Prints statistics about the class/image distribution.'''
        self._calc_class_stats()
        print("{:s} class stats {:s}".format('=' * 8, '=' * 8))
        print("class weights:")
        print(" ", self.class_weights)


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
        """
        kwargs -
        """
        ImageDatatype.__init__(self, self.coco, **kwargs)
        """
            kwargs -
        """
        ClassificationTask.__init__(self, self.coco, **kwargs)
        """
            kwargs -
        """

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


class ImageSemanticSegmentationDataset(CocoDataset, ImageDatatype, SemanticSegmentationTask):
    # TODO:
    #   *  Class statistics readout
    #   *  Support for computing class weights given current dataset config
    #   *  Support for forcing class balance by selecting IDs evenly
    #   *  Generator data order optimization
    #   *  Support for visualising data sample or prediction with same format
    def __init__(self, json_path, **kwargs):
        CocoDataset.__init__(self, json_path, **kwargs)
        ImageDatatype.__init__(self, self.coco, **kwargs)
        SemanticSegmentationTask.__init__(self, self.coco, **kwargs)

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
