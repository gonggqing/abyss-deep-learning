# from abc import ABCMeta, abstractmethod
from collections import Counter
from contextlib import redirect_stdout
from sys import stderr
import concurrent.futures
import itertools
import os
import random

from mrcnn.utils import Dataset as MatterportMrcnnDataset
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
    '''An dataset that fits the COCO JSON model.'''
    def __init__(self, json_path, **kwargs):
        '''Base type for datasets using the COCO JSON data model.'''
        self.json_path = json_path
        self.image_dir = kwargs.get('image_dir', None)
        with redirect_stdout(stderr):
            self._coco = COCO(json_path)
        CocoInterface.__init__(self, self.coco, **kwargs)
        self.data_ids = kwargs.pop('data_ids', [image['id'] for image in self.coco.imgs.values()])

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

        return self._preprocess_data(imread(path, plugin='imread'))

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
            if self.translator.filter(annotation)
            for caption in self.translator.translate(annotation)
            ]))
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
    def __init__(self, coco, translator=None, **kwargs):
        CocoInterface.__init__(self, coco, **kwargs)
        assert isinstance(translator, (AnnotationTranslator, type(None)))
        self.translator = translator or AnnotationTranslator()
        self.num_classes = len(self.coco.cats) + 1
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

class MaskRcnnInstSegDataset(CocoDataset, ImageDatatype, MatterportMrcnnDataset):
    '''
    NOTE: This dataset scales between +/- 127.5 rather than +/- 1.0.
    '''
    def __init__(self, json_path, config, **kwargs):
        import importlib
        from bidict import bidict
        CocoDataset.__init__(self, json_path, **kwargs)
        ImageDatatype.__init__(self, self.coco, **kwargs)
        MatterportMrcnnDataset.__init__(self, class_map=None)
        self.load_coco(image_dir=self.image_dir)
        self.prepare(class_map=None)
        self.internal_to_original_ids = {
            img_id: img_info['id']
            for img_id, img_info in enumerate(self.image_info)}
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.class_map = bidict({
            cat['id']: cat['name'] for cat in
            sorted(cats, key=lambda x: x['id'])})
        self.class_names = list(self.class_map.values())
        
        if isinstance(config, str):
            import importlib.util
            spec = importlib.util.spec_from_file_location("mrcnn.config_file", config)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.config = module.Config()
        else:
            self.config = config

    def load_coco(self, image_dir=None, class_ids=None, class_map=None):
        """Load a subset of the COCO dataset.
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        """
        coco = self.coco

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for idx in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[idx])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            if 'path' in coco.imgs[i]:
                path = coco.imgs[i]['path']
            else:
                path = os.path.join(image_dir, coco.imgs[i]['file_name'])
            self.add_image(
                "coco", image_id=i,
                path=path,
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            raise NotImplementedError("load_mask from COCO dataset but source is not.")

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        # Call super class to return an empty mask
        return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def mrcnn_generator(
            self, shuffle=True, augment=False, augmentation=None,
            random_rois=0, batch_size=1, detection_targets=False, no_augmentation_sources=None):
        from mrcnn.model import data_generator
        return data_generator(
            self, self.config, shuffle, augment, augmentation,
            random_rois, batch_size, detection_targets, no_augmentation_sources)

    def generator(self):
        raise NotImplementedError()

