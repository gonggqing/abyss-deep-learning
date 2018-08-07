'''Base classes for supervised and unsupervised datasets.
'''
from itertools import cycle
from random import shuffle

from pycocotools.coco import COCO
from skimage.color import gray2rgb
from skimage.io import imread
import numpy as np
from abc import ABC

############################# Supervised Datasets #############################

class SupervisedDataset(COCO):
    '''Base class for a supervised COCO style dataset where every image has 1 or more targets associated with it.'''
    def __init__(
            self, annotation_file, annotation_type, image_dir=None,
            class_ids=None, preload=False, class_map=None, **kwargs):
        '''annotation_type one of ['object', 'caption']'''
        COCO.__init__(self, annotation_file)
        self.coco = self
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.data = []

        # All images or a subset?
        if class_ids:
            image_ids = []
            for class_id in class_ids:
                image_ids.extend(list(self.getImgIds(catIds=[class_id])))
            # Remove duplicates
            self._image_ids_orig = list(set(image_ids))
        else:
            # All images
            class_ids = sorted(self.getCatIds())
            self._image_ids_orig = [img['id'] for img in self.imgs.values()]
            
        self.class_ids = class_ids
        self.num_classes = len(class_ids)
        
        if class_map:
            self.class_map = class_map
        else:
            self.class_map = {c: i for i, c in enumerate(self.cats.keys())}
        if preload:
            self.preload_images()

    def load_targets(self, img_id):
        raise NotImplementedError(
            "ImageTargetDataset::load_targets needs to be overloaded in subclass.")
    
    def generator(self, imgIds=None, shuffle_ids=False):
        if not imgIds:
            # imgIds = np.unique([ann['image_id'] for ann in self.loadAnns(ids=self.getAnnIds())])
            imgIds = self._image_ids_orig
        if shuffle_ids:
            shuffle(imgIds)
        for image_id in cycle(imgIds):
            yield self.load_image(image_id), self.load_targets(image_id)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_path = self.loadImgs([image_id])[0]['path']
        image = imread(image_path)
        if not isinstance(image, np.ndarray):
            # Temporary bugfix for PIL error
            image = imread(image_path, plugin='matplotlib') * 255
        if not isinstance(image, np.ndarray):
            raise Exception(
                "skimage.io.read failed: image is of type {:s}".format(str(type(image))))
        # If grayscale. Convert to RGB for consistency.
        if image.ndim == 1 or (image.ndim == 3 and image.shape[2] == 1):
            image = gray2rgb(np.squeeze(image))
        elif image.ndim == 3:
            if image.shape[2] > 3:
                image = image[..., 0:3]  # Remove any alpha channel
            elif image.shape[2] != 3:
                raise Exception(
                    "load_image tried to load an image with dims of {:s}".format(str(image.shape)))
        return image
    
    def preload_images(self):
        self.data = {
            image_id: (self.load_image(image_id), self.load_targets(image_id))
            for image_id in self._image_ids_orig}

    def apply(self, func_input, func_target=None, imgIds=[]):
        '''Applies functions to input and target, if the DB is preloaded'''
        assert self.data, "apply() only works on preloaded images"
        if not func_target:
            func_target = lambda x: x
        imgIds = imgIds or self._image_ids_orig
        for image_id in imgIds:
            self.data[image_id] = (func_input(image_id[0]), func_target(image_id[1]))

    def sample(self):
        image_id = np.random.choice(self._image_ids_orig, replace=False, size=None)
        return self.load_image(image_id), self.load_targets(image_id)

    def generator(self, imgIds=[], shuffle_ids=False):
        imgIds = imgIds or self._image_ids_orig
        imgIds = list(imgIds) # make a copy
        # catIds = catIds or self.class_ids
        if shuffle_ids:
            shuffle(imgIds)
        for image_id in cycle(imgIds):
            yield self.load_image(image_id), self.load_targets(image_id)


############################ Unsupervised Datasets ############################
#TODO: Base class for unsupervised datasets
