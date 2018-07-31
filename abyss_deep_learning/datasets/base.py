'''Base classes for supervised and unsupervised datasets.
'''
from itertools import cycle
from random import shuffle

from pycocotools.coco import COCO
from skimage.color import gray2rgb
from skimage.io import imread
import numpy as np

############################# Supervised Datasets #############################

class ImageTargetDataset(COCO):
    '''Base class for a supervised COCO style dataset where every image has 1 or more targets associated with it.'''
    def __init__(self, annotation_file, **kwargs):
        super().__init__(annotation_file, **kwargs)

    @property
    def num_classes(self):
        raise NotImplementedError("ImageTargetDataset::num_classes needs to be overloaded in subclass.")

    def load_image_targets(self, img_id):
        raise NotImplementedError("ImageTargetDataset::load_image_targets needs to be overloaded in subclass.")
        
    @property
    def image_ids(self):
        return list(self.imgs.keys())

    @property
    def num_images(self):
        return len(self.image_ids)

    def generator(self, imgIds=None, shuffle_ids=False):
        if not imgIds:
            # imgIds = np.unique([ann['image_id'] for ann in self.loadAnns(ids=self.getAnnIds())])
            imgIds = self.image_ids
        if shuffle_ids:
            shuffle(imgIds)
        for image_id in cycle(imgIds):
            yield self.load_image(image_id), self.load_image_targets(image_id)

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
            print(image)
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


############################ Unsupervised Datasets ############################
#TODO: Base class for unsupervised datasets
