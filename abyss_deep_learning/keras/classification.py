'''
Utilities and generators for classification tasks.
Generators expect yield type to be (image, target) and not in a batch.
'''

from itertools import cycle
from random import shuffle

from imgaug import augmenters as iaa
from pycocotools.coco import COCO
import numpy as np
from skimage.io import imread
from skimage.color import gray2rgb

class FromAnnDataset(COCO):
    def __init__(self, *args, **kwargs):
        super(FromAnnDataset, self).__init__(*args, **kwargs)
        
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
        
    def load_categories(self, image_id):
        assert isinstance(image_id, int), "Must pass exactly one ID"
        caps = [annotation['category_id']
         for annotation in self.loadAnns(self.getAnnIds([image_id]))]
        return set(caps)
    
    def num_images(self, imgIds=[], catIds=[]):
        return len(self.getAnnIds(imgIds, catIds))
        
    def generator(self, imgIds=[], shuffle_ids=False):
        if not imgIds:
            imgIds = [ann['image_id'] for ann in self.loadAnns(ids=self.getAnnIds())]
        if shuffle_ids:
            shuffle(imgIds)
        for image_id in cycle(imgIds):
            yield self.load_image(image_id), self.load_categories(image_id)

class ClassificationDataset(COCO):
    def __init__(self, caption_map, *args, **kwargs):
        super(ClassificationDataset, self).__init__(*args, **kwargs)
        
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
        
    def load_caption(self, image_id):
        assert isinstance(image_id, int), "Must pass exactly one ID"
        caps = [caption.split(',')
         for annotation in self.loadAnns(self.getAnnIds([image_id]))
         for caption in annotation['caption']]
        return set([i for f in caps for i in f])
    
    def num_images(self, imgIds=[], catIds=[]):
        return len(self.getAnnIds(imgIds, catIds))
        
    def generator(self, imgIds=[], shuffle_ids=False):
        if not imgIds:
            imgIds = [ann['image_id'] for ann in self.loadAnns(ids=self.getAnnIds())]
        if shuffle_ids:
            shuffle(imgIds)
        for image_id in cycle(imgIds):
            yield self.load_image(image_id), self.load_caption(image_id)

####### Generators ######
            
def caption_map_gen(gen, caption_map):
    for image, captions in gen:
        yield image, [caption_map[i] for i in captions]
        
def onehot_gen(gen, num_classes):
    for image, captions in gen:
        onehot = np.array([1 if i in captions else 0 for i in range(num_classes)])
        yield image, onehot

def augmentation_gen(gen, aug_config, enable=True):
    '''
    Data augmentation for classification task.
    Target is untouched.
    '''
    if not enable:
        while True:
            yield from gen
    seq = iaa.Sequential([
        iaa.Fliplr(aug_config['flip_lr_percentage']),
        iaa.Flipud(aug_config['flip_ud_percentage']),
        iaa.Affine(**aug_config['affine'])
    ])
    for image, target in gen:
        yield seq.augment_image(image), target
