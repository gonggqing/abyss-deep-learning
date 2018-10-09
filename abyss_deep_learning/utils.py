from collections import Counter
import os
import warnings

from pycocotools import mask as maskUtils
import cv2
import numpy as np
import PIL.Image

def cv2_to_Pil(image):
    image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(image)


def instance_to_caption(coco):
    '''convert a COCO dataset from instance labels to captions'''
    caption_map_r = {cat['id']: cat['name'] for cat in coco['categories']}
    annotations = {}
    for image in coco['images']:
        image_id = image['id']
        anns = [
            annotation for annotation in coco['annotations']
            if 'category_id' in annotation and annotation['image_id'] == image_id]
        anns_str = set([caption_map_r[ann['category_id']] for ann in anns]) if anns else {'background'}
        annotations[image_id] = {
            "caption": ','.join(list(anns_str)),
            "id": image_id,
            "image_id": image_id,
            "type": 'class_labels'
        }
    coco['annotations'] = list(annotations.values())
    coco.pop('categories', None)
    coco.pop('captions', None)
    return coco


def config_gpu(gpu_ids=[], allow_growth=False, log_device_placement=True):
    '''Setup which GPUs to use (or CPU), must be called before "import keras".
    Note that in Keras GPUs are allocated to processes, and memory is only released
    when that process ends.
    
    Args:
        gpu_ids (list, optional): A list of integer GPU IDs, or None for CPU.
        allow_growth (bool, optional): Allow memory growth in GPUs if True, else allocate entire GPUs.
        log_device_placement (bool, optional): Log device placement for debugging purposes.
    '''

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpu_ids is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''    
    elif gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    # default empty list does not overwrite CUDA_VISIBLE_DEVICES

    import keras.backend as K
    config = K.tf.ConfigProto(log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    K.set_session(K.tf.Session(config=config))


def import_config(args):
    import json
    import yaml
    extension = args.config.split('.')[-1].lower()
    with open(args.config, 'r') as cfg_file:
        if extension in ('yaml', 'yml'):
            cfg = yaml.load(cfg_file)
        elif extension == 'json':
            cfg = json.load(cfg_file)
        else:
            raise ValueError("Config file needs to be a .yaml or .json file")
    return cfg

# Find a balanced set
def balanced_set(coco, ignore_captions=None):
    ignore_captions = ignore_captions or []
    captions = [caption 
            for ann in coco.anns.values() if 'caption' in ann
           for caption in ann['caption'].split(',') if caption != "background" and caption not in ignore_captions]
    smallest_caption, smallest_caption_value = min(Counter(captions).items(), key=lambda x: x[1])
    
    unique_captions = np.unique(captions)
    images_in_caption = {
        caption: [ann['image_id'] for ann in coco.anns.values() if caption in ann['caption'].split(',')]
        for caption in unique_captions}
    for images in images_in_caption.values():
        np.random.shuffle(images)
    
    # Count how many captions are in each image
    captions_in_image = {
        image_id: ([
            caption
            for ann in coco.anns.values() if ann['image_id'] == image_id and 'caption' in ann
            for caption in ann['caption'].split(',') if len(caption) and caption != "background" and caption not in ignore_captions])
        for image_id in coco.imgs}
    balanced = []
    out = {caption: [] for caption in unique_captions}
    
    def add_to_counts(image_id):
        # Increment counts for all captions in image
        for caption in captions_in_image[image_id]:
            out[caption].append(image_id)
        # Remove image_id from all images_in_caption
        for images in images_in_caption.values():
            if image_id in images:
                images.pop(images.index(image_id))
    
    while any([len(out[caption]) < smallest_caption_value for caption in unique_captions]):
        least = min(out.items(), key=lambda x: len(x[1]))
        image_id = images_in_caption[least[0]].pop()
        add_to_counts(image_id)
        
    # print("balanced images in caption")
    # print({k: len(v) for k, v in out.items()})
    out = set([j
           for i in out.values()
          for j in i])

    return out


class tile_iterator(object):
    ''' tile_stride and window_sizes should be the half-size in [height, width] terms
        windows are the large rectangles that choose spacing
        tiles are the smaller rectangles that choose dimension and overlap if tile_stride > window_sizes
    '''

    def __init__(self, image, tile_stride, window_sizes, condition_function=None):
        self.image_shape = np.array(image.shape, dtype=np.int)
        # Enforce rectangular integer grid spacing
        if type(tile_stride) != tuple:
            raise Exception('tile_iterator',
                            'tile_stride must be rectangular (h, w)')
        self.tile_stride = np.array(tile_stride, dtype=np.int)
        # if (type(window_sizes) is not list) or any(map(lambda e: type(e) is not tuple, window_sizes)):
        if type(window_sizes) in [int, float]:
            self.window_sizes = [
                np.array((window_sizes, window_sizes), dtype=np.float32)]
        elif type(window_sizes) is list:
            self.window_sizes = [
                np.array([ws, ws], dtype=np.float32) for ws in window_sizes]
        else:
            raise Exception(
                'tile_iterator', 'window_sizes must scalar, list of scalars, or list of tuples')
        self.window_sizes = [np.floor(window_size)
                             for window_size in self.window_sizes]
        max_window_size = np.floor(self.window_sizes).max(axis=0)
        envelope = np.array([max_window_size[0]+1, self.image_shape[0] - max_window_size[0],
                             max_window_size[1]+1, self.image_shape[1] - max_window_size[1]], dtype=np.int)
        if image.ndim == 3:
            self.channels = image.shape[2]
        elif image.ndim == 2:
            self.channels = 1
        else:
            raise Exception('tile_iterator', 'image is not 2 or 3 dimensional')
        self.grid_points = list(product(
            xrange(envelope[0], envelope[1], tile_stride[0]), xrange(envelope[2], envelope[3], tile_stride[1])))
        if callable(condition_function):
            self.grid_points = [
                point for point in self.grid_points if condition_function(point)]
        self.iter = iter(self.grid_points)

    @staticmethod
    def around(point, size):
        '''get ROI around point given size window'''
        roi = np.array([point[0] - size[0], point[0] + size[0], point[1] -
                        size[1], point[1] + size[1]], dtype=np.int)  # x1 x2 y1 y2
        # roi[0:2] = np.maximum(np.minimum(roi[0:2], self.image_shape[0]), 0)
        # roi[2:4] = np.maximum(np.minimum(roi[2:4], self.image_shape[1]), 0)
        return roi

    def __iter__(self):
        return self

    def next(self):
        '''Get next iterate. Partial tiles are ignored. '''
        point = self.iter.next()
        rois = [self.around(point, window_size)
                for window_size in self.window_sizes]
        return {'center': point, 'rois': rois}


def instance_to_categorical(masks, mask_classes, num_classes):
    '''Convert a instance mask array into a categorical mask array.
    
    Args:
        masks (np.ndarray): An array of shape [height, width, #instances] of type np.bool.
        mask_classes (np.ndarray): An array of shape [#instances] of integer type.
        num_classes (int): The maximum number of classes to return in the last dimension.
    
    Returns:
        np.ndarray: An array of shape [height, width, max_classes] of type np.uint8.
    '''
    num_shapes = len(mask_classes)
    shape = masks.shape
    packed = np.zeros(shape[0:2] + (num_classes,), dtype=np.uint8)
    packed[..., 0] = 1
    for i in range(num_shapes):
        class_id = mask_classes[i]
        mask = masks[..., i]
        packed[..., class_id] |= mask
        packed[..., 0] &= ~mask
    return packed


def cat_to_onehot(cats, num_classes):
    """Convert categorical labels into a onehot.
    
    Args:
        cats (list of ints): A list of ints where each int encodes the class number.
        num_classes (int): The total number of classes.
    
    Returns:
        np.array: The onehot encoded label.
    """
    return np.array([1 if i in cats else 0 for i in range(num_classes)])

############################################################
# The following function is from pycocotools with a few changes.
############################################################

def ann_rle_encode(ann, height, width):
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

def warn_on_call(func, message):
    '''This is a decorator which can be used to emit a warning when the
    marked function is called.'''
    def new_func(*args, **kwargs):
        warnings.warn(
            "{}: {}".format(func.__name__, message),
            category=UserWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def warn_once(func, message):
    '''This is a decorator which can be used to emit a warning the first time
    that the marked function is called.'''
    def new_func(*args, **kwargs):
        if not new_func.__has_run__:
            warnings.warn(
                "{}: {}".format(func.__name__, message),
                category=UserWarning)
            new_func.__has_run__ = True
        return func(*args, **kwargs)
    new_func.__has_run__ = False
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func



