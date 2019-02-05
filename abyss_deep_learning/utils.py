from collections import Counter
import os
import warnings
import sys

from pycocotools import mask as maskUtils
import cv2
import numpy as np
import PIL.Image
import pandas as pd


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

def balanced_annotation_set(coco, ann_type='caption', num_anns=None, ignore=None):
    """Return a subset of image IDs that produces a balanced set with at least num_anns annotations per class.

    Args:
        coco (pycocotools.COCO): The loaded COCO database
        ann_type ('caption' or 'category_id'): Whether to operate on caption or instance data.
        num_anns (int, optional): The number of annotations per class to aim for.
            If None then make a balanced set equal to the smallest class count.
        ignore (list, optional): A list of captions or category IDs to ignore.

    Returns:
        list of ints: A list of the image ids that form the subset.
    """
    ignore = ignore or []
    annotations = [ann for ann in coco.anns.values() if ann_type in ann and ann[ann_type] not in ignore]
    captions = [
        ann[ann_type] for ann in annotations
        if ann_type in ann and ann[ann_type] not in ignore]
    count_captions = Counter(captions)
    unique_captions = np.unique(captions)
    image_ids_for_class = dict(pd.DataFrame(list(coco.anns.values()))[["image_id", ann_type]] \
        .groupby(ann_type).apply(lambda x: x['image_id'].as_matrix().tolist()))
    captions_in_image = { # Counts how many captions are in each image
        image_id: ([
            ann[ann_type]
            for ann in annotations if ann['image_id'] == image_id])
        for image_id in coco.getImgIds()}

    out = {caption: [] for caption in unique_captions}
    caption_count = {caption: 0 for caption in unique_captions}

    def add_to_counts(image_id):
        # Increment counts for all captions in image
        for caption in captions_in_image[image_id]:
            out[caption].append(image_id)
            caption_count[caption] += 1
        # Remove image_id from all images_in_caption
        for images in image_ids_for_class.values():
            if image_id in images:
                images.pop(images.index(image_id))

    target_set_size = num_anns or min(count_captions.values())
    while any([caption_count[caption] < target_set_size for caption in unique_captions]):
        least = min(out.items(), key=lambda x: len(x[1]))
        image_id = image_ids_for_class[least[0]].pop()
        add_to_counts(image_id)
    out = list(set([j
           for i in out.values()
          for j in i]))
    return out


def tile_gen(image, window_size, fill_const=0):
    '''window_size must be tuple of ints'''
    from itertools import product
#     print("tile gen")
    num_tiles = np.ceil(image.shape[0] / window_size[0]), np.ceil(image.shape[1] / window_size[1])
    num_tiles = [int(i) for i in num_tiles]

    for i, j in product(*[range(k) for k in num_tiles]):
        window = np.ones(tuple(window_size) + image.shape[2:], dtype=image.dtype) * fill_const
        y1, y2 = np.array([i, i + 1]) * window_size[0]
        x1, x2 = np.array([j, j + 1]) * window_size[1]
        y2a = np.minimum(image.shape[0], y2)
        x2a = np.minimum(image.shape[1], x2)
        h, w = y2a - y1, x2a - x1
        window[:h, :w, ...] = image[y1:y2a, x1:x2a, ...]
        yield window

def detile(tiles, window_size, image_size):
    """Reassembles tiled images.

    Args:
        tiles (iterable of np.ndarray): The list of tiles to assemble, in order.
        window_size (tuple of ints): The (height, width) size of the window to tile.
        image_size (tuple of ints): The original (height, width) of the image.

    Yields:
        np.ndarray: The reassembled image.
    """
    from itertools import product
#     print("detile")
    num_tiles = np.ceil(image_size[0] / window_size[0]), np.ceil(image_size[1] / window_size[1])
    num_tiles = [int(i) for i in num_tiles]
    image = None
    for (i, j), window in zip(product(*[range(k) for k in num_tiles]), tiles):
        if image is None:
            image = np.zeros(image_size + image_size[3:], dtype=window.dtype)
        y1, y2 = np.array([i, i + 1]) * window_size[0]
        x1, x2 = np.array([j, j + 1]) * window_size[1]
        y2a = np.minimum(image_size[0], y2)
        x2a = np.minimum(image_size[1], x2)
        h, w = y2a - y1, x2a - x1
        image[y1:y2a, x1:x2a, ...] = window[:h, :w, ...]
    return image

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


def imread(path, output_size=None, dtype=None):
    """Read an image from the filesystem, optionally resizing the image size and casting the bit_depth"""
    im = PIL.Image.open(path)
    if output_size:
        im = im.resize(output_size)
    im = np.array(im)
    if dtype:
        im.astype(dtype)
    return im


def image_streamer(sources, start=0, remap_func=None):
    '''A generator that produces image frames from multiple sources.
    Currently accepts video, images and COCO datasets and globs of these.

        sources: list of str; The file paths to the image sources.
                 Can be an image, video or COCO json, globs accepted.
        start: int (optional); Start from this position in the list.
        remap_func: lambda or function; A function that accepts a filename
                    parameter and outputs the path to the file. Used to 
                    change relative directories of COCO datasets.

    
    '''
    from warnings import warn
    from glob import glob
    from pycocotools.coco import COCO
    from skvideo.io import FFmpegReader
    from contextlib import closing, redirect_stdout

    def is_image(path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))
    def is_video(path):
        return path.lower().endswith(('.avi', '.mpg', '.mp4'))

    remap_func = remap_func or (lambda x: x)

    # Expand any globbed paths, but not for images since we want to keep the sequence
    full_sources = []
    for source in sources:
        if '*' in source and not is_image(source):
            full_sources += glob(source, recursive=True)
        else:
            full_sources.append(source)

    for source in full_sources[start:]:
        if is_video(source):
            with closing(FFmpegReader(source)) as reader:
                for frame_no, frame in enumerate(reader.nextFrame()):
                    yield source, frame_no, frame
        elif is_image(source):
            for frame_no, image_path in enumerate(glob(source, recursive=True)):
                yield image_path, frame_no, imread(remap_func(image_path))
        elif source.endswith('.json'):
            # COCO database
            with redirect_stdout(None):
                coco = COCO(source)
            for frame_no, image in enumerate(coco.loadImgs(coco.getImgIds())):
                #TODO: It's not clear how to address relative paths
                image_path = image['path'] if 'path' in image else remap_func(image['file_name'])
                yield image_path, frame_no, imread(image_path)
            del coco
        else:
            warn("Skipped an unknown source type {}.".format(source))


def print_v(*args, level=0):
    if print_v._level >= level:
        print(*args, file=sys.stderr)
print_v._level = 0

