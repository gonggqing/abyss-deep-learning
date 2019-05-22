import json
import logging
import os
import sys
import warnings
from collections import Counter
from contextlib import redirect_stdout
from io import TextIOWrapper
from numbers import Number
from typing import Tuple, Union, List

import PIL.Image
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from skimage.draw import polygon, polygon_perimeter, line
from skimage.measure import find_contours, approximate_polygon


def cv2_to_Pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(image)


def instance_to_caption(coco):
    """convert a COCO dataset from instance labels to captions"""
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
    """Setup which GPUs to use (or CPU), must be called before "import keras".
    Note that in Keras GPUs are allocated to processes, and memory is only released
    when that process ends.

    Args:
        gpu_ids (list, optional): A list of integer GPU IDs, or None for CPU.
        allow_growth (bool, optional): Allow memory growth in GPUs if True, else allocate entire GPUs.
        log_device_placement (bool, optional): Log device placement for debugging purposes.
    """

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
    captions_in_image = {  # Counts how many captions are in each image
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


def tile_gen(image, tile_size, stride=None, fill_const=0):
    from itertools import product
    if stride is None:
        stride = tile_size
    height, width, depth = image.shape
    tile_height, tile_width = tile_size
    stride_height, stride_width = stride
    num_tiles_y = int(np.floor((height - tile_height) / stride_height + 1))
    num_tiles_x = int(np.floor((width - tile_width) / stride_width + 1))

    for tile_y, tile_x in product(range(num_tiles_y), range(num_tiles_x)):
        tile = np.ones((tile_height, tile_width, depth), dtype=image.dtype) * fill_const
        y1, x1 = tile_y * stride_height, tile_x * stride_width
        y2, x2 = min(tile_height + stride_height * tile_y, height), min(tile_width + stride_width * tile_x, width)
        h, w = y2 - y1, x2 - x1
        tile[:h, :w, ...] = image[y1:y2, x1:x2, ...]
        yield tile


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
    num_channels = None
    image = None
    for (i, j), window in zip(product(*[range(k) for k in num_tiles]), tiles):
        if image is None:
            num_channels = window.shape[-1]
            # image = np.zeros(image_size + (num_channels,), dtype=window.dtype)  # TODO FIX then uncomment
            image = np.zeros(image_size + image_size[3:], dtype=window.dtype)
        y1, y2 = np.array([i, i + 1]) * window_size[0]
        x1, x2 = np.array([j, j + 1]) * window_size[1]
        y2a = np.minimum(image_size[0], y2)
        x2a = np.minimum(image_size[1], x2)
        h, w = y2a - y1, x2a - x1
        image[y1:y2a, x1:x2a, ...] = window[:h, :w, ...]
    return image


def instance_to_categorical(masks, mask_classes, num_classes):
    """Convert a instance mask array into a categorical mask array.

    Args:
        masks (np.ndarray): An array of shape [height, width, #instances] of type np.bool.
        mask_classes (np.ndarray): An array of shape [#instances] of integer type.
        num_classes (int): The maximum number of classes to return in the last dimension.

    Returns:
        np.ndarray: An array of shape [height, width, max_classes] of type np.uint8.
    """
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
    """This is a decorator which can be used to emit a warning when the
    marked function is called."""

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
    """This is a decorator which can be used to emit a warning the first time
    that the marked function is called."""

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


def imread(path: str, size: Tuple[int, int] = None, dtype=None):
    """
    Read an image from the file system, optionally resizing the image size.

    Args:
        path: Path to image, relative or absolute
        size: 2-tuple (width, height)
        dtype: numpy dtype to cast to

    """
    im = PIL.Image.open(path)
    if size is not None:
        im = im.resize(size, resample=PIL.Image.BICUBIC)
    return np.array(im, dtype=dtype)


def imwrite(im: Union[np.ndarray, PIL.Image.Image], path: str, size: Tuple[int, int] = None):
    """
    Write an image to file system, optionally resizing the image size

    Args:
        im: Image, either numpy array or PIL Image class
        path: Path to image, relative or absolute
        size: 2-tuple (width, height)

    Returns:

    """
    if isinstance(im, np.ndarray):
        im = PIL.Image.fromarray(im)
    assert isinstance(im, PIL.Image.Image), "Expected instance of PIL.Image.Image"
    if size is not None:
        im = im.resize(size, resample=PIL.Image.BICUBIC)
    im.save(path)


def close_contour(contour: np.array):
    """
    Snippet taken from pycococreator, ensures contour is closed, i.e. first point and last point are the same
    """
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    return contour


def polygon_x(polygon_: List[Number]) -> np.array:
    """
    Get x points from list of [x1, y1, x2, y2, ..., xN, yN]
    """
    return np.array(polygon_[::2])


def polygon_y(polygon_: List[Number]) -> np.array:
    """
    Get y points from list of [x1, y1, x2, y2, ..., xN, yN]
    """
    return np.array(polygon_[1::2])


def shoelace_area(polygon_: List[Union[int, float]]) -> float:
    """
    Uses shoelace formula to calculate the area of a polygon,
    do not use to calculate area for COCO representation of polygon
    """
    x = polygon_x(polygon_)
    y = polygon_y(polygon_)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_area(polygon_: List[Union[int, float]]) -> float:
    """
    Converts a list of polygons into binary mask,
    then to run-time length encoding format to calculate area using the pycocotools
    """
    return maskUtils.area(maskUtils.encode(np.asfortranarray(polygon_to_mask(polygon_).astype(np.uint8))))


def x_y_to_polygon(x: np.array, y: np.array) -> np.array:
    """
    Flattens an array of x points and an array of y points to [x1, y1, x2, y2, ..., xN, yN]
    """
    flat = np.empty([x.size + y.size], dtype=float)
    flat[0::2] = x
    flat[1::2] = y
    return flat


def polygon_to_mask(polygon_: List[Union[int, float]], value: float = 1) -> np.array:
    """
    Generates a mask from polygon, defaults to 1, i.e. a binary mask
    TODO: for efficeincy do min_x and min_y so the made grid is smaller
    """
    max_x = np.round(np.max(polygon_[::2])) + 1
    max_y = np.round(np.max(polygon_[1::2])) + 1
    grid = np.zeros([max_y, max_x])
    return draw_polygon(polygon_, grid, value)


def draw_polygon(polygon_: List[Union[int, float]], grid: np.array, value: float = 1) -> np.array:
    x = np.round(polygon_[::2]).astype(int)
    y = np.round(polygon_[1::2]).astype(int)
    grid[polygon(y, x)] = value
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    if nx > 2 or ny > 2:
        grid[polygon_perimeter(y, x)] = value
    elif nx == 2 or ny == 2:
        for i in range(len(x)-1):
            if(y[i] == y[i+1] and x[i] == x[i+1]):
                grid[y, x] = value # single pixel drawn
            else:
                grid[line(y[i], x[i], y[i+1], x[i+1])] = value # draw line
    elif nx == 1 and ny == 1:
        grid[y, x] = value
    else:
        logging.warning(f'empty polygon')
    return grid


def polygon_points(polygon_: List[Union[int, float]]) -> np.array:
    """
    Given a polygon list, generate all indices of the polygon mask
    """
    x = np.round(polygon_[::2]).astype(int)
    y = np.round(polygon_[1::2]).astype(int)
    return np.hstack([polygon(y, x), polygon_perimeter(y, x)])


def mask_to_polygon(mask: np.array, value: float = 1, tolerance: float = 0) -> List[Union[int, float]]:
    """
    Converts a mask of a given value to a list of polygon points
    """
    polygons = []
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    contours = find_contours(padded_mask, value - np.finfo(type(value)) * 2)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = approximate_polygon(contour, tolerance)
        contour = np.flip(contour, axis=1)
        contour = contour.ravel()
        contour[contour < 0] = 0
        contour = contour.tolist()
        polygons.append(contour)
    return polygons


def bbox_area(bbox: List[Number]) -> Number:
    """
    Calculate area of a bounding box in the COCO format
    """
    x, y, width, height = bbox
    return width * height


def bbox_to_polygon(bbox: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Polygon equivalent of a bounding box in the COCO format
    """
    x, y, width, height = bbox
    return [x, y, x + width, y, x + width, y + height, x, y + height]


def polygon_to_bbox(poly: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Bounding box equivalent of polygon in the COCO format
    """
    x = poly[::2]
    y = poly[1::2]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    return [min_x, min_y, max_x - min_x + 1, max_y - min_y + 1]


def do_overlap(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]):
    """
    Checks to see if two bounding boxes overlap eachother

    Args:
        bbox_a: (x1, y1, x2, y2) where <x1>,<y1> is top left and <x2>,<y2> is bottom right
        bbox_b: (x1, y1, x2, y2) where <x1>,<y1> is top left and <x2>,<y2> is bottom right
    """
    assert len(bbox_a) == 4, "There should be 4 values in bbox_a"
    assert len(bbox_b) == 4, "There should be 4 values in bbox_b"
    return not (bbox_a[0] > bbox_b[2] or bbox_b[0] > bbox_a[2] or bbox_a[3] < bbox_b[1] or bbox_b[3] < bbox_a[1])


class MyCOCO(COCO):
    """ Create COCO object by reading from file path or from stdin """

    class Verbose:
        @staticmethod
        def write(line: str):
            line = line.strip()
            if line:
                logging.info(line)

    def __init__(self, buffer: Union[str, TextIOWrapper] = None):
        if isinstance(buffer, str) or buffer is None:
            with redirect_stdout(MyCOCO.Verbose):
                super().__init__(annotation_file=buffer)
        elif isinstance(buffer, TextIOWrapper):
            json_string = buffer.read()
            if json_string:
                self.dataset = json.loads(json_string)
                self.createIndex()
            else:
                logging.error("Expecting input from stdin: received empty characters {}".format(repr(json_string)))
                sys.exit(1)
        else:
            logging.error("Unknown data type {}, exiting".format(type(buffer)))
            sys.exit(1)

    @property
    def info(self):
        return self.dataset.get('info', {})

    @property
    def annotations(self):
        return self.dataset.get('annotations', [])

    @property
    def images(self):
        return self.dataset.get('images', [])

    @property
    def categories(self):
        return self.dataset.get('categories', [])

    @info.setter
    def info(self, info):
        self.dataset['info'] = info

    @annotations.setter
    def annotations(self, annotations):
        self.dataset['annotations'] = annotations

    @images.setter
    def images(self, images):
        self.dataset['images'] = images

    @categories.setter
    def categories(self, categories):
        self.dataset['categories'] = categories

    def createIndex(self):
        with redirect_stdout(MyCOCO.Verbose):
            super().createIndex()


def image_streamer(sources, start=0, remap_func=None):
    """A generator that produces image frames from multiple sources.
    Currently accepts video, images and COCO datasets and globs of these.

        sources: list of str; The file paths to the image sources.
                 Can be an image, video or COCO json, globs accepted.
        start: int (optional); Start from this position in the list.
        remap_func: lambda or function; A function that accepts a filename
                    parameter and outputs the path to the file. Used to 
                    change relative directories of COCO datasets.

    
    """
    from warnings import warn
    from glob import glob
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
                # TODO: It's not clear how to address relative paths
                image_path = image['path'] if 'path' in image else remap_func(image['file_name'])
                yield image_path, frame_no, imread(image_path)
            del coco
        else:
            warn("Skipped an unknown source type {}.".format(source))


def print_v(*args, level=0):
    if print_v._level >= level:
        print(*args, file=sys.stderr)


print_v._level = 0
