from collections import Counter

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
    '''Setup GPU, must be called before "import keras"'''
    import tensorflow as tf
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    import keras.backend as K
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = allow_growth
    K.set_session(tf.Session(config=config))


# Find a balanced set
def balanced_set(coco, ignore_captions=None):
    ignore_captions = ignore_captions or []
    captions = [caption 
            for ann in coco.anns.values() if 'caption' in ann
           for caption in ann['caption'].split(',') if caption != "background" and caption not in ignore_captions]
    smallest_caption, smallest_caption_value = min(Counter(captions).items(), key=lambda x: x[1])
    
    unique_captions = np.unique(captions)
#     print("unique_captions", unique_captions)
    # Count how many images are in each label
    images_in_caption = {
        caption: [ann['image_id'] for ann in coco.anns.values() if caption in ann['caption'].split(',')]
        for caption in unique_captions}
    # print("images_in_caption", {k: len(i) for k, i in images_in_caption.items()})
    for images in images_in_caption.values():
        np.random.shuffle(images)
    
    # Count how many captions are in each image
    captions_in_image = {
        image_id: ([
            caption
            for ann in coco.anns.values() if ann['image_id'] == image_id and 'caption' in ann
            for caption in ann['caption'].split(',') if len(caption) and caption != "background" and caption not in ignore_captions])
        for image_id in coco.imgs}
    # print("captions_in_image")
    # print([len(captions) for image_id, captions in captions_in_image.items()])
    
    # print("smallest", smallest_caption, smallest_caption_value)
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

############################################################
# The following two functions are from pycocotools with a few changes.
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


def ann_to_mask(ann, output_shape):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = ann_rle_encode(ann, output_shape[0], output_shape[1])
    mask = maskUtils.decode(rle)
    return mask
