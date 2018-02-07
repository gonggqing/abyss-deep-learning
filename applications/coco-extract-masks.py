#!/usr/bin/env python3

import argparse
import os
from pycocotools.coco import COCO
from pycocotools import mask
import cv2
import numpy as np

def decode_rbg_label(encoded):
    annotation = np.bitwise_or(np.bitwise_or(
    encoded[:, :, 0].astype(np.uint32),
    encoded[:, :, 1].astype(np.uint32) << 8),
    encoded[:, :, 2].astype(np.uint32) << 16)
    return annotation

def encode_rgb_label(annotation, dtype=np.uint8):
    encoded = np.stack([
        np.bitwise_and(annotation, 255),
        np.bitwise_and(annotation >> 8, 255),
        np.bitwise_and(annotation >> 16, 255),
    ], axis=2).astype(dtype)
    return encoded

def foreach_coco_image(coco_path, function):
    db = COCO(coco_path)
    image_ids = db.getImgIds()
    images = db.loadImgs(image_ids)
    for image in images:
        annotations = db.loadAnns(db.getAnnIds(imgIds=[image['id']]))
        function(db, image, annotations)

def save_mask_as_image(db, image, annotations, destination_dir, multiclass=False, as_rgb=False):
    annotation_img = np.zeros((image['height'], image['width']), dtype=np.uint8)
    for annotation in annotations:
        label = db.annToMask(annotation)#.astype(np.uint8)
        if multiclass and annotation['category_id'] > 0:
            annotation_img &= (label << (int(annotation['category_id']) - 1))
        else:
            # Note the order of the annotations may change the way this appears
            annotation_img[label] = annotation['category_id']
    if as_rgb:
        annotation_img = encode_rgb_label(annotation_img)
    image_path = os.path.join(destination_dir, image['file_name'])
    cv2.imwrite(image_path, annotation_img)

def main(args):
    foreach_coco_image(args.coco_path,
            lambda db, image, annotation: \
                save_mask_as_image(db, image, annotation, args.destination_dir, args.multiclass, args.as_rgb)
    )


def get_args():
    '''Get args from the command line args'''
    parser = argparse.ArgumentParser(description="Extract labels from COCO JSON and dump them in to annotation images")
    parser.add_argument("coco_path", help="The coco JSON to parse.")
    parser.add_argument("destination_dir", help="The destination to dump the label images.")
    parser.add_argument("--multiclass", help="Create an encoded multi-class image", action='store_true')
    parser.add_argument("--as-rgb", help="Encode image classes into RGB", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(get_args())
