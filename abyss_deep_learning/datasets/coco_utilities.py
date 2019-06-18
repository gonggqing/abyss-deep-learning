from PIL import Image
import sys
import os
from datetime import datetime
import logging

__author__ = 'Toby Dunne'


def template_license():
    return [{
            'id': 0,
            'url': 'http://www.abysssolutions.com.au',
            'name': 'Commercial in Confidence - do not distribute.',
        }]


def template_info():
    return {
        'contributor': 'Abyss Solutions',
        'total_time': '00h00m00s',
        'year': str(datetime.now().year),
        'date_created': str(datetime.now()),
        'description': 'This is a dataset configured by Abyss Solutions.',
        'version': '1.0',
        'url': 'http://www.abysssolutions.com.au',
    }


def coco_is_valid(file_name, coco, mode):

    if mode == 'skip':
        return True

    full = mode == 'strict'

    is_valid = True

    try:
        if all([x not in coco for x in ['images', 'annotations', 'categories']]):
            logging.error("VALIDATION: coco must contain images annotations or categories")
            is_valid = False

        if 'annotations' in coco and not 'images' in coco:
            logging.error("VALIDATION: coco contains annotations with no images")
            is_valid = False

        if 'images' in coco:
            if any([not isinstance(x['id'], int) for x in coco['images']]):
                logging.error("VALIDATION: 1 or more images with id=None or not int")
                is_valid = False

            image_set = {el['id'] for el in coco['images']}
            if len(image_set) != len(coco['images']):
                logging.error("VALIDATION: duplicated image ids")
                is_valid = False

            if any([not isinstance(x['width'], int) for x in coco['images']]):
                logging.error("VALIDATION: 1 or more images with invalid width")
                is_valid = False

            if any([not isinstance(x['height'], int) for x in coco['images']]):
                logging.error("VALIDATION: 1 or more images with invalid width")
                is_valid = False

        if 'annotations' in coco:
            if any([not isinstance(x['id'], int) for x in coco['annotations']]):
                logging.error("VALIDATION: 1 or more annotation with id=None or not int")
                is_valid = False

            if any(['image_id' not in x or not isinstance(x['image_id'], int) for x in coco['annotations']]):
                logging.error("VALIDATION: 1 or more annotation with image_id=None or not int")
                is_valid = False

            # annotation must have caption or category_id
            # if category_id is present, it must be an int
            if any([('caption' not in x and 'category_id' not in x)
                    or ('category_id' in x and not isinstance(x['category_id'], int)) for x in coco['annotations']]):
                logging.error("VALIDATION: 1 or more annotation without caption and invalid category_id")
                is_valid = False

            annotation_set = {el['id'] for el in coco['annotations']}
            if len(annotation_set) != len(coco['annotations']):
                logging.error("VALIDATION: duplicated annotation ids")
                is_valid = False

        if 'categories' in coco:
            if any([not isinstance(x['id'], int) for x in coco['categories']]):
                logging.error("VALIDATION: 1 or more categories with id=None or not int")
                is_valid = False

            category_set = {el['id'] for el in coco['categories']}
            if len(category_set) != len(coco['categories']):
                logging.error("VALIDATION: duplicated category ids")
                is_valid = False

        if full:
            # full checks cross reference of each annotation
            if 'annotations' in coco and len(coco['annotations']) > 0:
                for ann in coco['annotations']:
                    if ann['image_id'] not in image_set:
                        logging.error("VALIDATION: annotation id {} refers to missing image id {}".format(ann['id'], ann['image_id']))
                        is_valid = False

                for ann in coco['annotations']:
                    if 'category_id' not in ann and 'caption' not in ann:
                        logging.error("VALIDATION: annotation id:{} without category_id or caption".format(ann['id']))
                        is_valid = False
                    elif 'category_id' in ann and ann['category_id'] not in category_set:
                        logging.error("VALIDATION: annotation id:{} refers to missing category id {}".format(ann['id'], ann['category_id']))
                        is_valid = False
                        break

    except Exception as ex:
        logging.error("Exception while validating {}".format(ex))
        raise
        is_valid = False

    if not is_valid:
        logging.error("Error file is not valid coco: {}".format(file_name))

    return is_valid


def coco_from_images(image_paths, width, height):
    coco_json = {
        'images'     : [],
        'annotations': [],
        'categories' : []
    }

    for i, image_path in enumerate(image_paths):
        if width is None or height is None:
            try:
                image_width, image_height = Image.open(image_path).size
            except OSError:
                print("coco-from-images: ERROR image read error '{}' - use --image_size to override".format(image_path),
                      file=sys.stderr)
                sys.exit(1)
        else:
            image_width = width
            image_height = height

        coco_json['images'].append(
             {
                'id'       : i,
                'file_name': os.path.basename(image_path),
                'path'     : image_path,
                'width'    : image_width,
                'height'   : image_height,
            }
        )
    return coco_json
