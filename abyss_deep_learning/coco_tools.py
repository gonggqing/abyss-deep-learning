from PIL import Image
import sys
import os
from datetime import datetime

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
