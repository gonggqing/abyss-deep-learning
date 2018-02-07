#!/usr/bin/env python
''' convert labelme format to coco json'''
from __future__ import print_function
from sys import stdout, stderr
import xml.etree.ElementTree as ET
import argparse
import json
import os
import numpy as np

# A single JSON file contains all the label information for a dataset.
# Structure of JSON:
#  annotations: One for each annotation; refers to image_id and category_id
#  catgories: The list of categories (classes) that exist in the annotations in this dataset
#  images: (filename, height, width id, license...) for each image in dataset
#  info: information about this dataset
#  licenses: information in the various licenses for the imagery in this dataset

def xml_text_value(element):
    '''extract text from XML element'''
    return "".join(element.itertext())
def xml_float_value(element):
    '''extract float from XML element'''
    return float("".join(element.itertext()))

def bounding_box(polygon):
    'return the bounding box of a given polygon'
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    return [min_x, min_y, max_x - min_x, max_y - min_y]

def area_polygon(polygon):
    x = [i[0] for i in polygon]
    y = [i[1] for i in polygon]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

def bool_or_text_to_int(item):
    if item is str and (item == 'yes' or item == 'true' or item == 'True'):
        return 1
    if item is int and item > 0:
        return 1
    if item is float and item > 0.0:
        return 1
    return 0

def json_from_labelme(labelme_file, args, class_names, images, categories, annotations):
    data_in = labelme_file.read()
    annotation = ET.fromstring(data_in)
    path_rel = xml_text_value(annotation.find('folder')) + '/' + \
     xml_text_value(annotation.find('filename'))
    url = "http://labelme2.csail.mit.edu/Release3.0/Images/" + path_rel
    # source = xml_text_value(annotation.find('source/submittedBy'))
    image_annotations = []
    for obj in annotation.findall('object'):
        class_name = xml_text_value(obj.find('name'))
        if args.no is not None and class_name in args.no:
            continue
        polygon = [
         (xml_float_value(pt.find('x')), xml_float_value(pt.find('y')))
         for pt in obj.find('polygon').findall('pt')
        ]
        if len(polygon) < 2:
            continue
        if class_name not in class_names:
            class_names[class_name] = len(class_names) + 1
            categories.append(
             {'name': class_name,
              'id': class_names[class_name],
              'supercategory': ''}
            )
        image_annotations.append(
         {
          "area": area_polygon(polygon),
          "bbox": bounding_box(polygon),
          "deleted": bool_or_text_to_int(xml_text_value(obj.find('deleted'))),
          "verified": bool_or_text_to_int(xml_text_value(obj.find('verified'))),
          "date": xml_text_value(obj.find('date')),
          "category_id": class_names[class_name],
          "labelme_id": int(xml_text_value(obj.find('id'))),
          "id": len(annotations) + 1 + len(image_annotations),
          "image_id": len(images) + 1,
          "iscrowd": 0,
          "occluded": bool_or_text_to_int(xml_text_value(obj.find('occluded'))),
          "segmentation": [[j for i in polygon for j in i]]
         }
        )
    if len(image_annotations):
        images.append({
         "coco_url": url,
         "url": url,
         "path": path_rel,
         "date_captured": "1970-01-01 00:00:00",
         "file_name": xml_text_value(annotation.find('filename')),
         "flickr_url": url,
         "height": int(xml_text_value(annotation.find('imagesize/nrows'))),
         "id": len(images) + 1,
         "license": 0,
         "width": int(xml_text_value(annotation.find('imagesize/ncols')))
        })
        annotations.extend(image_annotations)
    if args.verbose:
        print("Annotations: %d" % len(image_annotations), file=stderr)


def print_json(data, outfile=stdout, pretty=False):
    if pretty:
        print(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')), file=outfile)
    else:
        print(json.dumps(data, separators=(',', ':')), file=outfile)

def main(args):
    class_names = {}
    images = []
    categories = []
    annotations = []
    if args.verbose:
        print(args, file=stderr)
    for filename in os.listdir(args.xml_dir):
        if filename.lower().endswith(".xml"): 
            file_path = os.path.join(args.xml_dir, filename)
            if args.verbose:
                print(file_path, file=stderr)
            with file(file_path, 'r') as f:
                json_from_labelme(f, args, class_names, images, categories, annotations)
    data_out = {
     "info": {
      "contributor": "ACFR",
      "date_created": "2017-05-01 10:30:00.000000",
      "description": "This is a dataset for training agricultural datasets.",
      "url": "http://www.acfr.usyd.edu.au/",
      "version": "1.0",
      "year": 2017
     },
     "licenses": [
      {
       "id": 1,
       "name": "Attribution-NonCommercial-ShareAlike License",
       "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
      }
     ],
     "images": images,
     "annotations": annotations,
     "categories": categories
    }
    if args.name:
        data_out['info']['name'] = args.name
    print_json(data_out, outfile=stdout, pretty=args.pretty)

def get_args():
    parser = argparse.ArgumentParser(
     description="Create a COCO JSON annotation file from a directory of LabelMe XMLs."
    )
    parser.add_argument('xml_dir', help="the directory to look for the XMLs")
    parser.add_argument('--name', help="The name to identify this dataset by", \
     default=None, type=str)
    parser.add_argument('--no', help="The name of any category to exclude", \
     default=None, type=str)
    parser.add_argument('--pretty', action='store_true', help="pretty print json")
    parser.add_argument('--verbose', action='store_true', help="verbose output to stderr")
    args = parser.parse_args()
    if args.no is not None:
        args.no = args.no.split(',')
        if args.verbose:
            print('Excluding classes: ', file=stderr)
            print(args.no, file=stderr)
    return args

if __name__ == '__main__':
    main(get_args())