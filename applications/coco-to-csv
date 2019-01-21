#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import sys
from contextlib import redirect_stdout

from operator import itemgetter
from pycocotools.coco import COCO

DESCRIPTION = \
    """
Convert COCO json file and output csv formatted COCO data.
"""


def main(args=None):
    coco = load_coco(sys.stdin.read())
    section = coco.dataset[args.section]
    csv_writer = csv.writer(sys.stdout)

    # Find bbox index in list
    try:
        bbox_index = args.fields.index('bbox')
    except ValueError:
        bbox_index = None

    # Find image id and/or category id index in list
    if args.id_to_name and args.section == 'annotations':
        try:
            image_index = args.fields.index('image_id')
        except ValueError:
            image_index = None

        try:
            category_index = args.fields.index('category_id')
        except ValueError:
            category_index = None
    else:
        image_index = category_index = None

    if args.index_from is not None:
        if args.section == 'annotations':
            key = 'category_id'
        elif args.section == 'categories':
            key = 'id'
        else:
            key = None

        try:
            key_index = args.fields.index(key)
        except ValueError:
            key_index = None
        except TypeError:
            key_index = None

        min_value = min(coco.getCatIds())
        offset = args.index_from - min_value

    # Unpack info or license section
    if isinstance(section, dict):
        # Print headers
        if args.header:
            headers = section.keys()
            if args.fields:
                headers = [header for header in args.fields if header in section]
                non_headers = list(set(args.fields) - set(headers))
                if not headers:
                    die('cannot find keys {fields} in dict {section}'.format(fields=args.fields, section=args.section))
                if non_headers:
                    say('cannot find keys {fields} in dict {section}'.format(fields=non_headers,
                                                                             section=args.section),
                        verbose=args.verbose)
            csv_writer.writerow(headers)

        # Print values
        if args.fields:
            fields = [field for field in args.fields if field in section]
            if fields:
                values = [section[field] for field in fields]
                non_values = list(set(args.fields) - set(fields))
                if non_values:
                    say('cannot find keys {fields} in dict {section}'.format(fields=non_values,
                                                                             section=args.section),
                        verbose=args.verbose)
                if values:
                    csv_writer.writerow(values)
                else:
                    die('cannot find keys {fields} in dict {section}'.format(fields=values, section=args.section))
        else:
            csv_writer.writerow(section.values())

    # Unpack images, annotations or categories section
    elif isinstance(section, list):
        # Print headers
        if args.header:
            if args.fields:
                headers = [field for field in args.fields if field in section[0]]
                non_headers = list(set(args.fields) - set(headers))
                if not headers:
                    die('cannot find keys {fields} in dict {section}'.format(fields=args.fields, section=args.section))
                if non_headers:
                    say('cannot find keys {fields} in dict {section}'.format(fields=non_headers, section=args.section),
                        verbose=True)
            else:
                headers = section[0].keys()

            if bbox_index is not None:
                headers[bbox_index] = ['x', 'y', 'width', 'height'] if args.bbox_position == 'relative' else ['x1',
                                                                                                              'y1',
                                                                                                              'x2',
                                                                                                              'y2']

            if image_index is not None:
                headers[image_index] = 'file_name'

            if category_index is not None:
                headers[category_index] = 'name'
            csv_writer.writerow(flatten(headers))

        for entry in section:
            if args.fields:
                fields = [field for field in args.fields if field in entry]
                non_fields = list(set(args.fields) - set(fields))
                if fields:
                    values = [entry[field] for field in fields]
                    if bbox_index is not None:
                        x, y, width, height = bbox = values[bbox_index]
                        if args.bbox_position == 'absolute':
                            bbox[2] = x + width
                            bbox[3] = y + height
                        values[bbox_index] = bbox

                    if image_index is not None:
                        values[image_index] = coco.loadImgs(values[image_index]).pop()['file_name']

                    if category_index is not None:
                        values[category_index] = coco.loadCats(values[category_index]).pop()['name']

                    if args.index_from is not None and key_index is not None and category_index is None:
                        values[key_index] += offset

                    csv_writer.writerow(flatten(values))
                else:
                    die('cannot find keys {fields} in dict {section}'.format(fields=args.fields, section=args.section))

                if non_fields:
                    say('cannot find keys {fields} in dict {section}'.format(fields=non_fields,
                                                                             section=args.section),
                        verbose=args.verbose)
            else:
                csv_writer.writerow(entry.values())
    sys.exit()


def say(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, file=sys.stderr, **kwargs)


def die(*args, **kwargs):
    say(*args, ': quitting', verbose=True, **kwargs)
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('section',
                        nargs='?',
                        default='annotations',
                        choices=['annotations', 'categories', 'info', 'images', 'licenses'],
                        help="General fields to extract from COCO json to csv",
                        )
    parser.add_argument('-f', '--fields', type=str, help="Comma separated values of COCO fields to extract from the "
                                                         "json file and export to a csv format")
    parser.add_argument('-w', '--what', choices=['retinanet'], help="Format the csv output to be compatible with one "
                                                                    "of the supported options")
    parser.add_argument('-p', '--bbox-position', type=str, default='relative', choices=['absolute', 'relative'],
                        help="Output values of bbox as either x1,y1,x2,y2 or x,y,width,height")
    parser.add_argument('-i', '--index-from', type=int, default=0, help="Change category indexing values to start "
                                                                        "from specified value")
    parser.add_argument('-c', '--id-to-name', action='store_true', help="Convert annotation ids for image id and "
                                                                        "category id to respective names")
    parser.add_argument('-head', '--header', action='store_true', help="The first row of the csv output will contain "
                                                                       "the field names")
    parser.add_argument('-v', '--verbose', action='store_true', help="More output to stderr")
    args = parser.parse_args()
    # Attribute Error is raised if split is called on None type
    # Handles case if fields has no argument given
    if args.what == 'retinanet':
        if args.section == 'annotations':
            args.fields = 'image_id,bbox,category_id'
            args.id_to_name = True
            args.bbox_position = 'absolute'
        elif args.section == 'categories':
            args.fields = 'name,id'

    try:
        args.fields = list(filter(None, args.fields.split(',')))
    except AttributeError:
        pass

    return args


def flatten(l):
    result = []
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


class Verbose:
    @staticmethod
    def write(line):
        line = line.strip()
        if line:
            say(line)


def load_coco(json_buffer):
    with redirect_stdout(Verbose):
        coco = COCO()
        coco.dataset = json.loads(json_buffer)
        coco.createIndex()
    return coco


if __name__ == '__main__':
    main(args=get_args())
