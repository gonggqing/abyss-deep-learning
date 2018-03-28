'''
Classes to handle COCO datasets
@author: Steven Potiris <spotiris@gmail.com>
'''
from __future__ import print_function
import sys
import os
import json
from random import shuffle
import bidict
import numpy as np
from pycocotools import mask as coco_mask

def next_smallest_free_id(id_list):
    return int(np.min([i for i in np.arange(1, len(id_list) + 2) if i not in id_list]))

class CocoDataset(object):
    '''Class to load, manipulate and save COCO datasets'''
    class Utils(object):
        '''Helper functions'''
        @staticmethod
        def polygon_from_bbox(bbox):
            return np.array([
                bbox[0], bbox[1],
                bbox[0] + bbox[2], bbox[1],
                bbox[0] + bbox[2], bbox[1] + bbox[3],
                bbox[0], bbox[1] + bbox[3],
                bbox[0], bbox[1]
            ]).tolist()

        @staticmethod
        def bbox_from_polygon2D(polygon):
            'return the bounding box of a given polygon'
            min_x, min_y = np.min(polygon, axis=0)
            max_x, max_y = np.max(polygon, axis=0)
            return np.array([min_x, min_y, max_x - min_x, max_y - min_y]).tolist()

        @staticmethod
        def bbox_from_points(polygon):
            'return the bounding box of a given set of points [Nx2]'
            min_x, min_y = np.min(polygon, axis=0)
            max_x, max_y = np.max(polygon, axis=0)
            return np.array([min_x, min_y, max_x - min_x, max_y - min_y]).tolist()

        @staticmethod
        def area_polygon(polygon):
            pts_x = [i[0] for i in polygon]
            pts_y = [i[1] for i in polygon]
            return 0.5 * np.abs(np.dot(pts_x, np.roll(pts_y, 1)) - np.dot(pts_y, np.roll(pts_x, 1)))

    @staticmethod
    def from_COCO(coco, image_dir=None, limit_images=None):
        dataset = CocoDataset()
        for category in coco.loadCats(ids=coco.getCatIds()):
            dataset.add_category(
                category['name'], id_number=int(category['id']), supercategory=category['supercategory']
            )
        images = coco.loadImgs(ids=coco.getImgIds())
        shuffle(images)
        for i, image in enumerate(images):
            if limit_images is None or i < limit_images:
                path = image['path'] if 'path' in image else os.path.join(image_dir, image['file_name'])
                dataset.add_image(
                    (image['height'], image['width']), image['file_name'], image['flickr_url'],
                    force_id=int(image['id']), path=path
                )
        image_ids = [image['id'] for image in dataset.images]
        for annotation in coco.loadAnns(coco.getAnnIds()):
            if int(annotation['image_id']) in image_ids:
                dataset.add_annotation(
                    int(annotation['image_id']),
                    int(annotation['category_id']),
                    annotation['segmentation']
                )
        return dataset

    @staticmethod
    def get_label_type(segm):
        if isinstance(segm, dict): # mask being read from COCO json
            return 'coco_mask'
        if isinstance(segm, list) and len(segm) == 4:
            return 'bbox'
        if isinstance(segm, list):
            return 'poly'
        if isinstance(segm, np.ndarray) and segm.ndim == 2:
            return 'mask'
        print("DEBUG INFO:")
        print(type(segm))
        if isinstance(segm, np.ndarray):
            print(segm.shape)
        raise RuntimeError("Unknown label type")

    @staticmethod
    def serialize_label(label):
        '''convert whatever label type into a polygon '''
        label_type = CocoDataset.get_label_type(label)
        segm, area, bbox = None, None, None
        if label_type == 'coco_mask':
            # raise NotImplementedError("Haven't done coco_mask -> poly yet")
            bbox = coco_mask.toBbox(label).tolist()
            area = int(coco_mask.area(label))
            segm = label
            if not isinstance(segm["counts"], str):
                # TODO find out why this doesnt work on coco-calc-masks
                segm["counts"] = segm["counts"].decode("utf-8")
            # segm = [CocoDataset.Utils.polygon_from_bbox(bbox)]
        elif label_type == 'bbox':
            bbox = label if isinstance(bbox, list) else label.tolist()
            area = label[2] * label[3]
            segm = [CocoDataset.Utils.polygon_from_bbox(bbox)]
        elif label_type == 'poly':
            points = np.array(label).reshape((-1, 2))
            bbox = CocoDataset.Utils.bbox_from_points(points)
            area = CocoDataset.Utils.area_polygon(points)
            segm = [points.ravel().tolist()]
            # import matplotlib.pyplot as plt
            # plt.plot(points[:, 0], points[:, 1], '.')
            # plt.show()
        elif label_type == 'mask': # No conversion to poly
            # raise NotImplementedError("Haven't done mask -> poly yet")
            segm = coco_mask.encode(np.asfortranarray(label.astype(np.uint8)))
            segm["counts"] = segm["counts"].decode("utf-8")
            area = int(coco_mask.area(segm))
            bbox = coco_mask.toBbox(segm).tolist()
        return (segm, bbox, area)

    def __init__(self, name=None, is_pretty=False, verbose=False):
        self.name = name
        self.verbose = verbose
        self.pretty = is_pretty
        self.class_dict = bidict.bidict()
        self.image_ids = set([])
        self.images = []
        self.category_ids = set([])
        self.categories = []
        self.annotations = []
        self.info = {
            "contributor": "ACFR",
            "date_created": "2017-05-01 10:30:00.000000",
            "description": "This is a dataset for training agricultural datasets.",
            "url": "http://www.acfr.usyd.edu.au/",
            "version": "1.0",
            "year": 2017
        }
        if name is not None:
            self.info['name'] = name
        self.licenses = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

    def add_category(self, class_name, id_number=None, supercategory=''):
        if id_number is None:
            id_number = next_smallest_free_id(self.category_ids)
        if (class_name not in self.class_dict) and (id_number not in self.class_dict.inv):
            self.class_dict[class_name] = id_number
            self.categories.append(
                {
                    'name': class_name,
                    'id': int(self.class_dict[class_name]),
                    'supercategory': supercategory
                }
            )
        self.category_ids.add(id_number)
        return id_number

    def add_image(
            self, image_size, filename, url,
            force_id=None, license=0, date_captured=None, path=None,
            **kwargs
        ):
        if force_id in self.image_ids:
            raise Exception("add_image: tried to add an image with force_id and encountered a key clash")
        if date_captured is None:
            date_captured = "1970-01-01 00:00:00"
        id_number = force_id if force_id is not None else next_smallest_free_id(self.image_ids)
        record = {
            "id": int(id_number),
            "width": int(image_size[1]),
            "height": int(image_size[0]),
            "file_name": str(filename),
            "license": int(license),
            "flickr_url": str(url),
            "coco_url": str(url),
            "date_captured": date_captured
        }
        record.update(kwargs)
        if path is not None:
            record['path'] = path
        self.images.append(record)
        self.image_ids.add(id_number)
        return id_number

    def add_annotation(self, image_id, category_id, segm, other=None):
        if category_id not in self.class_dict.inv:
            raise Exception("no category id exists {}".format(category_id))
        (segm, bbox, area) = CocoDataset.serialize_label(segm)
            # segm = [[j for i in polygon for j in i]]
        if area > 1:
            annotation_id = next_smallest_free_id([ann['id'] for ann in self.annotations])
            annotation = {
                "area": float(area),
                "bbox": bbox if isinstance(bbox, list) else bbox.tolist(),
                "category_id": int(category_id),
                "id": int(annotation_id),
                "image_id": int(image_id),
                "iscrowd": 0,
                "segmentation": segm
            }
            if other is not None:
                annotation.update(other)
            self.annotations.append(annotation)
        else:
            if self.verbose:
                print(
                    "Warning: Skipped annotation on image {:d} category {:d} due to area <= 1"
                    .format(image_id, category_id)
                )

    def split(self, splits, verbose=False):
        image_set = set([img['id'] for img in self.images])
        datasets = []
        for split in splits:
            num_choices = int(round(len(self.images) * split))
            if verbose:
                print("{:d} images, choosing {:d}".format(len(image_set), num_choices), file=sys.stderr)
            split_ids = np.random.choice(
                list(image_set), num_choices, replace=False
            ).tolist()
            image_set -= set(split_ids)
            dataset = CocoDataset(is_pretty=self.pretty)
            for category in self.categories:
                dataset.add_category(category['name'], int(category['id']), category['supercategory'])
            dataset.images = [
                image for image in self.images \
                if image['id'] in split_ids
            ]
            dataset.annotations = [
                annotation for annotation in self.annotations \
                if annotation['image_id'] in split_ids
            ]
            datasets.append(dataset)
        return datasets

    def save(self, path):
        '''Save the COCO dataset to a file'''
        with open(path, 'w') as handle:
            return handle.write(str(self))

    def __add__(self, other):
        if not isinstance(other, CocoDataset):
            raise Exception("Can only add together two CocoDataset objects")
        if self.name and other.name:
            self.name += "_" + other.name
        elif not self.name and other.name:
            self.name = "_" + other.name
        ## Merge categories
        cat_map = {} #maps foreign to new category IDs
        cat_ids = set([cat['id'] for cat in self.categories])
        for cat in self.categories:
            for cat_foreign in other.categories:
                #TODO check supercategories
                if cat['name'] == cat_foreign['name']:
                    # Merge these IDs, local ID will take precedence, foreign will be lost
                    id_val = cat['id']
                elif cat['id'] == cat_foreign['id']:
                    # Allocate a new ID later
                    id_val = next_smallest_free_id(cat_ids)
                    self.add_category(cat_foreign['name'], id_val, supercategory=cat_foreign['supercategory'])
                else:
                    # No clashes or equivalence, add new category with given ID
                    id_val = cat_foreign['id']
                    self.add_category(
                        cat_foreign['name'],
                        cat_foreign['id'],
                        supercategory=cat_foreign['supercategory']
                    )
                cat_map[cat_foreign['id']] = id_val
                cat_ids.add(id_val)
        # Merge images
        # Never reuse image ids... refer to by filename/path if you need a unique key
        img_map = {}
        img_ids = set([img['id'] for img in self.images])
        # print("Image map:")
        # print(image_map)
        print("Category map:", file=sys.stderr)
        print(cat_map, file=sys.stderr)
        for image in other.images:
            id_val = next_smallest_free_id(img_ids)
            img_map[image['id']] = id_val
            path = image['path'] if 'path' in image else None
            self.add_image(
                (image['height'], image['width']),
                image['file_name'], image['flickr_url'],
                force_id=id_val, path=path
            )
            img_ids.add(id_val)
        # Merge annotations, mapping image and category IDs
        for annotation in other.annotations:
            self.add_annotation(
                img_map[annotation['image_id']],
                cat_map[annotation['category_id']],
                annotation['segmentation']
            )
        return self

    def __str__(self):
        data_out = {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        if self.pretty:
            return json.dumps(data_out, sort_keys=True, indent=4, separators=(',', ': '))
        return json.dumps(data_out, sort_keys=True, separators=(',', ':'))


class CocoResults(object):
    def __init__(self, is_pretty=False):
        self.results = []
        self.pretty = is_pretty

    def add(self, obj, guess=None):
        '''guess the result type and add it to the record'''
        if guess == 'bbox':
            self.results.append(obj)

    def add_detection(self, image_id, category_id, score, detection):
        '''takes a bbox, polygon, keypoints or mask
        bbox should be list [x, y, w, h]
        polygon should be list [x1, y1, x2, y2, ...]
        mask should be np.array with ndim == 2
        keypoints should be a list [x1,y1,v1,...,xk,yk,vk]'''
        result = {
            "image_id": int(image_id),
            "category_id": int(category_id),
            "score": float(score)
        }
        if isinstance(detection, np.ndarray) and detection.size % 3 == 0:
            # Keypoints (x, y, v)
            result["keypoints"] = detection
        elif isinstance(detection, np.ndarray) and detection.ndim == 2:
            # mask segmentation
            result["segmentation"] = coco_mask.encode(np.asfortranarray(detection))
            result["segmentation"]["counts"] = result["segmentation"]["counts"].decode('utf-8')
            result["bbox"] = coco_mask.toBbox(result["segmentation"]).tolist()
        else:
            if len(detection) == 4: #bbox
                result["bbox"] = detection.tolist() \
                    if isinstance(detection, np.ndarray) \
                    else detection
                result["segmentation"] = \
                 [CocoDataset.Utils.polygon_from_bbox(result["bbox"])]
            else: #polygon
                poly = np.array(detection).reshape((-1, 2)).tolist()
                result["bbox"] = CocoDataset.Utils.bbox_from_polygon2D(poly)
                result["segmentation"] = np.array(poly).reshape((1, -1)).tolist()
        self.results.append(result)

    def add_detection_keypoints(self, image_id, category_id, score, keypoints):
        self.results.append(
            {
                "image_id" : image_id,
                "category_id" : category_id,
                "keypoints" : keypoints, #[x1,y1,v1,...,xk,yk,vk],
                "score" : score,
            }
        )
    # def add_caption(self, image_id, category_id, score, keypoints):
    #     self.results.append(
    #         {
    #             "image_id" : int(image_id),
    #             "caption" : str,
    #         }
    #     )

    def __str__(self):
        if self.pretty:
            return json.dumps(self.results, sort_keys=True, indent=4, separators=(',', ': '))
        return json.dumps(self.results, separators=(',', ':'))
