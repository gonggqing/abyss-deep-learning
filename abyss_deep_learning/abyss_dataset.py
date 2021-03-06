"""
abyss-dataset
"""

from itertools import cycle
from random import shuffle
from sys import stderr
import os
import time

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

import abyss_maskrcnn.utils as utils

############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    '''Legacy class, see abyss_deep_learning.datasets.base and
    abyss_deep_learning.keras classification/segmentation/detection.'''
    def __init__(self, *args, **kwargs):
        '''(Deprecated) Call load_coco'''
        self.data = []
        super(CocoDataset, self).__init__(*args, **kwargs)

    def load_coco(
        self, dataset_path, image_dir=None,
        class_ids=None,
        preload=False, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_path: Thepath to the COCO dataset JSON.
        image_dir: The base path of the RGB images, if None then look for 'path' key in JSON
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        tile: None, or a tuple (h, w) of the tile size. Use to separate large images into tiles.
        return_coco: If True, returns the COCO object.
        auto_download: REMOVED
        """

        # if auto_download is True:
        #     self.auto_download(os.path.dirname(dataset_path), subset, year)
        self.coco = COCO(dataset_path)

        # All images or a subset?
        if class_ids:
            image_ids = []
            for class_id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[class_id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            class_ids = sorted(self.coco.getCatIds())
            image_ids = list(self.coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, self.coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            path = os.path.join(
                image_dir, coco.imgs[i]['file_name']) if image_dir != None else coco.imgs[i]['path']
            self.add_image(
                "coco", image_id=i,
                path=path,
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if preload:
            self.preload_images()
        return self.coco

    def preload_images(self):
        print("Preloading images.", file=stderr)
        self.data = {
            image_id: (self.load_image(image_id), self.load_mask(image_id))
            for image_id in self.image_ids}

    def load_image(self, image_id):
        if self.data:
            return self.data[image_id][0]
        return super(CocoDataset, self).load_image(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        return super(CocoDataset, self).image_reference(image_id)

    def apply(self, func_input, func_target=None, imgIds=[]):
        '''Applies functions to input and target, if the DB is preloaded'''
        assert self.data, "apply() only works on preloaded images"
        if not func_target:
            func_target = lambda x: x
        imgIds = imgIds or self.image_ids
        for image_id in imgIds:
            self.data[image_id] = (func_input(image_id[0]), func_target(image_id[1]))



############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        time_start = time.time()
        result = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - time_start)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           result["rois"], result["class_ids"],
                                           result["scores"], result["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    coco_eval = COCOeval(coco, coco_results, eval_type)
    coco_eval.params.imgIds = coco_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
