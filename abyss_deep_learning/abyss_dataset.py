"""
abyss-dataset
"""

import os
import time
from sys import stderr

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import abyss_maskrcnn.utils as utils

############################################################
#  Dataset
############################################################


class CocoDataset(utils.Dataset):
    def __init__(self, *args, **kwargs):
        self.data = []
        super(CocoDataset, self).__init__(*args, **kwargs)

    def load_coco(self, dataset_path, image_dir=None, class_ids=None, preload=False, return_coco=False):
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
        coco = COCO(dataset_path)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for class_id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[class_id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

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
        return coco

    def preload_images(self):
        print("Preloading images.", file=stderr)
        self.data = {
            image_id: (self.load_image(image_id), self.load_mask(image_id))
            for image_id in self.image_ids}

    def load_image(self, image_id):
        if self.data:
            return self.data[image_id][0]
        return super(CocoDataset, self).load_image(image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        if self.data:
            return self.data[image_id][1]
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                mask = ann_to_mask(
                    annotation, image_info["height"], image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if mask.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, ann_to_mask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if mask.shape[0] != image_info["height"] or mask.shape[1] != image_info["width"]:
                        mask = np.ones(
                            [image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(mask)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        # Otherwise call super class to return an empty mask
        return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        return super(CocoDataset, self).image_reference(image_id)

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


def ann_to_mask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = ann_rle_encode(ann, height, width)
    mask = maskUtils.decode(rle)
    return mask


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
