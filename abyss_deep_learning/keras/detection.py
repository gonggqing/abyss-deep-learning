"""
abyss-dataset
"""

from itertools import cycle
from random import shuffle
from sys import stderr
import os
import time

from pycocotools import mask as maskUtils
# from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from mrcnn.utils import Dataset as MrcnnDataset

from abyss_deep_learning.datasets.base import SupervisedDataset

############################################################
#  Dataset
############################################################
from mrcnn.utils import Dataset as MrcnnDataset

class MaskRcnnDataset(SupervisedDataset, MrcnnDataset):
    def __init__(self, annotation_file, image_dir=None, class_ids=None, preload=False, class_map=None):
        """Load a subset of the COCO dataset.
        dataset_path: Thepath to the COCO dataset JSON.
        image_dir: The base path of the RGB images, if None then look for 'path' key in JSON
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        tile: None, or a tuple (h, w) of the tile size. Use to separate large images into tiles.
        return_coco: If True, returns the COCO object.
        """
        SupervisedDataset.__init__(self, annotation_file, image_dir, class_ids, preload, class_map)
        _image_ids = self._image_ids_orig
        MrcnnDataset.__init__(self, class_map)
        self._image_ids = _image_ids
        
        # Add classes
        for i in self.class_ids:
            self.add_class("coco", i, self.loadCats(i)[0]["name"])

        # Add images
        for i in self.image_ids:
            path = os.path.join(
                image_dir, self.coco.imgs[i]['file_name']) if image_dir != None else self.coco.imgs[i]['path']
            self.add_image(
                "coco", image_id=i,
                path=path,
                width=self.coco.imgs[i]["width"],
                height=self.coco.imgs[i]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(
                    imgIds=[i], catIds=self.class_ids, iscrowd=None)))
        
        self.prepare(class_map)
        self._image_id_map = {self.image_info[i]['id']: i for i in self.image_ids}
    
    def mrcnn_generator(self, config, shuffle=True, augment=False, augmentation=None,
                        random_rois=0, batch_size=1, detection_targets=False, no_augmentation_sources=None):
        return data_generator(
            self, config, shuffle, augment, augmentation,
            random_rois, batch_size, detection_targets, no_augmentation_sources)
    
    def load_targets(self, image_id):
        '''Loads the mask for the given COCO source ID.'''
        return self.__load_mask(self._image_id_map[image_id])

    def __load_mask(self, image_id):
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
            return super().load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                mask = self.coco.annToMask(annotation)
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
        return super().load_mask(image_id)
    
    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        return super().image_reference(image_id)





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
        image_results = build_coco_results(
            dataset, coco_image_ids[i:i + 1],
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


################ Visualisation methods ###########

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] *
            (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    import colorsys
    from random import shuffle
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    shuffle(colors)
    return colors

def label2rgb_instances(masks, image, class_ids=None, scores=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """

    # Number of instances
    N = masks.shape[0]
    print(masks.shape, image.shape)
    if N:
        colors = random_colors(N)
        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]
            if not np.any(masks[..., i]):
                continue
            mask = masks[..., i]
            masked_image = apply_mask(masked_image, mask, color)
        print("N=", N, "masked_image.shape=", masked_image.shape)
        return masked_image.astype(np.uint8)
    return image
