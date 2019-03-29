from abyss_deep_learning.datasets.coco import *
from mrcnn.utils import Dataset as MatterportMrcnnDataset

class InstSegDataset(CocoDataset, ImageDatatype, MatterportMrcnnDataset):
    '''
    NOTE: This dataset scales between +/- 127.5 rather than +/- 1.0.
    '''

    def __init__(self, json_path, config, **kwargs):
        import importlib
        from bidict import bidict
        CocoDataset.__init__(self, json_path, **kwargs)
        ImageDatatype.__init__(self, self.coco, **kwargs)
        MatterportMrcnnDataset.__init__(self, class_map=None)
        self.load_coco(image_dir=self.image_dir)
        self.prepare(class_map=None)
        self.internal_to_original_ids = {
            img_id: img_info['id']
            for img_id, img_info in enumerate(self.image_info)}
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.class_map = bidict({
            cat['id']: cat['name'] for cat in
            sorted(cats, key=lambda x: x['id'])})
        self.class_names = list(self.class_map.values())

        if isinstance(config, str):
            import importlib.util
            spec = importlib.util.spec_from_file_location("mrcnn.config_file", config)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.config = module.Config()
        else:
            self.config = config

    def load_coco(self, image_dir=None, class_ids=None, class_map=None):
        """Load a subset of the COCO dataset.
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        """
        coco = self.coco

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for idx in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[idx])))
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
            if 'path' in coco.imgs[i]:
                path = coco.imgs[i]['path']
            else:
                path = os.path.join(image_dir, coco.imgs[i]['file_name'])
            self.add_image(
                "coco", image_id=i,
                path=path,
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

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
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            raise NotImplementedError("load_mask from COCO dataset but source is not.")

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        # Call super class to return an empty mask
        return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
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

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def mrcnn_generator(
            self, shuffle=True, augment=False, augmentation=None,
            random_rois=0, batch_size=1, detection_targets=False, no_augmentation_sources=None):
        from mrcnn.model import data_generator
        return data_generator(
            self, self.config, shuffle, augment, augmentation,
            random_rois, batch_size, detection_targets, no_augmentation_sources)

    def generator(self):
        raise NotImplementedError()
