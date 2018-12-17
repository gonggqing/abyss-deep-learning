#!/usr/bin/env python
from distutils.core import setup

setup(
    name='Abyss Deep Learning',
    version="100",
    description='Abyss Deep Learning',
    author='Abyss Solutions',
    author_email='tech@abysssolutions.com.au',
    url='http://github.com/abyss-solutions/deep-learning',
    packages=[
        'abyss_deep_learning',
        'abyss_deep_learning.base',
        'abyss_deep_learning.keras',
        'abyss_deep_learning.datasets',
    ],
    package_data={
        # 'abyss_deep_learning': ["third-party"]
    },
    scripts=[
        "applications/coco-calc-masks",
        "applications/coco-caption-count",
        "applications/coco-check-data-pollution",
        "applications/coco-extract-masks",
        "applications/coco-grep",
        "applications/coco-from-video",
        "applications/coco-merge",
        "applications/coco-repath",
        "applications/coco-split",
        "applications/coco-sample",
        "applications/coco-to-csv",
        "applications/labelbox-to-coco",
        "applications/coco-sample",
        "applications/coco-viewer",
        "applications/image-dirs-to-coco",
        "applications/keras-graph",
        "applications/labelme-to-coco",
        "applications/maskrcnn-find-lr",
        "applications/maskrcnn-predict",
        "applications/maskrcnn-test",
        "applications/maskrcnn-trainval",
        "applications/coco-filter"],
)
