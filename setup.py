#!/usr/bin/env python3
from distutils.core import setup

setup(
    name='Abyss Deep Learning',
    version="1.0.0",
    description="""
        Scripts and utilities for deep learning. Contains most coco scripts and extra
        utilities for pre-processing.""",
    author='Abyss Solutions',
    author_email='tech@abysssolutions.com.au',
    url='http://github.com/abyss-solutions/deep-learning',
    packages=[
        'abyss_deep_learning',
        'abyss_deep_learning.base',
        'abyss_deep_learning.keras',
        'abyss_deep_learning.datasets',
        'abyss_deep_learning.datasets.mrcnn',
    ],
    scripts=[
        "applications/abyss-segmentation-train",
        "applications/argparse.bash",
        "applications/coco-calc",
        "applications/coco-check-data-pollution",
        "applications/coco-draw",
        "applications/coco-extract-masks",
        "applications/coco-from-csv",
        "applications/coco-from-images",
        "applications/coco-from-segmentation",
        "applications/coco-from-video",
        "applications/coco-images",
        "applications/coco-merge",
        "applications/coco-metrics",
        "applications/coco-overlay-masks",
        "applications/coco-sample",
        "applications/coco-select",
        "applications/coco-split",
        "applications/coco-stats",
        "applications/coco-to-csv",
        "applications/coco-to-segmentation",
        "applications/coco-to-tfrecord",
        "applications/coco-to-voc",
        "applications/coco-view",
        "applications/coco-viewer",
        "applications/deeplabv3+",
        "applications/keras-graph",
        "applications/labelbox-to-coco",
        "applications/labelme-to-coco",
        "applications/maskrcnn-find-lr",
        "applications/maskrcnn-predict",
        "applications/maskrcnn-test",
        "applications/maskrcnn-trainval",
        "applications/retinanet-predict",
        "applications/tf-segmentation-predict",
        ],
    data_files=[
        ('etc/abyss/ml/segmentation/training',
            ['etc/abyss/ml/segmentation/training/config.json']),
    ], 
    install_requires=['numpy', 'pillow-simd', 'pandas', 'simplification', 'PyYAML', 'opencv-python', 'scipy', 'scikit-image']
)
