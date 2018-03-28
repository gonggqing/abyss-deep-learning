#!/usr/bin/env python
from distutils.core import setup

#This is a list of files to install, and where
#(relative to the 'root' dir, where setup.py is)
#You could be more specific.

setup(
	name='Abyss Deep Learning',
	version="100",
	description='Abyss Deep Learning',
	author='Abyss Solutions',
	author_email='tech@abysssolutions.com.au',
	url='http://github.com/abyss-solutions/deep-learning',
	packages=['abyss_deep_learning', 'abyss_maskrcnn'],
	package_data={
		# 'abyss_deep_learning': ["third-party"]
	},
	scripts=[
		"applications/coco-calc-masks",
		"applications/coco-check-data-pollution",
		"applications/coco-extract-masks",
		"applications/coco-merge",
		"applications/coco-split",
		"applications/coco-to-csv",
		"applications/coco-viewer",
		"applications/image-dirs-to-coco",
		"applications/keras-graph",
		"applications/labelme-to-coco",
		"applications/maskrcnn-find-lr",
		"applications/maskrcnn-predict",
		"applications/maskrcnn-test",
		"applications/maskrcnn-trainval"],
) 
