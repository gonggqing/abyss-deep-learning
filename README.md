# deep-learning

## Installation
```bash 
mkdir -p ~/src/abyss
cd ~/src/abyss
git clone https://github.com/abyss-solutions/deep-learning.git
cd deep-learning
./configure.sh
```

## Applications
* labelme-to-coco: Convert labelme dataset into COCO JSON
* coco-extract-masks: Extract masks from COCO JSON to pngs
* coco-viewer: View coco images and annotations
* maskrcnn-trainval: Train a Mask RCNN network with COCO JSON dataset
* maskrcnn-predict: Predict on input images or streams using a trained Mask RCNN model and output labels or overlay

## TODO
* maskrcnn-test: Test a Mask RCNN network with COCO JSON dataset
* maskrcnn-predict
  * Add feature: Take in STDIN cv-cat stream and output prediction cv-cat stream
* Tutorials
