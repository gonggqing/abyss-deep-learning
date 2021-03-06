# Instructions to Train Retinanet

## Installation

### Install tensorflow-gpu natively

...Unless you want to train on CPU...

You also may consider create a docker from tensorflow-gpu docker and add retinanet to it. Installing tensorflow-gpu cuda dependencies natively may be pain due to version discrepancies. 

```
sudo pip3 install tensorflow-gpu
```
Then, install CUDA by following the twenty intuitively obvious commands here: https://www.tensorflow.org/install/gpu 

### Install keras-retinanet

Follow installation instructions here: https://github.com/fizyr/keras-retinanet

Install missing dependencies with sudo pip3 install later. Apart from tensorflow, the rest seems straigthforward.

### Download model
...e.g. from here: https://github.com/fizyr/keras-retinanet/releases

## Create the retinanet csv dataset

Starting from the coco-boxes.json...

```bash
cat coco-boxes.train.json | coco-to-csv annotations -w retinanet > train_annotations.csv
cat coco-boxes.train.json | coco-to-csv categories -w retinanet > class_mapping.csv
cat coco-boxes.val.json | coco-to-csv annotations -w retinanet > val_annotations.csv
```

If you want:
Rename retina-annotations/train/ to train_annotations

## Link to images, masks

```bash
ln -s "/mnt/pond/processed/industry-data/anadarko/gunnison/r2s/sphericals/CD -/cubes_small" images
ln -s "/mnt/pond/processed/industry-data/anadarko/gunnison/r2s/sphericals/CD -/masks_small" masks  # Optional
```

## Run the training script

Need to calculate steps (num_images/batchsize)
```bash
echo "$(cat train_annotations.csv | wc -l) / batchsize" | bc
```

```bash
python3 ~/src/abyss/keras-retinanet/keras_retinanet/bin/train.py --epochs 100 --steps 70 --batch-size 4 \
    --weights /home/users/jsh/scratch/anadarko/object-recognition/coco_weights/resnet50_coco_best_v2.1.0.h5 \
    --backbone resnet50 --image-min-side 1000 --image-max-side 1000 \
    --tensorboard-dir /home/users/jsh/scratch/anadarko/object-recognition/logs \
    --snapshot-path /home/users/jsh/scratch/anadarko/object-recognition/snapshots/ \
    csv /home/users/jsh/data/anadarko/object-recognition/train_annotations.csv \
    /home/users/jsh/data/anadarko/object-recognition/class_mapping.csv \
    --val-annotations /home/users/jsh/data/anadarko/object-recognition/val_annotations.csv
```
