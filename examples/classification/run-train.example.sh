# this file includes good hyperparameter settings for CCTV training. 
# note that step-size in 'lr-schedule-params' is calculated from: number_of_training_examples / batch-size * 6, as recommended on page 3 of the original cyclic-learning paper: https://arxiv.org/pdf/1506.01186.pdf
SCRATCH_DIR=.
python3 ~/src/abyss/deep-learning/examples/classification/train_cctv_classifier.py "$SCRATCH_DIR"/datasets/coco.train.json  \
    --val-coco-path  "$SCRATCH_DIR"/datasets/coco.val.json \
    --scratch-dir "$SCRATCH_DIR"/ \
    --category-map "$SCRATCH_DIR"/fault-detection.category-map.json \
    --batch-size=16 \
    --lr=1e-6 \
    --workers 10\
    --gpus 1 \
    --trains-project cctv-ml-experiments \
    --early-stopping-patience 100\
    --l12-regularisation 0.001,0.001 \
    --optimizer 'sgd' \
    --optimizer-args "{'momentum':0.9}" \
    --epochs 1000 \
    --cache-val \
    --histogram-freq 10\
    --lr-schedule 'cyclic' \
    --lr-schedule-params "{'max_lr':1e-4,'step_size':11250}"\
    --trains-experiment example-CCTV-experiment
