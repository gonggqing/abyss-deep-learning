#!/bin/bash
LOGDIR=/mnt/ssd1/processed/industry-data/anadarko/gunnison/models/segmentation/deeplab/20190305.2_class_noBN
DEEPLAB_DIR=/home/users/spo/src/tensorflow_models/research/deeplab


# Copy the checkpoint to a checkpoint_temp file
cp $LOGDIR/checkpoint $LOGDIR/checkpoint_temp


CSV_RESULTS=$LOGDIR/eval_results.csv
rm $CSV_RESULTS
echo "StepNumber,mIOU" > $CSV_RESULTS

for FILE in $(find $LOGDIR/checkpoints -name *model.ckpt*); do
  if [[ $FILE = *"model.ckpt"*".index" ]]; then
    STEP_NUM=$(echo $(basename $FILE) | sed 's/model.ckpt-//g' | sed 's/.index//g')
    MODEL_PATH="/mnt/ssd1/processed/industry-data/anadarko/gunnison/models/segmentation/deeplab/20190305.2_class_noBN/checkpoints/model.ckpt-$STEP_NUM"
    echo "model_checkpoint_path: \"$MODEL_PATH\" "> $LOGDIR/checkpoint
    CUDA_VISIBLE_DEVICES=0,0 \
    bash ~/src/abyss/deep-learning/applications/deeplabv3+ \
     --deeplab-dir $DEEPLAB_DIR \
     eval \
     $LOGDIR/tfrecord \
     $LOGDIR 2>&1 | tee tmp.txt

     cat tmp.txt | grep "model.ckpt"
     MIOU=$(cat tmp.txt | grep miou | sed 's/miou_1.0//g' | sed 's/\[//g' | sed 's/\]//g')
     echo "$STEP_NUM,$MIOU" >> $CSV_RESULTS
     rm tmp.txt
  fi
done

cat $CSV_RESULTS | sort -n > $CSV_RESULTS
