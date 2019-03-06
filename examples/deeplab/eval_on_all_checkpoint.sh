#!/bin/bash
LOGDIR=/mnt/ssd1/processed/industry-data/anadarko/gunnison/models/segmentation/deeplab/20190305.2_class_noBN
DEEPLAB_DIR=/home/users/spo/src/tensorflow_models/research/deeplab


# Copy the checkpoint to a checkpoint_temp file
cp $LOGDIR/checkpoint $LOGDIR/checkpoint_temp
chmod 777 $LOGDIR/checkpoint_temp
chmod 777 $LOGDIR/checkpoint

CSV_RESULTS=$LOGDIR/eval_results.csv
rm $CSV_RESULTS
echo "StepNumber,mIOU" > $CSV_RESULTS
chmod 777 $CSV_RESULTS

for FILE in $(find $LOGDIR/checkpoints -name *model.ckpt*); do
  if [[ $FILE = *"model.ckpt"*".index" ]]; then
    STEP_NUM=$(echo $(basename $FILE) | sed 's/model.ckpt-//g' | sed 's/.index//g')
    MODEL_PATH=$LOGDIR/checkpoints/model.ckpt-$STEP_NUM
    echo "model_checkpoint_path: \"$MODEL_PATH\" "> $LOGDIR/checkpoint
    CUDA_VISIBLE_DEVICES=0,0 \
    bash ~/src/abyss/deep-learning/applications/deeplabv3+ \
     --deeplab-dir $DEEPLAB_DIR \
     eval \
     $LOGDIR/tfrecord \
     $LOGDIR 2>&1 | tee $LOGDIR/tmp.txt

     cat $LOGDIR/tmp.txt | grep "model.ckpt"
     MIOU=$(cat $LOGDIR/tmp.txt | grep miou | sed 's/miou_1.0//g' | sed 's/\[//g' | sed 's/\]//g')
     echo "$STEP_NUM,$MIOU" >> $CSV_RESULTS
     echo "# # # # # # # # # #"
     cat $CSV_RESULTS
     rm $LOGDIR/tmp.txt
     echo "# # # # # # # # # #"
  fi
done
echo "- - - - - - - -" 
cat $CSV_RESULTS | sort -n > $CSV_RESULTS
cat $CSV_RESULTS

gnuplot -p << EOF
set datafile separator ","
set xlabel "StepNum"
set ylabel "mIOU"
set title "StepNum v mIOU"
plot "$CSV_RESULTS" using 1:2 with linespoints
EOF


