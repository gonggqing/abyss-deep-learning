#!/bin/bash
SCRIPTNAME="$(basename $0)"
ARGPARSE_DESCRIPTION="Validate models
"
source "$HOME/src/abyss/deep-learning/applications/argparse.bash" || exit 1 # This imports functions info, error, optional_flag, bool_flag, assert
argparse "$@" <<EOF || exit 1
parser.add_argument('source_dir', help='source_dir')
parser.add_argument('model_name_dir', help='model_name_dir')
parser.add_argument('--scores', help='comma separated minimum scores', default='0.1,0.5')
parser.add_argument('--ious', help='comma separated ious', default='0.3,0.5')
parser.add_argument('--run-inference', help='0 to disable inference (use existing)', default='1')
EOF

OUTPUT_DIR=$MODEL_NAME_DIR
MODEL_DIR=$SOURCE_DIR/$MODEL_NAME_DIR

WORKFLOW=$OUTPUT_DIR/validation.multiscale/workflow
VALIDATION=$OUTPUT_DIR/validation.multiscale

mkdir -p $WORKFLOW
mkdir -p $VALIDATION

#input files
TRUTH_JSON=/mnt/pond/processed/industry-data/bhp/201903.fm-pilot/object-detection/labels/current-best-labels.split/val_full_image_flat.json
H5=$MODEL_DIR/model/model.h5

#generated files
PRED_CSV=$WORKFLOW/pred.csv
PRED_JSON=$WORKFLOW/pred.json
IMG_LIST=$WORKFLOW/img_list.txt

echo IOU $IOUS
echo SCORES $SCORES

RUN_INFERENCE=1

#generate img list from json
cat $TRUTH_JSON | jq .images[].path | sed "s#\"##g" > $IMG_LIST

# generate predictions
if [ "$RUN_INFERENCE" == "1" ]
then
  retinanet-predict $IMG_LIST --weights $H5 --convert-model --batch-size 5 --image-min-side 1000 --image-max-side 1000 --filter-overlaps --remove --verbose > $WORKFLOW/pred_scale.csv
  
  rm $WORKFLOW/pred_stride.csv
  touch $WORKFLOW/pred_stride.csv
  IFS=$'\n'
  for i in $( cat $IMG_LIST );
  do
	echo $i
  	retinanet-predict <( echo $i ) --weights $H5 --convert-model --batch-size 5 --image-min-side 500 --image-max-side 500 --tile-size "500,500" --stride "256,256" --filter-overlaps --remove --verbose >> $WORKFLOW/pred_stride.csv
  done
fi

# generated for visualisation/comparison of scales only
#echo create coco json detections on scaled images
#cat $WORKFLOW/pred_scale.csv | coco-from-csv --fields path,annotation_id,x1,y1,x2,y2,score,category_id --map <( echo Flange,0 ) | coco-calc assign-new-image-ids coco $TRUTH_JSON > $WORKFLOW/pred_scale.json
#echo create coco json detections on strided patches
#cat $WORKFLOW/pred_stride.csv | coco-from-csv --fields path,annotation_id,x1,y1,x2,y2,score,category_id --map <( echo Flange,0 ) | coco-calc assign-new-image-ids coco $TRUTH_JSON > $WORKFLOW/prev_stride.json

# combine scale and stride inference
cat $WORKFLOW/pred_scale.csv > "$PRED_CSV"
cat $WORKFLOW/pred_stride.csv >> "$PRED_CSV"

# generate pred json - ensure image id's match TRUTH_JSON
cat "$PRED_CSV" | coco-from-csv --fields path,annotation_id,x1,y1,x2,y2,score,category_id --map <( echo Flange,0 ) | coco-calc assign-new-image-ids coco $TRUTH_JSON > "$PRED_JSON"

IFS=',' read -ra thresh_vals <<< "$SCORES"

for thresh_val in ${thresh_vals[@]};
do
    PRED=$WORKFLOW/pred_$thresh_val.json
    echo "$PRED"
    cat $PRED_JSON | coco-select annotations "a.score > $thresh_val" | coco-calc nms > $PRED

    IFS=',' read -ra iou_vals <<< "$IOUS"
    for iou_val in ${iou_vals[@]};
    do
        
        PAIR=$VALIDATION/score_${thresh_val}_iou_${iou_val}
        mkdir -p $PAIR

        echo Generate coco annotations labeled as TP, FP, TN...
        #cat $PRED | coco-metrics tfpn --truth $TRUTH_JSON --bbox --score-threshold $thresh_val --iou-threshold $iou_val > $PAIR/tfpn.json
        echo cat $PRED \| coco-metrics tfpn --truth $TRUTH_JSON --bbox --score-threshold $thresh_val --iou-threshold $iou_val \> $PAIR/tfpn.json

        echo Generate confusion matrix...
        cat $PRED  |  coco-metrics confusion-matrix --bbox --score-threshold $thresh_val --iou-threshold $iou_val --truth $TRUTH_JSON --plot --save-figure > $PAIR/confusion_matrix.csv && mv confusion_matrix.png $PAIR/confusion_matrix.png 

        echo Generate confusion matrix normalized...
        cat $PRED  | coco-metrics confusion-matrix --bbox --score-threshold $thresh_val --iou-threshold $iou_val --normalize --truth $TRUTH_JSON --plot --save-figure > $PAIR/confusion_matrix_norm.csv && mv confusion_matrix_normalized.png $PAIR/confusion_matrix_normalized.png

        #echo Generate per image confusion matrix...
        #cat $PRED | coco-metrics confusion-matrix --bbox --score-threshold $thresh_val --iou-threshold $iou_val --csv-output-per-image --truth $TRUTH_JSON > $PAIR/confusion_matrix_per_image.csv

        #echo Generate per image confusion matrix normalized...
        #cat $PRED | coco-metrics confusion-matrix --bbox --score-threshold $thresh_val --iou-threshold $iou_val --csv-output-per-image --normalize --truth $TRUTH_JSON > $PAIR/confusion_matrix_per_image_normalised_$thresh_val.csv

        echo Generate class confusion...
        cat $PRED | coco-metrics confusion --bbox --truth $TRUTH_JSON --score-threshold $thresh_val --iou-threshold $iou_val > $PAIR/confusion.json
    done
    
done

echo Done!

