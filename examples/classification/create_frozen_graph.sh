#!/usr/bin/env bash

TFLOW_ROOTDIR=/usr/local/lib/python3.5/dist-packages/tensorflow

FREEZE_GRAPH_SCRIPT=$TFLOW_ROOTDIR/python/tools/freeze_graph.py

python3 export_tf_checkpoint.py $1 /tmp/image_classifier.ckpt > /tmp/output_node_name.txt

cp $FREEZE_GRAPH_SCRIPT .

python freeze_graph.py \
    --input_meta_graph=/tmp/image_classifier.ckpt.meta \
    --input_checkpoint=/tmp/image_classifier.ckpt \
    --output_graph=$2 \
    --output_node_names=$(cat /tmp/output_node_name.txt) \
    --input_binary=true

rm freeze_graph.py