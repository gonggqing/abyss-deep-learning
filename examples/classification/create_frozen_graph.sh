#!/usr/bin/env bash

usage()
{
    echo "usage: ./create_frozen_graph.sh keras_image_classifier_model.h5 output_frozen_graph.pb"
}

# Check there is two input arguments
if [ $# -ne 2 ]; then
    usage
    exit
fi


# Check the keras model exists
if [ ! -f "$1" ]; then
    echo "The keras model $1 does not exist"
    exit
fi

# If TENSORFLOW_ROOTDIR doesn't exist - set it
if [[ -z "${TENSORFLOW_ROOTDIR}" ]]; then
    TENSORFLOW_ROOTDIR=/usr/local/lib/python3.5/dist-packages/tensorflow
fi

# Setup the path to the frozen graph script
FREEZE_GRAPH_SCRIPT=$TENSORFLOW_ROOTDIR/python/tools/freeze_graph.py

# Loads the ImageClassifier model (using its custom load function) and exports the session to /tmp/image_classifier.ckpt .
# Also exports the output node name to /tmp/output_node_name.tx
python3 export_tf_checkpoint.py $1 /tmp/image_classifier.ckpt > /tmp/output_node_name.txt

# The freeze_graph.py script needs to be in the same directory according to the examples.
# Copy it to the current directory
cp $FREEZE_GRAPH_SCRIPT .


# Creates the frozen graph
python freeze_graph.py \
    --input_meta_graph=/tmp/image_classifier.ckpt.meta \
    --input_checkpoint=/tmp/image_classifier.ckpt \
    --output_graph=$2 \
    --output_node_names=$(cat /tmp/output_node_name.txt) \
    --input_binary=true


# Clean up
rm freeze_graph.py
rm /tmp/output_node_name.txt
rm /tmp/image_classifier.ckpt*

