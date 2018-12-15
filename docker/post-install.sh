#!/bin/bash
# This needs to be run after docker install
# TODO: Find out why this doesn't work in setup.sh

cd $HOME/src/abyss/crfasrnn_keras/src/cpp && \
PATH=$PATH:/usr/local/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 make all && \
cd ../.. && \
pip3 install --user .
