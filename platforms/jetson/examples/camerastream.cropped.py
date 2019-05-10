#!/usr/bin/env python3
import cv2
import subprocess
import shlex
import numpy as np
import sys
#subprocess.call(shlex.split('./test.sh param1 param2'))

cap = cv2.VideoCapture(0)
count = 0
while True:
    ret, img = cap.read()
    #img = cv2.resize(img, (513,513))
    img = img[300:813,300:813]
    
    cv2.imshow("you",img)
    #print(img.dtype, img.shape, img.size)
    sys.stdout.buffer.write(img.tobytes())
    sys.stdout.flush()
    #print("Flushed")
    cv2.waitKey(1)
    #count += 1

cap.release()
#tensor-cat --frozen-graph "$frozen_graph" \
#                        --input "$options_input" \
#                        --input-shape "$options_input_shape" \
#                        --output "$options_output" \
#                        $output_max_channels_option \
#                        $verbose_option \
#                        $@
