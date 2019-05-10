#!/usr/bin/env python3
import numpy as np
import sys
import cv2

#data = bytearray(513*513*3*8)
shape = (513,513,1)

count = 0
print("Overlay started")
while True:
	#n = proc.stdout.readinto(data)
	#arr = np.frombuffer(data, np.uint8)
	#print(arr.size)
	print("Reading from stdin...")
	#print(count, len(sys.stdin.buffer.read()))
	arr = np.frombuffer(sys.stdin.buffer.read(shape[0]*shape[1]*shape[2]*4), dtype=np.uint32).reshape((513,513))
	arr = arr.astype(np.uint8)
	arr = cv2.normalize(arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	print(arr, arr.shape)
	#if(arr.any()):
	cv2.imshow("Out",arr)
	cv2.waitKey(1)
	#count += 1
print("Overlay done")
