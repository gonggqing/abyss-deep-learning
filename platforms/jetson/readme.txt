# cv-cat "view" from camera was running into issues where the image would not be displayed correctly

./camerastream.cropped.py | tensor-cat --frozen-graph frozen_inference_graph.pb  --input ImageTensor:0 --input-shape 1,513,513,3 --output SemanticPredictions:0 | ./overlay.py 

# TODO: add --gpu-alloc argument to tensor-cat to allocate a certain % of GPU memory only
# eg. --gpu-alloc 0.5 to allocate only 50% of available memory to be used by Tensorflow

