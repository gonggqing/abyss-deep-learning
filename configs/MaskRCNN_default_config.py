import os
import sys
# Note the directory MASK_RCNN_PATH should be exported as e.g. /home/whoever/src/abyss/deep-learning/third-party/Mask_RCNN
sys.path.append(os.environ['MASK_RCNN_PATH'])

from config import Config

class TrainConfig(Config):
    '''See Mask_RCNN/config.py for more parameters'''
    NAME = "default-model" # This is the prefix for any saved models
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # background + 1 classes
    IMAGE_MIN_DIM = 600
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 60
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 5
#     USE_MINI_MASK
#     MASK_SHAPE
    
class InferenceConfig(Config):
    # The following parameters have to match the TrainConfig
    NUM_CLASSES = 1 + 1  # background + 1 classes, has to match above
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # The following parameters do not have to match
    NAME = "default-model"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 600
    IMAGE_MAX_DIM = 1024
    STEPS_PER_EPOCH = 1000
    DETECTION_MIN_CONFIDENCE = 0.99
    DETECTION_NMS_THRESHOLD = 0.3
    POST_NMS_ROIS_INFERENCE = 1000
