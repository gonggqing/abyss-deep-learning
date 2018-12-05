#!/usr/bin/python3

import argparse
import glob
import json
import logging
import os
import re
from inspect import currentframe

import cv2
from pycocotools.coco import COCO


def debug_print(arg):
    """Debugging tool to print line number followed by normal print statement arguments

    """
    frame_info = currentframe()
    print("Line", frame_info.f_back.f_lineno, ":", arg)


def get_unique_list_of_videos(coco):
    """Gets a unique list of all videos in the dataset

    Args:
        coco (COCO): A coco dataset

    Returns:
        list: List of strings of all videos in the dataset

    """
    video_list = []
    imgIds = coco.getImgIds(imgIds=[])  # Load all images
    for id in imgIds:
        video_path = coco.loadImgs(id)[0]["path"]
        #debug_print("video path is {}".format(video_path))
        video_name = os.path.basename(video_path).split(':')[0]
        video_name = re.sub(r" \([0-9]\)", "", video_name)
        #debug_print(video_name)
        video_list.append(video_name)
    return list(set(video_list))


def check_number_of_videos_per_json(coco):
    """Ensures there is only one video in each dataset

    Args:
        coco (COCO): Description of parameter `coco`.

    Returns:
        bool: True if there is less than 1 video in the dataset

    """
    video_list = get_unique_list_of_videos(coco)
    if len(video_list) > 1:
        return False
    return True


def get_video_basename(coco):
    """Get the basename of the video, with no .mp4

    Args:
        coco (COCO): A coco dataset

    Returns:
        str: basename of video

    """
    return get_unique_list_of_videos(coco)[0].split(".")[0]


def get_video_path(coco):
    """Get the video path

    Args:
        coco (COCO): Get the video path

    Returns:
        str: The video path

    """
    imgIds = coco.getImgIds(imgIds=[])  # Load all images
    id = imgIds[0]
    video_path = coco.loadImgs(id)[0]["path"].split(":")[0]
    debug_print(video_path)
    return video_path


def get_video_at_default_path(coco, default_path):
    """Gets the video at the provided video default path. Full path is default_path/basename.

    Args:
        coco (COCO): A coco dataset
        default_path (str): The default path of all the videos

    Returns:
        str: Video Path

    """
    return os.path.join(default_path, get_unique_list_of_videos(coco)[0])


def get_args():
    parser = argparse.ArgumentParser(
        description="Creates a dataset")
    parser.add_argument("cf_json_dir", help="Path to cloud factory JSONs")
    parser.add_argument(
        "output_dir", help="Path to the new directory")
    parser.add_argument("default_video_path", help="Default video path")
    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(filename='error.log', filemode='w', level=logging.DEBUG)

    for f in glob.glob(os.path.join(args.cf_json_dir, "**", "*.json"), recursive=True):
        coco = COCO(annotation_file=f)
        if not check_number_of_videos_per_json(coco):
            raise EnvironmentError("JSON has more than one video")
        video_basename = get_video_basename(coco)
        # Make a new folder for the video, and frames
        if not os.path.exists(os.path.join(args.output_dir, video_basename, "frames")):
            os.makedirs(os.path.join(args.output_dir, video_basename, "frames"))
        else:
            continue
        frames_path = os.path.join(args.output_dir, video_basename, "frames")

        # Open the video

        video_path = get_video_at_default_path(coco, args.default_video_path)
        cap = cv2.VideoCapture(video_path)

        # Save all the images to the frames folder
        for im_idx in range(len(coco.dataset['images'])):
            # Original Image Path
            img_path = coco.dataset['images'][im_idx]["path"]
            # The image frame number
            img_frame_number = int(img_path.split(":")[-1])
            #print(img_frame_number)
            # New Path
            new_path = os.path.join(frames_path, video_basename + "_" + str(img_frame_number) + ".png")
            # Set the frame in the video
            cap.set(1, img_frame_number)
            # Read the Image
            ret, img = cap.read()
            if ret:
                # Write the image to the new path
                cv2.imwrite(new_path, img)
            else:
                debug_print("FAILED TO GET FRAME")
                logging.debug("Failed to get frame {} from {}".format(img_frame_number, video_path))
                #debug_print(f)
                #debug_print(get_unique_list_of_videos(coco))
                #return
            # New Coco Dataset
            coco.dataset['images'][im_idx]["path"] = new_path
        # Write the new dataset
        with open(os.path.join(args.output_dir, video_basename, 'coco-dataset.json'), 'w') as outfile:
            json.dump(coco.dataset, outfile, sort_keys=True, indent=4)


if __name__ == "__main__":
    main(get_args())

    # print(video_list)
