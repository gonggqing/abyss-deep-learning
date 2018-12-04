#!/usr/bin/python3


import numpy as np
import json
from pycocotools.coco import COCO
import os
import argparse
import glob
import cv2


def get_unique_list_of_videos(coco):
    video_list = []
    imgIds = coco.getImgIds(imgIds=[]) # Load all images
    for id in imgIds:
        video_path = coco.loadImgs(id)[0]["path"]
        video_name = os.path.basename(video_path).split(':')[0].strip(" (1)")
        video_list.append(video_name)
    return list(set(video_list))


def check_number_of_videos_per_json(coco):
    video_list = get_unique_list_of_videos(coco)
    if len(video_list) > 1:
        return False
    return True


def get_video_basename(coco):
    return get_unique_list_of_videos(coco)[0].split(".")[0]


def get_video_path(coco):
    imgIds = coco.getImgIds(imgIds=[]) # Load all images
    id = imgIds[0]
    video_path = coco.loadImgs(id)[0]["path"].split(":")[0]
    print(video_path)
    return video_path

def get_video_at_default_path(coco, default_path):
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
    for f in glob.glob(os.path.join(args.cf_json_dir, "**", "*.json"), recursive=True):
        coco = COCO(annotation_file=f)
        if not check_number_of_videos_per_json(coco):
            raise EnvironmentError("JSON has more than one video")
        video_basename = get_video_basename(coco)
        # Make a new folder for the video, and frames
        if not os.path.exists(os.path.join(args.output_dir, video_basename, "frames")):
            os.makedirs(os.path.join(args.output_dir, video_basename, "frames"))
        frames_path = os.path.join(args.output_dir, video_basename, "frames")

        cap = cv2.VideoCapture(get_video_at_default_path(coco, args.default_video_path))

        # Save all the images to the frames folder
        for im_idx in range(len(coco.dataset['images'])):
            # Original Image Path
            img_path = coco.dataset['images'][im_idx]["path"]
            # The image frame number
            img_frame_number = int(img_path.split(":")[-1])
            print(img_frame_number)
            # New Path
            new_path = os.path.join(frames_path, video_basename + "_" + str(img_frame_number) + ".png")
            # Set the frame in the video
            cap.set(1, img_frame_number)
            # Read the Image
            ret, img = cap.read()
            if ret == True:
                # Write the image to the new path
                cv2.imwrite(new_path, img)
            else:
                print("FAILED TO GET FRAME")
            # New Coco Dataset
            coco.dataset['images'][im_idx]["path"] = new_path

        # json.dump(open(os.path.join(args.output_dir, video_basename, 'coco-dataset.json'), 'w'))
        with open(os.path.join(args.output_dir, video_basename, 'coco-dataset.json'), 'w') as outfile:
            json.dump(coco.dataset, outfile, sort_keys=True, indent=4)
        # json.dumps(coco.dataset, os.path.join(args.output_dir, video_basename, 'coco-dataset.json'))
        # print(coco.dataset, os.path.join(args.output_dir, video_basename, 'coco-dataset.json'))


if __name__ == "__main__":
    main(get_args())


    # print(video_list)
