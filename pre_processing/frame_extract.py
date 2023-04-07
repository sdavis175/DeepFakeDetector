# Mitchell Klingler
# Shane Davis
# CAP6135
# Implementation of Detecting Deepfakes with Metric Learning

#### TODO #### 

# Implement a way to not extract the frames from List_of_testing_videos.txt

#### TODO ####

import os
import cv2
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--training_dir", type=str)
parser.add_argument("--testing_set", type=str)
parser.add_argument("--output_dir", type=str)

args = parser.parse_args()

folder = args.training_dir

videos_path = glob.glob(os.path.join(folder, "*.mp4"))

for video_index, video_path in enumerate(videos_path):
    capture = cv2.VideoCapture(video_path)
    vid = video_path.split("/")[-1]
    vid = vid.split(".")[0]
    frameRate = capture.get(5)
    video_frame_folder = f"{args.output_dir}/video_{video_index}"

    if not os.path.exists(video_frame_folder):
        os.makedirs(video_frame_folder)

    while capture.isOpened():
        frameId = capture.get(1)  # current frame number
        ret, frame = capture.read()
        if not ret:
            break

        filename = f"{video_frame_folder}/image_{frameId + 1}.jpg"
        cv2.imwrite(filename, frame)

    capture.release()

    if video_index % 100 == 0:
        print(f"Number of videos done: {video_index + 1}")
