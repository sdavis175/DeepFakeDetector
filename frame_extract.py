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

args = parser.parse_args()

folder = args.training_dir

videos_path = glob.glob(os.path.join(folder, "*.mp4"))

c=0
for video_path in videos_path:
    capture = cv2.VideoCapture(video_path)
    vid = video_path.split("/")[-1]
    vid = vid.split(".")[0]
    frameRate = capture.get(5)

    if not os.path.exists("../train_frames" + "/video_" + str(int(c))):
        os.makedirs("../train_frames" + "/video_" + str(int(c)))

    while capture.isOpened():
        frameId = capture.get(1)  # current frame number
        ret, frame = capture.read()
        if not ret:
            break

        filename = (
            "../train_frames"
            + "/video_"
            + str(int(c))
            + "/image_"
            + str(int(frameId) + 1)
            + ".jpg"
        )
        cv2.imwrite(filename, frame)

    capture.release()

    if c % 100 == 0:
        print("Number of videos done:", c)
    c += 1