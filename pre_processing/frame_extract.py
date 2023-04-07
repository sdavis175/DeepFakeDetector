import os
import cv2
import glob
import argparse
from tqdm import tqdm

# Takes in a directory of videos and outputs the frames of each video into individual indexed directories

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()
videos_path = glob.glob(os.path.join(args.input_dir, "*.mp4"))

for video_index, video_path in tqdm(enumerate(videos_path), total=len(videos_path)):
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

        filename = f"{video_frame_folder}/image_{int(frameId + 1)}.jpg"
        cv2.imwrite(filename, frame)

    capture.release()
