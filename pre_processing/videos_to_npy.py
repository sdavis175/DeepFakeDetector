import cv2
import argparse
import numpy as np
from keras.utils import np_utils
import glob
from os.path import join, basename
from os import listdir
from random import shuffle
from tqdm import tqdm

SYNTHETIC_LABEL = 0
REAL_LABEL = 1

# Takes the input real/synthetic video directories and takes the first specified amount of video frames and resizes them
# Saves them into a .npy format along with the associated labels

parser = argparse.ArgumentParser()
parser.add_argument(
    "--real_dir",
    required=True,
    type=str,
    help="Directory of real videos"
)
parser.add_argument(
    "--synthetic_dir",
    required=True,
    type=str,
    help="Directory of synthetic videos"
)
parser.add_argument(
    "--output_dir",
    required=True,
    type=str,
    help="Directory of output .npy"
)
parser.add_argument(
    "--img_size",
    type=int,
    help="Resize face image",
    default=160
)
parser.add_argument(
    "--frames_per_video",
    type=int,
    help="Number of frames per video to consider",
    default=25
)
args = parser.parse_args()

# Load all the video folders from the directories
real_videos = [join(args.real_dir, video_folder) for video_folder in listdir(args.real_dir)]
synthetic_videos = [join(args.synthetic_dir, video_folder) for video_folder in listdir(args.synthetic_dir)]
all_videos = real_videos + synthetic_videos
shuffle(all_videos)

train_data = []
train_label = []

for video_dir in tqdm(all_videos):
    video_frames_paths = sorted(glob.glob(join(video_dir, "*.jpg")),
                                key=lambda frame_path: int("".join(filter(str.isdigit, basename(frame_path))))
                                )[:args.frames_per_video]
    label = SYNTHETIC_LABEL if video_dir in synthetic_videos else REAL_LABEL if video_dir in real_videos else None
    assert label is not None

    for frame_path in video_frames_paths:
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA
        )
        train_data.append(frame)
        train_label.append(label)

print("Saving data and labels")
train_data = np.array(train_data)
train_label = np_utils.to_categorical(np.array(train_label))
np.save(join(args.output_dir, f"data_{args.frames_per_video}.npy"), train_data)
np.save(join(args.output_dir, f"label_{args.frames_per_video}.npy"), train_label)
