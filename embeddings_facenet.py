import os
import numpy as np
from keras.preprocessing import image
from keras_facenet import FaceNet
from random import shuffle
from tqdm import tqdm
from glob import glob
import tensorflow as tf

train_data = []
train_label = []
count = 0

embedder = FaceNet()

print("getting video paths")
real_videos = [os.path.join("../face_frames_split/face_frames_split/reduced/real", video_folder) for video_folder in os.listdir("../face_frames_split/face_frames_split/reduced/real")]
synthetic_videos = [os.path.join("../face_frames_split/face_frames_split/reduced/synthesis", video_folder) for video_folder in os.listdir("../face_frames_split/face_frames_split/reduced/synthesis")]
all_videos = real_videos + synthetic_videos
shuffle(all_videos)

print("Getting image paths")
# Define the list of image paths and labels
image_paths = []
labels = []
for video_dir in tqdm(all_videos, desc="Videos Loaded"):
    video_frames_paths = sorted(glob(os.path.join(video_dir, "*.jpg")),
                                key=lambda frame_path: int("".join(filter(str.isdigit, os.path.basename(frame_path))))
                                )[:]
    video_label = 0 if video_dir in synthetic_videos else 1 if video_dir in real_videos \
        else None
    assert video_label is not None""
    for frame_path in video_frames_paths:
        image_paths.append(frame_path)
        labels.append(video_label)

frames = len(image_paths)

print("Embedding images...")
for (img_path, label) in zip(image_paths, labels):
    print(count, " of ", frames)
    img = tf.keras.utils.load_img(img_path)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    embeddings = embedder.embeddings(x)
    train_data.append(embeddings)
    train_label += [label]

    if count % 10000 == 0:
        print("Number of files done:", count)
    count += 1

train_data = np.array(train_data)
train_label = np.array(train_label)

np.save("train_data_reduced_facenet_embeddings.npy", train_data)
np.save("train_label__reduced_facenet_embeddings.npy", train_label)
print("Files saved....")