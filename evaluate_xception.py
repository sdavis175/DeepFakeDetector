import torch

from pre_processing.videos_to_tf_dataset import create_dataset

import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import cv2
import time
import argparse
import numpy as np
from keras.models import Model, Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import RandomFlip, RandomZoom, RandomRotation, RandomTranslation
from keras.layers.core import Dropout, Dense
from keras.optimizers import Nadam
from keras.applications.xception import Xception
#from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
#from keras_efficientnets import EfficientNetB5, EfficientNetB0
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import tensorflow as tf

def ignore_warnings(*args, **kwargs):
    pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frames_per_video", type=int, default=25)
    parser.add_argument("--weights_save_name", type=str, default="xception")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--datafolder", type=str, required=True)
    args = parser.parse_args()

    # Generate Xception model
    input_size = (args.image_size, args.image_size, 3)  # rgb images
    model = Xception(weights="imagenet", include_top=False, input_shape=input_size)

    output = model.output
    output = GlobalAveragePooling2D()(output)
    output = Dense(512, activation="relu", kernel_initializer="he_uniform")(output)
    output = Dropout(0.4)(output)
    output = Dropout(0.5)(output)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform"
    )(output)
    model = Model(inputs=model.input, outputs=predictions)
    model.load_weights("trained_wts/" + args.weights_save_name + ".hdf5")
    mtcnn = MTCNN(
        margin=40,
        select_largest=False,
        post_process=False,
        device="cuda:0"
    )

    y_predictions = []
    y_probabilities = []
    videos_done = 0

    test_data = pd.read_csv(args.test_file, delimiter=" ")

    videos = test_data[test_data.columns[1]]
    true_labels = test_data[test_data.columns[0]]
    print(videos)

    start = time.time()

    print(len(videos))

    for video in videos:
        print(args.datafolder + video)
        cap = cv2.VideoCapture(args.datafolder + video)
        batches = []

        # Number of frames taken into consideration for each video
        while (cap.isOpened() and len(batches) < 25):
            ret, frame = cap.read()
            if ret is not True:
                break

            frame = cv2.resize(frame, (args.image_size, args.image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            try:
                face = face.permute(1, 2, 0)
                face = tf.image.resize(face, (args.image_size, args.image_size))
                face = np.asarray(tf.cast(face, dtype=tf.int32))
                batches.append(face)
            except AttributeError as e:
                print(e)
                print("Image Skipping")

        batches = np.asarray(batches).astype("float32")
        batches /= 255

        predictions = model.predict(batches)
        # Predict the output of each frame
        # axis =1 along the row and axis=0 along the column
        predictions_mean = np.mean(predictions, axis=0)
        y_probabilities += [predictions_mean]
        y_predictions += [predictions_mean.argmax(0)]

        cap.release()

        if videos_done % 10 == 0:
            print("Number of videos done:", videos_done)
        videos_done += 1

    print("Accuracy Score:", accuracy_score(true_labels, y_predictions))
    print("Precision Score:", precision_score(true_labels, y_predictions))
    print("Recall Score:", recall_score(true_labels, y_predictions))
    print("F1 Score:", f1_score(true_labels, y_predictions))

    # Saving predictions and probabilities for further calculation
    # of AUC scores.
    np.save("Y_predictions.npy", y_predictions)
    np.save("Y_probabilities.npy", y_probabilities)

    end = time.time()
    dur = end - start

    if dur < 60:
        print("Execution Time:", dur, "seconds")
    elif dur > 60 and dur < 3600:
        dur = dur / 60
        print("Execution Time:", dur, "minutes")
    else:
        dur = dur / (60 * 60)
        print("Execution Time:", dur, "hours")


if __name__ == "__main__":
    main()