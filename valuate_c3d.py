import os

import numpy as np
import cv2
import time
import argparse
import pandas as pd
import random

from keras.layers import Activation
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Input

# from schedules import onetenth_4_8_12
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings

# Create face detector
mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device="cuda:0"
)

def c3d_model(batch_size):
    """ Return the Keras model of the network
    """
    main_input = Input(shape=(batch_size, 112, 112, 3), name="main_input")
    # 1st layer group
    x = Conv3D(
        64,
        kernel_size=(3, 3, 3),
        activation="tanh",
        padding="same",
        name="conv1",
        strides=(1, 1, 1),
    )(main_input)
    x = MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1, 2, 2), padding="valid", name="pool1"
    )(x)
    # 2nd layer group
    x = Conv3D(
        128,
        kernel_size=(3, 3, 3),
        activation="tanh",
        padding="same",
        name="conv2",
        strides=(1, 1, 1),
    )(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool2"
    )(x)
    # 3rd layer group
    x = Conv3D(
        256,
        kernel_size=(3, 3, 3),
        activation="tanh",
        padding="same",
        name="conv3a",
        strides=(1, 1, 1),
    )(x)
    x = Conv3D(
        256,
        kernel_size=(3, 3, 3),
        activation="tanh",
        padding="same",
        name="conv3b",
        strides=(1, 1, 1),
    )(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool3"
    )(x)
    # 4th layer group
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="tanh",
        padding="same",
        name="conv4a",
        strides=(1, 1, 1),
    )(x)
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="tanh",
        padding="same",
        name="conv4b",
        strides=(1, 1, 1),
    )(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same", name="pool4"
    )(x)
    # 5th layer group
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="tanh",
        padding="same",
        name="conv5a",
        strides=(1, 1, 1),
    )(x)
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="tanh",
        padding="same",
        name="conv5b",
        strides=(1, 1, 1),
    )(x)
    x = ZeroPadding3D(padding=(0, 1, 1))(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same", name="pool5"
    )(x)
    x = Flatten()(x)
    # FC layers group
    x = Dense(2048, activation="tanh", name="fc6")(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="tanh", name="fc7")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation="softmax", name="fc8")(x)

    model = Model(inputs=main_input, outputs=predictions)
    return model

def process_batch(video_paths, batch_size, data_folder=""):
    num = len(video_paths)
    batch = np.zeros((num, batch_size, 112, 112, 3), dtype="float32")
    labels = np.zeros(num, dtype="int")
    i = 0
    for video in video_paths:
        cap = cv2.VideoCapture(data_folder + video)
        batches = []
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)
            try:
                face = face.permute(1, 2, 0).float().numpy()
                face = cv2.resize(face, (171, 128))
                batch[i][counter][:][:][:] = face[8:120, 30:142, :]
                batches.append(face)
            except AttributeError:
                print("Image Skipping")
            if counter == batch_size-1:
                break
            counter += 1
        cap.release()
        label = 1 if "synthesis" in str(data_folder + video) else 0
        label = int(label)
        labels[i] = label
        i += 1
    return batch, labels

def preprocess(inputs):
    inputs /= 255.0
    return inputs

def generator_test_batch(train_vid_list, batch_size, num_classes, data_folder):
    num = len(train_vid_list)

    for i in range(int(num / batch_size)):
        a = i * batch_size
        b = (i + 1) * batch_size
        x_train, x_labels = process_batch(
            train_vid_list[a:b],
            batch_size,
            data_folder
        )
        x = preprocess(x_train)
        y = np_utils.to_categorical(np.array(x_labels), num_classes)
        yield x, y

def main():
    start = time.time()

    num_classes = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frames_per_video", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weights_save_name", type=str, default="c3d")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True)

    args = parser.parse_args()

    test_data = pd.read_csv(args.test_file, delimiter=" ")

    videos = test_data[test_data.columns[1]]
    true_labels = test_data[test_data.columns[0]]

    model = c3d_model(batch_size=args.batch_size)

    lr = 0.005
    sgd = SGD(learning_rate=lr, momentum=0.9, nesterov=True, clipnorm=1)
    model.compile(
        loss="binary_crossentropy",
        optimizer=sgd,
        metrics=["accuracy"]
    )

    model.load_weights("trained_wts/" + args.weights_save_name + ".hdf5")

    print(len(videos))

    probabs = model.predict(
        generator_test_batch(videos, args.batch_size, num_classes, args.data_folder),
        steps=len(videos),
        verbose=1
    )

    y_pred = probabs.argmax(1)

    #print(true_labels)
    #print(y_pred)

    true_labels = true_labels[0:len(true_labels) - (len(true_labels) % args.batch_size)]

    print("Accuracy Score:", accuracy_score(true_labels, y_pred))
    print("Precision Score", precision_score(true_labels, y_pred))
    print("Recall Score:", recall_score(true_labels, y_pred))
    print("F1 Score:", f1_score(true_labels, y_pred))

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