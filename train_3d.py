from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Input, Activation
from keras.layers import RandomFlip, RandomZoom, RandomRotation, RandomTranslation
import tensorflow as tf
from keras.utils import np_utils
from pathlib import Path
from keras.regularizers import l2


import os
import time
import argparse
import matplotlib.pyplot as plt
from random import shuffle
import random
import cv2
import numpy as np

from DeepFakeDetector.pre_processing.videos_to_tf_dataset import create_dataset


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

def plot_history(history, result_dir):
    plt.plot(history.history["accuracy"], marker=".")
    plt.plot(history.history["val_accuracy"], marker=".")
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid()
    plt.legend(["accuracy", "val_accuracy"], loc="lower right")
    plt.savefig(os.path.join(result_dir, "model_accuracy.png"))
    plt.close()

    plt.plot(history.history["loss"], marker=".")
    plt.plot(history.history["val_loss"], marker=".")
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    plt.legend(["loss", "val_loss"], loc="upper right")
    plt.savefig(os.path.join(result_dir, "model_loss.png"))
    plt.close()


def save_history(history, result_dir):
    loss = history.history["loss"]
    acc = history.history["acc"]
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_acc"]
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, "result.txt"), "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write(
                "{}\t{}\t{}\t{}\t{}\n".format(
                    i, loss[i], acc[i], val_loss[i], val_acc[i]
                )
            )
        fp.close()

def process_batch(video_paths, batch_size, train=True):
    num = len(video_paths)
    batch = np.zeros((num, batch_size, 112, 112, 3), dtype="int")
    labels = np.zeros(num, dtype="int")
    for i in range(num):
        path = video_paths[i]
        label = 1 if "synthesis" in str(video_paths[i]) else 0
        label = int(label)
        imgs = os.listdir(path)
        imgs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)
            try:
                for j in range(batch_size):
                    img = imgs[j]
                    image = cv2.imread(os.path.join(path, img))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (171, 128))
                    if is_flip == 1:
                        image = cv2.flip(image, 1)
                    batch[i][j][:][:][:] = image[
                        crop_x: crop_x + 112, crop_y: crop_y + 112, :
                    ]
                labels[i] = label
            except:
                print(len(imgs))
                print(path)
        else:
            for j in range(batch_size):
                img = imgs[j]
                image = cv2.imread(path + "/" + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                batch[i][j][:][:][:] = image[8:120, 30:142, :]
            labels[i] = label

    batch = np.asarray(batch, dtype='float32')

    return batch, labels


def preprocess(inputs):
    inputs /= 255.0
    return inputs


def generator_train_batch(train_vid_list, batch_size, num_classes):
    num = len(train_vid_list)
    while True:
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            x_train, x_labels = process_batch(
                train_vid_list[a:b],
                batch_size,
                train=True
            )
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            yield x, y


def generator_val_batch(val_vid_list, batch_size, num_classes):
    num = len(val_vid_list)
    while True:
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test, y_labels = process_batch(
                val_vid_list[a:b],
                batch_size,
                train=False
            )
            x = preprocess(y_test)
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frames_per_video", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weights_save_name", type=str, default="c3d")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--synthetic_dir", type=str, required=True)
    args = parser.parse_args()

    num_classes = 2

    print("creating dataset")
    train_path = ["../face_frames_split/face_frames_split/training/synthesis", "../face_frames_split/face_frames_split/training/real"]

    list_1 = [os.path.join(train_path[0], x) for x in os.listdir(train_path[0])]
    list_0 = [os.path.join(train_path[1], x) for x in os.listdir(train_path[1])]

    for i in range(1):
        vid_list = list_1 + list_0[i * (len(list_1)): (i + 1) * (len(list_1))]
        print(vid_list)
        print(len(vid_list))
        shuffle(vid_list)

        # Distrbution of training data as 80/20 according to FF++ paper
        train_vid_list = vid_list[: int(0.8 * len(vid_list))]
        val_vid_list = vid_list[int(0.8 * len(vid_list)):]

    print("Dataset Loaded...")

    model = c3d_model(batch_size=args.batch_size)

    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True, clipnorm=1)
    model.compile(
        loss="binary_crossentropy",
        optimizer=sgd,
        metrics=["accuracy"]
    )

    # Model fitting
    history = model.fit(
        generator_train_batch(train_vid_list, args.batch_size, num_classes),
        validation_data=generator_val_batch(
            val_vid_list,
            args.batch_size,
            num_classes
        ),
        epochs=args.epochs,
        verbose=1,
    )

    # Make results directory
    if not os.path.exists("results/"):
        os.mkdir("results/")
    plot_history(history, "results/")
    save_history(history, "results/")
    model.save_weights("results/" + args.weights_save_name + ".hdf5")

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