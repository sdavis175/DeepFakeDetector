from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Input
from keras.layers import RandomFlip, RandomZoom, RandomRotation, RandomTranslation
from keras.utils import np_utils

import os
import time
import argparse
import matplotlib.pyplot as plt
from random import shuffle
import random
import cv2
import numpy as np


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
    train_path = [args.synthetic_dir, args.real_dir]

    list_1 = [os.path.join(train_path[0], x) for x in os.listdir(train_path[0])]
    list_0 = [os.path.join(train_path[1], x) for x in os.listdir(train_path[1])]

    train_data_augmentation_model = Sequential(
        [
            RandomRotation(factor=1 / 12, fill_mode="nearest"),
            RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode="nearest"),
            RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="nearest"),
            RandomFlip("horizontal"),
        ],
        name="train_data_augmentation",
    )


    for i in range(1):
        vid_list = list_1 + list_0[i * (len(list_1)): (i + 1) * (len(list_1))]
        shuffle(vid_list)

        # Distrbution of training data as 80/20 according to FF++ paper
        train_vid_list = vid_list[: int(0.8 * len(vid_list))]
        val_vid_list = vid_list[int(0.8 * len(vid_list)):]

    model = c3d_model(batch_size=args.batch_size)

    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True, clipnorm=1)
    model.compile(
        loss="binary_crossentropy",
        optimizer=sgd,
        metrics=["accuracy"]
    )
    train_vid_list = train_vid_list * args.epochs
    val_vid_list = val_vid_list * args.epochs
    # Model fitting
    history = model.fit(
        generator_train_batch(train_vid_list, args.batch_size, num_classes),
        validation_data=generator_val_batch(
            val_vid_list,
            args.batch_size,
            num_classes
        ),
        validation_steps=len(val_vid_list) // args.epochs,
        steps_per_epoch=len(train_vid_list) // args.epochs,
        epochs=args.epochs,
        verbose=1,
    )

    print(history.history)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots/train_c3d.png")
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