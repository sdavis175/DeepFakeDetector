from keras.applications.xception import Xception
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dropout, Dense
from keras.models import Model
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger
import tensorflow as tf

import numpy as np
import time
import argparse
from os.path import exists
from os import makedirs
from matplotlib import pyplot as plt


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weights_save_name", type=str, default="model")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--train_labels_path", type=str, required=True)

    args = parser.parse_args()

    # Get training data
    train_data = np.load(args.train_data_path)
    train_label = np.load(args.train_labels_path)

    print("Dataset Loaded...")

    trainX, valX, trainY, valY = train_test_split(
        train_data, train_label, test_size=0.1, shuffle=False
    )
    print(trainX.shape, valX.shape, trainY.shape, valY.shape)

    trainAug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valAug = ImageDataGenerator(rescale=1.0 / 255.0)

    # Generate Xception model
    input_size = (args.image_size, args.image_size, 3)  # rgb images
    model = Xception(weights="imagenet", include_top=False, input_shape=input_size)

    output = model.output
    output = GlobalAveragePooling2D()(output)
    output = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        output
    )
    output = Dropout(0.4)(output)
    output = Dropout(0.5)(output)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform")(
        output
    )
    model = Model(inputs=model.input, outputs=predictions)

    for layer in model.layers:
        layer.trainable = True

    optimizer = Nadam(
        learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    # Number of trainable and non-trainable parameters
    # trainable_count = int(
    #     np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    # )
    # non_trainable_count = int(
    #     np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    # )
    #
    # print("Total params: {:,}".format(trainable_count + non_trainable_count))
    # print("Trainable params: {:,}".format(trainable_count))
    # print("Non-trainable params: {:,}".format(non_trainable_count))

    if not exists("./trained_wts"):
        makedirs("./trained_wts")
    if not exists("./training_logs"):
        makedirs("./training_logs")
    if not exists("./plots"):
        makedirs("./plots")

    # Keras backend
    model_checkpoint = ModelCheckpoint(
        "trained_wts/" + args.weights_save_name + ".hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0)
    csv_logger = CSVLogger(
        "training_logs/xception.log",
        separator=",",
        append=True,
    )

    print("Training is going to start in 3... 2... 1... ")

    # Model Training
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=args.batch_size),
        steps_per_epoch=len(trainX) // args.batch_size,
        validation_data=valAug.flow(valX, valY),
        validation_steps=len(valX) // args.batch_size,
        epochs=args.epochs,
        callbacks=[model_checkpoint, stopping, csv_logger],
    )

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = stopping.stopped_epoch + 1
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots/training_plot.png")

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
    # Don't pre-allocate all my VRAM
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    main()
