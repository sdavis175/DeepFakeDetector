from pre_processing.videos_to_tf_dataset import create_dataset

from keras.applications.xception import Xception
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dropout, Dense
from keras.layers import RandomFlip, RandomZoom, RandomRotation, RandomTranslation, Normalization
from keras.models import Model, Sequential
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger
from keras.metrics import Precision, Recall
import tensorflow as tf
from transformers import TFViTForImageClassification, ViTImageProcessor, TFDeiTForImageClassification

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
    parser.add_argument("--frames_per_video", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weights_save_name", type=str, default="vit")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--synthetic_dir", type=str, required=True)
    args = parser.parse_args()

    train_data_augmentation_model = Sequential(
        [
            Normalization(mean=[0.5, 0.5, 0.5], variance=[0.5, 0.5, 0.5]),
            RandomRotation(factor=1 / 12, fill_mode="nearest"),
            RandomZoom(height_factor=0.15, width_factor=0.15, fill_mode="nearest"),
            RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="nearest"),
            RandomFlip("horizontal"),
        ],
        name="train_data_augmentation",
    )

    val_data_augmentation_model = Sequential(
        [
            Normalization(mean=[0.5, 0.5, 0.5], variance=[0.5, 0.5, 0.5])
        ],
        name="val_data_augmentation",
    )

    def train_data_augmentation(image, label):
        image = train_data_augmentation_model(image)
        image = tf.transpose(image, perm=[2, 0, 1])  # ViT wants (channels, width, height)
        return image, label

    def val_data_augmentation(image, label):
        image = val_data_augmentation_model(image)
        image = tf.transpose(image, perm=[2, 0, 1])  # ViT wants (channels, width, height)
        return image, label

    # Get training data and apply data augmentation
    dataset = create_dataset(real_dir=args.real_dir,
                             synthetic_dir=args.synthetic_dir,
                             frames_per_video=args.frames_per_video,
                             img_size=args.image_size)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset = dataset.take(train_size).map(train_data_augmentation)
    val_dataset = dataset.skip(train_size).map(val_data_augmentation)
    print("Dataset Loaded...")

    # Generate ViT model
    model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224",
                                                        num_labels=2,
                                                        ignore_mismatched_sizes=True)
    # model.layers[0].trainable = False

    optimizer = Nadam(
        learning_rate=1e-5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", Precision(), Recall()]
    )
    model.summary()

    if not exists("./trained_wts"):
        makedirs("./trained_wts")
    if not exists("./training_logs"):
        makedirs("./training_logs")
    if not exists("./plots"):
        makedirs("./plots")

    # Keras backend
    model_checkpoint = ModelCheckpoint(
        f"trained_wts/{args.weights_save_name}.hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=0
    )
    csv_logger = CSVLogger(
        f"training_logs/{args.weights_save_name}.csv",
        separator=",",
        append=True,
    )

    print("Training is going to start in 3... 2... 1... ")

    # Model Training
    H = model.fit(
        train_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE),
        validation_data=val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE),
        epochs=args.epochs,
        callbacks=[model_checkpoint, stopping, csv_logger],
    )

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(f"plots/train_{args.weights_save_name}.png")
    model.save(f"./models/{args.weights_save_name}")

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
