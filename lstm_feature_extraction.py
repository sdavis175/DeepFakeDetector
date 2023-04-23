import os

from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dropout, Dense
from keras.models import Model
import numpy as np
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
from keras.optimizers import Nadam
from keras.applications.xception import Xception
import argparse
import pandas as pd
import tensorflow as tf


def ignore_warnings(*args, **kwargs):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--weights_save_name", type=str, default="xception")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--data_folder", type=str, default="../Celeb-DF-v2/")
    parser.add_argument("--test_file", type=str, required=True)
    args = parser.parse_args()

    # MTCNN face extraction from frames
    imageio.core.util._precision_warn = ignore_warnings

    # Create face detector
    mtcnn = MTCNN(
        margin=40,
        select_largest=False,
        post_process=False,
        device="cuda:0"
    )

    # Generate Xception model
    input_size = (args.image_size, args.image_size, 3)  # rgb images
    base = Xception(weights="imagenet", include_top=False, input_shape=input_size)

    headModel = base.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(512, activation="tanh", kernel_initializer="he_uniform", name="fc1")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=base.input, outputs=predictions)

    model.load_weights("trained_wts/" + args.weights_save_name + ".hdf5")
    print("weights loaded...")

    model_lstm = Model(
        inputs=base.input,
        outputs=model.get_layer("fc1").output
    )

    for layer in base.layers:
        layer.trainable = True

    optimizer = Nadam(
        learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    print("creating dataset")
    test_data = pd.read_csv(args.test_file, delimiter=" ")

    videos = test_data[test_data.columns[1]]
    true_labels = test_data[test_data.columns[0]]
    print("dataset loaded...")

    features = []
    counter = 0
    labels = []

    for video in videos:
        print(f"Looking into video: {os.path.join(args.data_folder, video)}")
        cap = cv2.VideoCapture(os.path.join(args.data_folder, video))
        labels += [1 if "synthesis" in str(video) else 0]

        batches = []

        while cap.isOpened() and len(batches) < args.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            if h >= 1080 and w >= 1920:
                frame = cv2.resize(
                    frame,
                    (640, 480),
                    interpolation=cv2.INTER_AREA
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            try:
                face = face.permute(1, 2, 0)
                face = tf.image.resize(face, (args.image_size, args.image_size))
                face = np.asarray(tf.cast(face, dtype=tf.int32))
                batches.append(face)
            except AttributeError:
                print("Image Skipping")

        cap.release()
        batches = np.array(batches).astype("float32")
        batches /= 255

        # fc layer feature generation
        predictions = model_lstm.predict(batches)

        features += [predictions]

        if counter % 50 == 0:
            print("Number of videos done:", counter)
        counter += 1

    features = np.array(features)
    labels = np.array(labels)

    print(features.shape, labels.shape)

    np.save("lstm_40fpv_data.npy", features)
    np.save("lstm_40fpv_labels.npy", labels)

if __name__ == "__main__":
    main()