from pre_processing.videos_to_tf_dataset import create_dataset

from argparse import ArgumentParser
import tensorflow as tf
from keras.layers import Normalization
from keras.models import Model, Sequential, load_model
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm


def evaluate_vit(args):
    val_data_augmentation_model = Sequential(
        [
            Normalization(mean=[0.5, 0.5, 0.5], variance=[0.5, 0.5, 0.5])
        ],
        name="val_data_augmentation",
    )

    def val_data_augmentation(image, label):
        image = val_data_augmentation_model(image)
        image = tf.transpose(image, perm=[2, 0, 1])  # ViT wants (channels, width, height)
        return image, label

    # Get testing data and apply data augmentation
    dataset = create_dataset(real_dir=args.real_dir,
                             synthetic_dir=args.synthetic_dir,
                             frames_per_video=args.frames_per_video,
                             img_size=args.image_size)
    dataset = dataset.map(val_data_augmentation)
    print("Dataset Loaded...")

    # Load the pre-trained model
    model = load_model(args.model_dir)
    model: Model

    # Run inference on the dataset
    y_true, y_pred = [], []
    for batch in tqdm(dataset.batch(args.batch_size), desc="Batches"):
        X_batch, y_batch = batch
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(model.predict({"pixel_values": X_batch}, verbose=False)["logits"], axis=1))

    # Generate stats
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    # Don't pre-allocate all my VRAM
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    parser = ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frames_per_video", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    evaluate_vit(parser.parse_args())
