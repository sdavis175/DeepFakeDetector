import tensorflow as tf
import os
from glob import glob


def create_dataset(dataset_path: str, is_training=True,
                   num_frames=25,
                   img_size=224,
                   preprocessing_function=lambda image: image,
                   shuffle_buffer_size=10000):
    # Define the list of image paths and labels
    classes = ["synthetic", "real"]
    image_paths = []
    labels = []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, "training" if is_training else "testing", class_name)
        for video_name in os.listdir(class_dir):
            video_dir = os.path.join(class_dir, video_name)
            frame_paths = sorted(glob(os.path.join(video_dir, "*.jpg")),
                                 key=lambda frame_path: int("".join(filter(str.isdigit, os.path.basename(frame_path))))
                                 )
            if len(frame_paths) < num_frames:
                continue
            for i in range(1, num_frames + 1):
                image_paths.append(tf.constant(frame_paths[i - 1], dtype=tf.string))
                labels.append(tf.constant(class_idx, dtype=tf.int8))

    # Convert the python lists to tensors and build the dataset
    dataset = tf.data.Dataset.from_tensor_slices((tf.stack(image_paths), tf.stack(labels)))

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size,
                              reshuffle_each_iteration=True)

    # Create a function to load and preprocess the images
    def load_and_preprocess_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (img_size, img_size))
        image = tf.cast(image, tf.float32)
        image /= 255.0
        image = preprocessing_function(image)
        label = tf.cast(label, tf.int32)
        return image, label

    # Map the function over the dataset
    dataset = dataset.map(load_and_preprocess_image)

    return dataset


if __name__ == '__main__':
    # Create the training dataset
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    train_dataset = create_dataset("../datasets/Celeb-DF-v2/face_frames_split/", shuffle_buffer_size=150000,
                                   is_training=True).batch(32).prefetch(tf.data.AUTOTUNE)
    print("Finished loading")
    print([x for x in train_dataset.take(1)])
