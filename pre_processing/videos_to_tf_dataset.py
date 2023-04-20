import tensorflow as tf
from os import listdir
from os.path import join, basename
from random import shuffle
from glob import glob
from tqdm import tqdm

SYNTHETIC_LABEL = 0
REAL_LABEL = 1
AMOUNT_OF_LABELS = 2


def create_dataset(real_dir: str, synthetic_dir: str,
                   frames_per_video=25,
                   img_size=224,
                   preprocessing_function=None):
    # Load all the video folders from the directories
    real_videos = [join(real_dir, video_folder) for video_folder in listdir(real_dir)]
    synthetic_videos = [join(synthetic_dir, video_folder) for video_folder in listdir(synthetic_dir)]
    all_videos = real_videos + synthetic_videos
    shuffle(all_videos)

    # Define the list of image paths and labels
    image_paths = []
    labels = []
    for video_dir in tqdm(all_videos, desc="Videos Loaded"):
        video_frames_paths = sorted(glob(join(video_dir, "*.jpg")),
                                    key=lambda frame_path: int("".join(filter(str.isdigit, basename(frame_path))))
                                    )[:frames_per_video]
        if len(video_frames_paths) < frames_per_video:
            continue
        video_label = SYNTHETIC_LABEL if video_dir in synthetic_videos else REAL_LABEL if video_dir in real_videos \
            else None
        assert video_label is not None
        for frame_path in video_frames_paths:
            image_paths.append(tf.constant(frame_path, dtype=tf.string))
            labels.append(tf.constant(video_label, dtype=tf.uint8))

    # Convert the python lists to tensors and build the dataset
    dataset = tf.data.Dataset.from_tensor_slices((tf.stack(image_paths), tf.stack(labels)))

    # Create a function to load and preprocess the images
    def load_and_preprocess_image(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        if preprocessing_function is None:
            # Resize and normalize
            image = tf.image.resize(image, (img_size, img_size))
            image /= 255.0
        else:
            image = preprocessing_function(image)

        label = tf.one_hot(label, AMOUNT_OF_LABELS)
        return image, label

    # Map the function over the dataset
    dataset = dataset.map(load_and_preprocess_image)

    return dataset


if __name__ == '__main__':
    # Create the training dataset
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    dataset = create_dataset(real_dir="../datasets/Celeb-DF-v2/face_frames_split/training/real/",
                             synthetic_dir="../datasets/Celeb-DF-v2/face_frames_split/training/synthetic/",
                             ).batch(32).prefetch(tf.data.AUTOTUNE)
    print("Finished loading")
    print([x for x in dataset.take(1)])
