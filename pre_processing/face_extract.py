from facenet_pytorch import MTCNN
import cv2
from PIL import Image
from os import listdir, makedirs
import glob
from os.path import join, exists, basename
from skimage.io import imsave
import imageio.core.util
import torch
import argparse
from tqdm import tqdm

# Takes an input directory of folders filled with frame images and outputs the exported faces from the MTCNN

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
args = parser.parse_args()

imageio.core.util._precision_warn = lambda *a, **k: None

mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
)

for video_folder in tqdm(listdir(args.input_dir)):
    frame_files = glob.glob(join(args.input_dir, video_folder, "*.jpg"))
    video_output_dir = join(args.output_dir, video_folder)

    if not exists(video_output_dir):
        makedirs(video_output_dir)

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        face = mtcnn(frame)
        try:
            imsave(join(video_output_dir, basename(frame_file)), face.permute(1, 2, 0).int().numpy(),
                   check_contrast=False)
        except AttributeError:
            print("Image skipping")
