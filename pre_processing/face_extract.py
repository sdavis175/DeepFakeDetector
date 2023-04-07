# Mitchell Klingler 
# Shane Davis
# CAP 6135 Final Project

from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
from os import listdir, makedirs
import glob
from os.path import join, exists
from skimage.io import imsave
import imageio.core.util
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--training_dir')
parser.add_argument('--destination_dir')

args = parser.parse_args()


use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def ignore_warnings(*args, **kwargs):
    pass
imageio.core.util._precision_warn = ignore_warnings

mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device=device
)

counter = 0
for j in listdir(args.training_dir):
    for file in listdir(join(args.training_dir, j)):
        imgs = glob.glob(join(args.training_dir, j, "*.jpg"))
        if counter % 1000 == 0:
            print("Number of videos done:", counter)
        if not exists(join(args.destination_dir, j)):
            makedirs(join(args.destination_dir, j))
        for k in imgs:
            frame = cv2.imread(k)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)
            try:
                print(join(args.destination_dir, j, k.split("/")[-1]))
                imsave(
                    join(args.destination_dir, j, k.split("/")[-1]),
                    face.permute(1, 2, 0).int().numpy(),
                )
            except AttributeError:
                print("Image skipping")
        counter += 1