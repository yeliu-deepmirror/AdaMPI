from pathlib import Path
import argparse
from PIL import Image
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import gc
from transformers import DPTForDepthEstimation, DPTImageProcessor
from moviepy.editor import ImageSequenceClip

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto,
    write_mpi_to_binary,
    process_image,
    write_array,
    merge_rgba_layers,
)
from model.AdaMPI import MPIPredictor


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_path', type=str)
parser.add_argument('--output_path', type=str, default="debug/video_mpi")
# parser.add_argument('--width', type=int, default=384)
# parser.add_argument('--height', type=int, default=256)
parser.add_argument('--ckpt_path', type=str, default="weight/adampi_32p.pth")
opt, _ = parser.parse_known_args()

cap = cv2.VideoCapture(opt.video_path)

if (cap.isOpened()== False):
    print("Error opening video stream or file")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("open video", video_width, video_height, frame_count, fps)

# process width height should be 128 * n
resize_factor = 1
n_w = int(resize_factor * video_width / 128)
n_h = int(resize_factor * video_height / 128)
width = n_w * 128
height = n_h * 128
print("process size", width, height)


print("Load the models.")
model_depth = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

# load pretrained model
ckpt = torch.load(opt.ckpt_path)
model_mpi = MPIPredictor(
    width=width,
    height=height,
    num_planes=ckpt["num_planes"],
)
model_mpi.load_state_dict(ckpt["weight"])
model_mpi = model_mpi.eval()

# predict MPI planes
print("predict MPI planes...")

os.makedirs(opt.output_path, exist_ok=True)
os.makedirs(opt.output_path + "/tmp", exist_ok=True)
# Read until video is completed
cnt = 1
while(cap.isOpened()):
    ret, image_cv = cap.read()
    if ret == False:
        break
    print("process", cnt, "/", frame_count)
    image_path = opt.output_path + '/tmp.jpg'
    cv2.imwrite(image_path, image_cv)

    with torch.no_grad():
        rgba_layers = process_image(image_path, (height, width), model_mpi, model_depth, image_processor, "cpu")

    # print(rgba_layers.shape)
    rgb, alpha = merge_rgba_layers(rgba_layers)
    # print(large_rgba_image.shape)
    cv2.imwrite(opt.output_path + "/tmp/" + str(cnt) + "rgb.jpg", rgb)
    cv2.imwrite(opt.output_path + "/tmp/" + str(cnt) + "alpha.jpg", alpha)
    cnt = cnt + 1

    gc.collect()
    # torch.cuda.empty_cache()

cap.release()

print("Done")
