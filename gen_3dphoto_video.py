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

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto,
    write_mpi_to_binary,
    process_image,
    write_array,
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
video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("open video", video_width, video_height, frame_count)

# process width height should be 128 * n
n_w = int(video_width / 128)
n_h = int(video_height / 128)
width = n_w * 128
height = n_h * 128
print("process size", width, height)


print("Load the models.")
model_depth = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").cuda()
image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

# load pretrained model
ckpt = torch.load(opt.ckpt_path)
model_mpi = MPIPredictor(
    width=384,
    height=256,
    num_planes=ckpt["num_planes"],
)
model_mpi.load_state_dict(ckpt["weight"])
model_mpi = model_mpi.eval()

# predict MPI planes
print("predict MPI planes...")

os.makedirs(opt.output_path, exist_ok=True)
# Read until video is completed
cnt = 1
while(cap.isOpened()):
    ret, image_cv = cap.read()
    if ret == False:
        break
    print("process", cnt, "/", frame_count)
    image_path = opt.output_path + '/' + str(cnt) + '.jpg'
    cv2.imwrite(image_path, image_cv)
    cnt = cnt + 1

    with torch.no_grad():
        rgba = process_image(image_path, (height, width), model_mpi, model_depth, image_processor, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

cap.release()

# with torch.no_grad():
#     rgba = process_image(opt.img_path, (opt.height, opt.width), model_mpi, model_depth, image_processor, "cpu")
#     write_array(rgba, (-1, -1), opt.img_path + ".bin")

print("Done")
