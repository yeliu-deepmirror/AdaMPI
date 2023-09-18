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
parser.add_argument('--output_path', type=str, default="debug/video_mpi")
parser.add_argument('--ckpt_path', type=str, default="weight/adampi_32p.pth")
opt, _ = parser.parse_known_args()

start_cnt = 150
end_cnt = 377
fps = 25


print("make rgb video")
rgb_frames = []
for i in range(start_cnt, end_cnt):
    if i%30 == 0:
        print("process", i, "/", cnt)
    rgb = cv2.imread(opt.output_path + "/tmp/" + str(i) + "rgb.jpg")
    rgb_frames.append(rgb)

video_path = opt.output_path + '/mpi_rgb.mp4'
rgb_clip = ImageSequenceClip(rgb_frames, fps=fps)
rgb_clip.write_videofile(video_path, codec='mpeg4', logger=None, bitrate='8000k')
# rgb_clip.write_videofile(video_path, verbose=False)
rgb_frames.clear()


print("make alpha video")
alpha_frames = []
for i in range(start_cnt, end_cnt):
    if i%30 == 0:
        print("process", i, "/", cnt)
    alpha = cv2.imread(opt.output_path + "/tmp/" + str(i) + "alpha.jpg")
    alpha_frames.append(alpha)

# MP4 not supported alpha channel so make a new file for alpha
alpha_path = opt.output_path + '/mpi_alpha.mp4'
rgb_clip_alpha = ImageSequenceClip(alpha_frames, fps=fps)
rgb_clip_alpha.write_videofile(alpha_path, codec='mpeg4', logger=None, bitrate='2000k')
# rgb_clip_alpha.write_videofile(alpha_path)
alpha_frames.clear()

print("Done")
