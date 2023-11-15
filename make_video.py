from pathlib import Path
import argparse
from PIL import Image
import glob
import cv2
import os
import numpy as np
import gc
from moviepy.editor import ImageSequenceClip


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output_path', type=str, default="")
parser.add_argument('--ckpt_path', type=str, default="weight/adampi_32p.pth")
parser.add_argument('--start_cnt', type=int, default=1)
parser.add_argument('--end_cnt', type=int, default=0)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--resize_factor', type=float, default=1.0)
parser.add_argument('--resize_factor_a', type=float, default=1.0)
opt, _ = parser.parse_known_args()

fps = 25

end_index = opt.end_cnt
if end_index < opt.start_cnt:
    images = glob.glob(opt.output_path + "/tmp/*.jpg")
    end_index = int(len(images) / 2)

# opencv video make better quality (even larger size)
print("make alpha video cv")
opencv_video_a = None
for i in range(opt.start_cnt, end_index, opt.interval):
    if i%30 == 0:
        print("process", i, "/", end_index)
    rgb = cv2.imread(opt.output_path + "/tmp/" + str(i) + "alpha.jpg")
    rgb_size = (int(rgb.shape[1] * opt.resize_factor_a), int(rgb.shape[0] * opt.resize_factor_a))
    rgb = cv2.resize(rgb, rgb_size)
    if opencv_video_a is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        opencv_video_a = cv2.VideoWriter(opt.output_path + '/cv_mpi_alpha.mp4', fourcc, 25, rgb_size)
    opencv_video_a.write(rgb)
opencv_video_a.release()

print("make rgb video cv")
opencv_video_l = None
opencv_video_r = None
for i in range(opt.start_cnt, end_index, opt.interval):
    if i%30 == 0:
        print("process", i, "/", end_index)
    rgb = cv2.imread(opt.output_path + "/tmp/" + str(i) + "rgb.jpg")
    rgb_size = (int(rgb.shape[1] * opt.resize_factor), int(rgb.shape[0] * opt.resize_factor))
    rgb = cv2.resize(rgb, rgb_size)

    sub_width = int(rgb_size[0] * 0.5)
    sub_size = (sub_width, rgb_size[1])
    rgb_l = rgb[:, 0:sub_width, :]
    rgb_r = rgb[:, sub_width:, :]
    if opencv_video_l is None:
        print(sub_size)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        opencv_video_l = cv2.VideoWriter(opt.output_path + '/cv_mpi_rgb_l.mp4', fourcc, 25, sub_size)
        opencv_video_r = cv2.VideoWriter(opt.output_path + '/cv_mpi_rgb_r.mp4', fourcc, 25, sub_size)

    opencv_video_l.write(rgb_l)
    opencv_video_r.write(rgb_r)
opencv_video_l.release()
opencv_video_r.release()

print("Done")
