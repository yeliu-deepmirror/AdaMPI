from pathlib import Path
import argparse
from PIL import Image
import cv2
import os
import numpy as np
import gc
from moviepy.editor import ImageSequenceClip


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output_path', type=str, default="debug/video_mpi")
parser.add_argument('--ckpt_path', type=str, default="weight/adampi_32p.pth")
opt, _ = parser.parse_known_args()

start_cnt = 1
end_cnt = 50

fps = 25
resize_factor = 1
resize_factor_a = 0.25

print("make alpha video cv")
opencv_video_a = None
for i in range(start_cnt, end_cnt):
    if i%30 == 0:
        print("process", i, "/", end_cnt)
    rgb = cv2.imread(opt.output_path + "/tmp/" + str(i) + "alpha.jpg")
    rgb_size = (int(rgb.shape[1] * resize_factor_a), int(rgb.shape[0] * resize_factor_a))
    rgb = cv2.resize(rgb, rgb_size)
    if opencv_video_a is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        opencv_video_a = cv2.VideoWriter(opt.output_path + '/cv_mpi_alpha.mp4', fourcc, 25, rgb_size)
    opencv_video_a.write(rgb)
opencv_video_a.release()

# opencv video make better quality (even larger size)
print("make rgb video cv")
opencv_video_1 = None
opencv_video_2 = None
opencv_video_3 = None
opencv_video_4 = None
for i in range(start_cnt, end_cnt):
    if i%30 == 0:
        print("process", i, "/", end_cnt)
    rgb = cv2.imread(opt.output_path + "/tmp/" + str(i) + "rgb.jpg")
    rgb_size = (int(rgb.shape[1] * resize_factor), int(rgb.shape[0] * resize_factor))
    rgb = cv2.resize(rgb, rgb_size)

    sub_width = int(rgb_size[0] * 0.25)
    sub_size = (sub_width, rgb_size[1])
    rgb_1 = rgb[:, 0:sub_width, :]
    rgb_2 = rgb[:, sub_width:2*sub_width, :]
    rgb_3 = rgb[:, 2*sub_width:3*sub_width, :]
    rgb_4 = rgb[:, 3*sub_width:, :]
    if opencv_video_1 is None:
        print(rgb_1.shape)
        print(rgb_2.shape)
        print(rgb_3.shape)
        print(rgb_4.shape)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        opencv_video_1 = cv2.VideoWriter(opt.output_path + '/cv_mpi_rgb_1.mp4', fourcc, 25, sub_size)
        opencv_video_2 = cv2.VideoWriter(opt.output_path + '/cv_mpi_rgb_2.mp4', fourcc, 25, sub_size)
        opencv_video_3 = cv2.VideoWriter(opt.output_path + '/cv_mpi_rgb_3.mp4', fourcc, 25, sub_size)
        opencv_video_4 = cv2.VideoWriter(opt.output_path + '/cv_mpi_rgb_4.mp4', fourcc, 25, sub_size)

    opencv_video_1.write(rgb_1)
    opencv_video_2.write(rgb_2)
    opencv_video_3.write(rgb_3)
    opencv_video_4.write(rgb_4)
opencv_video_1.release()
opencv_video_2.release()
opencv_video_3.release()
opencv_video_4.release()


if False:
    print("make rgb video")
    rgb_frames = []
    for i in range(start_cnt, end_cnt):
        if i%30 == 0:
            print("process", i, "/", end_cnt)
        rgb = cv2.imread(opt.output_path + "/tmp/" + str(i) + "rgb.jpg")
        rgb_frames.append(cv2.resize(rgb, (int(rgb.shape[1] * resize_factor), int(rgb.shape[0] * resize_factor))))
        gc.collect()

    video_path = opt.output_path + '/mpi_rgb.mp4'
    rgb_clip = ImageSequenceClip(rgb_frames, fps=fps)
    # https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html#moviepy.video.VideoClip.VideoClip.write_videofile
    rgb_clip.write_videofile(video_path, codec='mpeg4', bitrate='8000k')
    rgb_frames.clear()
    gc.collect()

    print("make alpha video")
    alpha_frames = []
    for i in range(start_cnt, end_cnt):
        if i%30 == 0:
            print("process", i, "/", end_cnt)
        alpha = cv2.imread(opt.output_path + "/tmp/" + str(i) + "alpha.jpg")
        alpha_frames.append(cv2.resize(alpha, (int(alpha.shape[1] * resize_factor), int(alpha.shape[0] * resize_factor))))

    # MP4 not supported alpha channel so make a new file for alpha
    alpha_path = opt.output_path + '/mpi_alpha.mp4'
    rgb_clip_alpha = ImageSequenceClip(alpha_frames, fps=fps)
    rgb_clip_alpha.write_videofile(alpha_path, codec='mpeg4', bitrate='2000k')
    alpha_frames.clear()

print("Done")
