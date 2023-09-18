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

start_cnt = 150
end_cnt = 377
fps = 25
resize_factor = 1

# opencv video make better quality (even larger size)
print("make rgb video cv")
opencv_video = None
for i in range(start_cnt, end_cnt):
    if i%30 == 0:
        print("process", i, "/", end_cnt)
    rgb = cv2.imread(opt.output_path + "/tmp/" + str(i) + "rgb.jpg")
    rgb_size = (int(rgb.shape[1] * resize_factor), int(rgb.shape[0] * resize_factor))
    rgb = cv2.resize(rgb, rgb_size)
    if opencv_video is None:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        opencv_video = cv2.VideoWriter(opt.output_path + '/cv_mpi_rgb.mp4', fourcc, 25, rgb_size)
    opencv_video.write(rgb)
opencv_video.release()

print("make alpha video cv")
opencv_video = None
for i in range(start_cnt, end_cnt):
    if i%30 == 0:
        print("process", i, "/", end_cnt)
    rgb = cv2.imread(opt.output_path + "/tmp/" + str(i) + "alpha.jpg")
    rgb_size = (int(rgb.shape[1] * resize_factor), int(rgb.shape[0] * resize_factor))
    rgb = cv2.resize(rgb, rgb_size)
    if opencv_video is None:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        opencv_video = cv2.VideoWriter(opt.output_path + '/cv_mpi_alpha.mp4', fourcc, 25, rgb_size)
    opencv_video.write(rgb)
opencv_video.release()


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
