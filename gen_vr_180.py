import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import argparse
from PIL import Image
import cv2
import glob
import os
import numpy as np
import gc
import moviepy
from moviepy.editor import ImageSequenceClip
import tensorflow as tf
from utils.render import mpi

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_path', type=str, default="")
parser.add_argument('--output_path', type=str, default="debug/video_mpi")
parser.add_argument('--output_name', type=str, default="VR_LR")
parser.add_argument('--start_cnt', type=int, default=1)
parser.add_argument('--end_cnt', type=int, default=0)
parser.add_argument('--interval', type=int, default=1)
parser.add_argument('--len_distance', type=float, default=0.01)
opt, _ = parser.parse_known_args()

# python gen_vr_180.py --output_path /mnt/gz01/experiment/liuye/mpi/video_mpi_mom_not_home --start_cnt=1 --end_cnt=300
depths_raw = mpi.make_depths(1.0, 100.0, 8 * 4).numpy()
print("depths_raw mpi:", depths_raw)

fps = 30

audio_clip = None
if opt.video_path:
    video = moviepy.editor.VideoFileClip(opt.video_path)
    fps = video.fps
    print("video fps :", fps)
    audio_clip = video.audio
    if audio_clip is not None:
        print("load audio from :", opt.video_path)
        audio_clip.write_audiofile(opt.output_path + "/audio.mp3")


def render(alphas, rgbs, depths, cols = 8, rows = 4):
    # make alpha and rgb to layers : [..., L, H, W, C+1] MPI layers back to front, alpha in last channel.
    ele_width = int(alphas.shape[1] / cols)
    ele_height = int(alphas.shape[0] / rows)

    images = []
    # convert_func = tf.convert_to_tensor(convert_func, dtype=tf.int32)
    for i in range(cols * rows):
        col_id = i%8
        row_id = int(i/8)
        rgb = rgbs[ele_height*row_id : ele_height*(row_id+1), ele_width*col_id : ele_width*(col_id+1), :]
        alpha = alphas[ele_height*row_id : ele_height*(row_id+1), ele_width*col_id : ele_width*(col_id+1), 0]
        alpha = np.expand_dims(alpha, axis=2)
        rgba = np.concatenate([rgb, alpha], axis=2)
        images.append(rgba)
    layers_np = np.array(images, dtype=np.float32) / 255.0
    layers = tf.convert_to_tensor(layers_np, dtype=tf.float32)

    # The reference camera position can just be the identity
    # depths = mpi.make_depths(1.0, 100.0, cols * rows).numpy()
    reference_pose = tf.constant(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    # Accurate intrinsics are only important if we are trying to match a ground
    # truth output. Here we just give intrinsics for a 16:9 image with the
    # principal point in the center.
    intrinsics = tf.constant([1.0, 1.0 * ele_width/ele_height, 0.5, 0.5])

    x_offsets = [-opt.len_distance, opt.len_distance]
    rended_images = []
    for x_offset in x_offsets:
        target_pose = tf.constant(
            [[1.0, 0.0, 0.0, x_offset], [0.0, 1.0, 0.0, 0], [0.0, 0.0, 1.0, 0]])

        image = mpi.render(layers, depths,
                           reference_pose, intrinsics,   # Reference view
                           target_pose, intrinsics,      # Target view
                           height=ele_height, width=ele_width)
        image_cv = (255.0 * image.numpy()).astype(np.uint8)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        rended_images.append(image_cv)

    # make LR image
    image_LR = np.concatenate([rended_images[1], rended_images[0]], axis=1)
    return image_LR

end_index = opt.end_cnt
if end_index < opt.start_cnt:
    # process full
    images = glob.glob(opt.output_path + "/tmp/*.jpg")
    end_index = int(len(images) / 2)

# opencv video make better quality (even larger size)
output_video_path = opt.output_path + "/" + opt.output_name + '.mp4'
print("make VR LR 180 video cv.", opt.start_cnt, opt.interval, end_index)
print("save to", output_video_path)
opencv_video_vr = None
for i in range(opt.start_cnt, end_index, opt.interval):
    if i%30 == 0:
        print("process", i, "/", end_index)
    alphas = cv2.imread(opt.output_path + "/tmp/" + str(i) + "alpha.jpg")
    rgbs = cv2.imread(opt.output_path + "/tmp/" + str(i) + "rgb.jpg")

    # read depth if available
    depths = depths_raw
    depths_file = opt.output_path + "/tmp/" + str(i) + "depths.npy"
    if os.path.exists(depths_file):
        with open(depths_file, 'rb') as f:
            depths = np.load(f).squeeze()
    else:
        print("Warning: depth not found for", i)

    # render LR images
    image_LR = render(alphas, rgbs, depths)
    image_LR_size = (int(image_LR.shape[1]), int(image_LR.shape[0]))

    # write the image
    if opencv_video_vr is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        opencv_video_vr = cv2.VideoWriter(output_video_path, fourcc, fps, image_LR_size)
    opencv_video_vr.write(image_LR)

if opencv_video_vr is not None:
    opencv_video_vr.release()


if audio_clip is not None:
    video_clip = moviepy.editor.VideoFileClip(output_video_path)
    audio_clip = audio_clip.subclip(0, video_clip.end)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(opt.output_path + "/" + opt.output_name + '_with_audio.mp4')


print("Done")
