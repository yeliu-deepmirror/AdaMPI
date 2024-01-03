from pathlib import Path
import argparse
from PIL import Image
import cv2
import glob
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
    process_image_with_depth,
    write_array,
    merge_rgba_layers,
)
from model.AdaMPI import MPIPredictor


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--session_path', type=str)
parser.add_argument('--output_path', type=str, default="data")
parser.add_argument('--ckpt_path', type=str, default="weight/adampi_32p.pth")
parser.add_argument('--cache_dir', type=str, default="weight")
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--resize_factor', type=float, default=1.0)

opt, _ = parser.parse_known_args()

image_paths = glob.glob(opt.session_path + "/frame*.jpg")
frame_count = len(image_paths)
print("read", len(image_paths), "image_paths")
image_paths.sort()

images_0 = cv2.imread(image_paths[0]);
print("image_paths size", images_0.shape)

# process width height should be 128 * n
n_w = max(int(opt.resize_factor * images_0.shape[1] / 128), 1)
n_h = max(int(opt.resize_factor * images_0.shape[0] / 128), 1)
width = n_w * 128
height = n_h * 128
print("process size", width, height)

# load pretrained model
ckpt = torch.load(opt.ckpt_path)
model_mpi = MPIPredictor(
    width=width,
    height=height,
    num_planes=ckpt["num_planes"],
)
model_mpi.load_state_dict(ckpt["weight"])
if opt.device == "cuda":
    model_mpi = model_mpi.cuda()
model_mpi = model_mpi.eval()

# predict MPI planes
print("predict MPI planes...")

os.makedirs(opt.output_path, exist_ok=True)
os.makedirs(opt.output_path + "/tmp", exist_ok=True)
# Read until video is completed
for i in range(0, frame_count):
    depth_path = image_paths[i].replace("frame", "depth").replace("jpg", "png")
    print("process", i, "/", frame_count, image_paths[i], depth_path)

    depth_np = cv2.imread(depth_path)[:, :, 0].astype(np.float32)
    depth_np = np.expand_dims(depth_np, axis=(0, 1))
    with torch.no_grad():
        rgba_layers, mpi_depths = process_image_with_depth(image_paths[i], depth_np, (height, width), model_mpi, opt.device)

    # align the mpi depth to raw depth
    depth_align = torch.from_numpy(depth_np)
    depth_align = F.interpolate(depth_align, size=(height, width), mode='bilinear', align_corners=True).squeeze()

    mask_col = None
    for layer in range(rgba_layers.shape[0]):
        alpha = rgba_layers[layer, :, :, 3]
        mask = (alpha > 0.8)
        if mask_col is None:
            mask_col = mask
        else:
            mask = np.logical_and(np.logical_not(mask_col), mask)
            mask_col = np.logical_and(mask_col, mask)
        depth_masked = depth_align[mask]
        print(depth_masked)
        depth_mean = depth_masked.mean()

        print(layer, mpi_depths[layer], depth_mean)

    # rgba_layers[layer_id, :, :, :]

    rgb, alpha = merge_rgba_layers(rgba_layers)
    # print(large_rgba_image.shape)
    cv2.imwrite(opt.output_path + "/tmp/" + str(i) + "rgb.jpg", rgb)
    cv2.imwrite(opt.output_path + "/tmp/" + str(i) + "alpha.jpg", alpha)
    # save the depth_paths (in reversed order to fit MPI order)
    with open(opt.output_path + "/tmp/" + str(i) + "depths.npy", 'wb') as f:
        mpi_depths_inv = np.flip(mpi_depths)
        np.save(f, mpi_depths_inv)

    gc.collect()

print("Done")
