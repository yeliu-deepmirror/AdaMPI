from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DPTForDepthEstimation, DPTImageProcessor

from utils.utils import (
    image_to_tensor,
    disparity_to_tensor,
    render_3dphoto,
    write_mpi_to_binary,
)
from model.AdaMPI import MPIPredictor


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_path', type=str, default="images/music.png")
parser.add_argument('--disp_path', type=str, default=None)
parser.add_argument('--width', type=int, default=384)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--save_path', type=str, default="debug/music.mp4")
parser.add_argument('--ckpt_path', type=str, default="weight/adampi_32p.pth")
opt, _ = parser.parse_known_args()


# load input
image = image_to_tensor(opt.img_path).cuda()  # [1,3,h,w]
if opt.disp_path is not None:
    disp = disparity_to_tensor(opt.disp_path).cuda()  # [1,1,h,w]
else:
    # Use MiDaS to generate depth map
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").cuda()
    image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    with torch.no_grad():
        image_tmp = Image.open(opt.img_path).convert("RGB")
        inputs = image_processor(image_tmp, return_tensors="pt")
        midas_depth = model(pixel_values=inputs['pixel_values'].cuda()).predicted_depth.unsqueeze(1)

        # Dump depth map for debugging
        midas_depth_np = output = F.interpolate(midas_depth, size=(opt.height, opt.width), mode="bilinear", align_corners=True).squeeze().cpu().numpy()
        formatted = (midas_depth_np * 255 / np.max(midas_depth_np)).astype("uint8")
        Path("debug").mkdir(parents=True, exist_ok=True)
        Image.fromarray(formatted).save("debug/midas_depth.png")

        disp = midas_depth / torch.max(midas_depth)

image = F.interpolate(image, size=(opt.height, opt.width), mode='bilinear', align_corners=True)
disp = F.interpolate(disp, size=(opt.height, opt.width), mode='bilinear', align_corners=True)

print("image", image.shape)
print("depth", disp.shape)

# load pretrained model
ckpt = torch.load(opt.ckpt_path)
model = MPIPredictor(
    width=opt.width,
    height=opt.height,
    num_planes=ckpt["num_planes"],
)
model.load_state_dict(ckpt["weight"])
model = model.cuda()
model = model.eval()

# predict MPI planes
print("predict MPI planes...")
with torch.no_grad():
    pred_mpi_planes, pred_mpi_disp = model(image, disp)  # [b,s,4,h,w]
    write_mpi_to_binary(image, pred_mpi_planes, pred_mpi_disp, opt.save_path + ".bin")

# render 3D photo
K = torch.tensor([
    [0.58, 0, 0.5],
    [0, 0.58, 0.5],
    [0, 0, 1]
]).cuda()
K[0, :] *= opt.width
K[1, :] *= opt.height
K = K.unsqueeze(0)

print("render video...")
render_3dphoto(
    image,
    pred_mpi_planes,
    pred_mpi_disp,
    K,
    K,
    opt.save_path,
)
