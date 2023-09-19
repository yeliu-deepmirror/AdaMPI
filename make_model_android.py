from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from transformers import DPTForDepthEstimation, DPTImageProcessor
from model.AdaMPI import MPIPredictor

class MPIPredictorFull(nn.Module):
    def __init__(self, width, height, ckpt_path):
        super(MPIPredictorFull, self).__init__()
        self.model_depth_ = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
        self.image_processor_ = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        self.width_ = width
        self.height_ = height
        ckpt = torch.load(ckpt_path)
        self.model_mpi_ = MPIPredictor(
            width=width,
            height=height,
            num_planes=ckpt["num_planes"],
        )
        self.model_mpi_.load_state_dict(ckpt["weight"])
        # self.model_mpi = model_mpi.eval()

    def forward(self, src_imgs):
        inputs = self.image_processor_(src_imgs, return_tensors="pt")
        midas_depth = self.model_depth_(pixel_values=inputs['pixel_values']).predicted_depth.unsqueeze(1)

        disp = midas_depth / torch.max(midas_depth)

        image = F.interpolate(src_imgs, size=(self.height_, self.width_), mode='bilinear', align_corners=True)
        disp = F.interpolate(disp, size=(self.height_, self.width_), mode='bilinear', align_corners=True)

        return self.model_mpi_(image, disp)  # [b,s,4,h,w]


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output_path', type=str, default="weight/adampi_32p.ptl")
parser.add_argument('--width', type=int, default=384)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--ckpt_path', type=str, default="weight/adampi_32p.pth")
opt, _ = parser.parse_known_args()

# https://pytorch.org/mobile/android/

model_mpi = MPIPredictorFull(
    width=opt.width,
    height=opt.height,
    ckpt_path=opt.ckpt_path,
)
model_mpi = model_mpi.eval()

# model = torchvision.models.mobilenet_v2(pretrained=True)
# model.eval()
example_rgb = torch.rand(1, 3, opt.height, opt.width)
traced_script_module = torch.jit.trace(model_mpi, example_rgb)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter(opt.output_path)
