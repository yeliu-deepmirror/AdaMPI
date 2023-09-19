from pathlib import Path
import argparse
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from model.AdaMPI import MPIPredictor

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output_path', type=str, default="weight/adampi_32p.ptl")
parser.add_argument('--width', type=int, default=384)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--ckpt_path', type=str, default="weight/adampi_32p.pth")
opt, _ = parser.parse_known_args()

# https://pytorch.org/mobile/android/
ckpt = torch.load(opt.ckpt_path)
model_mpi = MPIPredictor(
    width=opt.width,
    height=opt.height,
    num_planes=ckpt["num_planes"],
)
model_mpi.load_state_dict(ckpt["weight"])
model_mpi = model_mpi.eval()

# model = torchvision.models.mobilenet_v2(pretrained=True)
# model.eval()
example_rgb = torch.rand(1, 3, opt.height, opt.width)
example_dep = torch.rand(1, 1, opt.height, opt.width)
traced_script_module = torch.jit.trace(model_mpi, (example_rgb, example_dep))
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter(opt.output_path)
