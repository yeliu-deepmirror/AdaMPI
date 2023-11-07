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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_path', type=str, default="")
parser.add_argument('--audio_path', type=str, default="")
opt, _ = parser.parse_known_args()

# python add_audio_to_video.py --video_path /AdaMPI/data/ipmanVR_LR.mp4 --audio_path /AdaMPI/data/audio.mp3

print("video_path:", opt.video_path)
print("audio_path:", opt.audio_path)

audio_clip = moviepy.editor.AudioFileClip(opt.audio_path)



if audio_clip is not None:
    print("add audio")
    video_clip = moviepy.editor.VideoFileClip(opt.video_path)
    audio_clip = audio_clip.subclip(0, video_clip.end)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(opt.video_path[:-4] + '_with_audio.mp4')


print("Done")
