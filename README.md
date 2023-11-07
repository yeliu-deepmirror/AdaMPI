## News
* `2023/7/31` We release the complete training code for our method on the COCO dataset!
* `2022/8/4` We release the code and the pretrained model for *AdaMPI*. Use it to create 3D photo from single in-the-wild 2D images!
* `2022/7/22` We release the code and the pretrained inpainting model for the *Warp-Back* strategy.

# AdaMPI

<img src="misc/teaser.gif" width="100%" >

This is a PyTorch implementation of the following paper:

**Single-View View Synthesis in the Wild with Learned Adaptive Multiplane Images**, SIGGRAPH 2022.

Yuxuan Han, Ruicheng Wang and Jiaolong Yang

[Project Page](https://yxuhan.github.io/AdaMPI/index.html) | [Video](https://www.youtube.com/watch?v=1tzYr8dp0oY) | [Paper](https://arxiv.org/abs/2205.11733)

<img src="misc/framework.png" width="100%" >

**Abstract**: *This paper deals with the challenging task of synthesizing novel views for in-the-wild photographs. Existing methods have shown promising results leveraging monocular depth estimation and color inpainting with layered depth representations. However, these methods still have limited capability to handle scenes with complex 3D geometry. We propose a new method based on the multiplane image (MPI) representation. To accommodate diverse scene layouts in the wild and tackle the difficulty in producing high-dimensional MPI contents, we design a network structure that consists of two novel modules, one for plane depth adjustment and another for depth-aware color prediction. The former adjusts the initial plane positions using the RGBD context feature and an attention mechanism. Given adjusted depth values, the latter predicts the color and density for each plane separately with proper inter-plane interactions achieved via a feature masking strategy. To train our method, we construct large-scale stereo training data using only unconstrained single-view image collections by a simple yet effective warp-back strategy.*

## Environment

run with docker:
```shell
IMG=mpi dev
```

create conda env
```shell
conda env create -f environment.yml
conda activate mpi
```

run generation video in cpu001
```shell
python gen_3dphoto_video.py --video_path debug/yard_friend.mp4 --output_path /mnt/gz01/experiment/liuye/mpi
python gen_3dphoto_video.py --video_path debug/yaorenmao.mp4 --output_path /mnt/gz01/experiment/liuye/mpi
```

Run in Determined
```
VIDEO_PATH=/mnt/data/yeliu/MPI/forest.mp4
OUTPUT_PATH=/mnt/data/yeliu/MPI/forest
WEIGHT_PATH=/mnt/data/yeliu/models/mpi_weight

python gen_3dphoto_video.py \
--video_path ${VIDEO_PATH} --output_path ${OUTPUT_PATH} \
--ckpt_path ${WEIGHT_PATH}/adampi_32p.pth --cache_dir ${WEIGHT_PATH} --device cuda
```

```shell
python make_video.py --output_path /mnt/gz01/experiment/liuye/mpi --start_cnt=100 --end_cnt=800
```

generate VR LR video
```shell
python gen_vr_180.py --output_path /mnt/gz01/experiment/liuye/mpi/video_mpi_yangtuo
python gen_vr_180.py --output_path /mnt/ml-experiment-data/yeliu/MPI/ipman --video_path /mnt/ml-experiment-data/yeliu/MPI/ipman_audio.mp4
python gen_vr_180.py --output_path /mnt/data/yeliu/MPI/ipman --video_path /mnt/data/yeliu/MPI/ipman_audio.mp4
```



## Document
This repo contain two parts:
* The train and inference code for the *AdaMPI*. You can use it to generate 3D photo from single-view in-the-wild image input, as shown in the [teaser gif](misc/teaser.gif). See [AdaMPI.md](doc/AdaMPI.md) for more details.

* The `clean and abstract` implementation of the *Warp-Back* strategy, you can use it to generate multi-view training data to train your own view synthesis model (e.g. [SynSin](https://github.com/facebookresearch/synsin), [WorldSheet](https://github.com/facebookresearch/worldsheet)) using only single-view image collections like [COCO](https://github.com/nightrome/cocostuff) dataset. See [Warp-Back.md](doc/Warp-Back.md) for more details.

## Contact
If you have any questions, please contact Yuxuan Han (hanyx22@mails.tsinghua.edu.cn).

## License and Citation
This repository can only be used for personal/research/non-commercial purposes.
Please cite the following paper if this model helps your research:

    @inproceedings{han2022single,
        author = {Han, Yuxuan and Wang, Ruicheng and Yang, Jiaolong},
        title = {Single-View View Synthesis in the Wild with Learned Adaptive Multiplane Images},
        booktitle = {ACM SIGGRAPH},
        year={2022}
    }

## Acknowledgments
* The code for inference and training is heavily borrowed from [MINE](https://github.com/vincentfung13/MINE), we thank the author for their great effort.
* We thank [Varun Jampani](https://varunjampani.github.io/) and [Meng-Li Shih](https://shihmengli.github.io/) for helpful suggestions and discussions to evaluate their method.
