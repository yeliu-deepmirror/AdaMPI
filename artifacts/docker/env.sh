conda create -n adampi python=3.8

# here we use pytorch 1.11.0 and CUDA 11.3 for an example

# install pytorch
pip install https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl

# install torchvision
pip install https://download.pytorch.org/whl/cu113/torchvision-0.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl

# install pytorch3d
conda install https://anaconda.org/pytorch3d/pytorch3d/0.6.2/download/linux-64/pytorch3d-0.6.2-py38_cu113_pyt1100.tar.bz2

# install other libs
pip install \
    numpy==1.19 \
    scikit-image==0.19.1 \
    scipy==1.8.0 \
    pillow==9.0.1 \
    opencv-python==4.4.0.40 \
    tqdm==4.64.0 \
    moviepy==1.0.3 \
    pyyaml \
    matplotlib \
    scikit-learn \
    lpips \
    kornia \
    focal_frequency_loss \
    tensorboard \
    transformers
