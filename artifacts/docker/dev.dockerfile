FROM aliyunregistry.deepmirror.com.cn/dm/uranus-dev:presubmit-39f860e9930ee0dddb6e3f2c1385374c97cdcfff

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    accelerate==0.20.3 \
    transformers \
    cython \
    scikit-image \
    scipy \
    pillow \
    moviepy \
    pyyaml \
    lpips \
    kornia \
    focal_frequency_loss \
    plyfile \
    tensorboard \
    tensorflow==2.11.0 \
    tensorflow-addons==0.19.0
