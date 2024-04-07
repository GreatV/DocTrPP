# DocTrPP: DocTr++ in PaddlePaddle

## Introduction

This is a PaddlePaddle implementation of DocTr++. The original paper is [DocTr++: Deep Unrestricted Document Image Rectification](https://arxiv.org/abs/2304.08796). The original code is [here](https://github.com/fh2019ustc/DocTr-Plus).

![demo](https://github.com/GreatV/DocTrPP/assets/17264618/4e491512-bfc4-4e69-a833-fd1c6e17158c)

## Requirements

- Python 3.10
- paddlepaddle-gpu 2.5.1

For dev environment, you may need install the following extra packages:

- [pre-commit](https://pre-commit.com/)
- [PaDiff](https://github.com/PaddlePaddle/PaDiff)
- pytest

## Training

1. Data Preparation

To prepare datasets, refer to [doc3D](https://github.com/cvlab-stonybrook/doc3D-dataset).

2. Training

```shell
sh train.sh
```

or

```shell
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0

python train.py --img-size 288 \
    --name "DocTr++" \
    --batch-size 12 \
    --lr 2.5e-5 \
    --exist-ok \
    --use-vdl
```

3. Load Trained Model and Continue Training

```shell
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0

python train.py --img-size 288 \
    --name "DocTr++" \
    --batch-size 12 \
    --lr 2.5e-5 \
    --resume "runs/train/DocTr++/weights/last.ckpt" \
    --exist-ok \
    --use-vdl
```

## Test and Inference

Test the dewarp result on a single image:

```shell
python predict.py -i "crop/12_2 copy.png" -m runs/train/DocTr++/weights/best.ckpt -o 12.2.png
```
![document image rectification](https://raw.githubusercontent.com/greatv/DocTrPP/main/doc/imgs/document_image_rectification.jpg)

## Model Download

The trained model can be downloaded from [here](https://github.com/GreatV/DocTrPP/releases/download/v0.0.2/best.ckpt).
