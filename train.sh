export OPENCV_IO_ENABLE_OPENEXR=1

python -m paddle.distributed.launch --gpus=0,1 train.py --data_path  ~/datasets/doc3d
