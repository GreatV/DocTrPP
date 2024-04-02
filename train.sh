export OPENCV_IO_ENABLE_OPENEXR=1
export FLAGS_logtostderr=0
export CUDA_VISIBLE_DEVICES=0

python train.py --img-size 288 \
    --name "DocTr++" \
    --batch-size 12 \
    --lr 1e-4 \
    --exist-ok \
    --use-vdl
