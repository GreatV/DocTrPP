# DocTrPP

[DocTr++](https://github.com/fh2019ustc/DocTr-Plus) in PaddlePaddle.


## Train

- Windows

```powershell
$env:OPENCV_IO_ENABLE_OPENEXR=1
python .\train.py --data_path D:\datasets\doc3d\
```

- Linux and MacOS

```bash
OPENCV_IO_ENABLE_OPENEXR="1" python ./train.py --data_path ~/datasets/doc3d
```
