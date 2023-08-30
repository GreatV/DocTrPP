import argparse
import os
import warnings

import cv2
import numpy as np
import paddle
from paddle.nn import functional as F
from PIL import Image

from GeoTr import GeoTr

warnings.filterwarnings("ignore")


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        pretrained_dict = paddle.load(path)
        model.set_state_dict(pretrained_dict["model_state"])
        return model


def rec(opt):
    img_list = os.listdir(opt.distorrted_path)
    if not os.path.exists(opt.gsave_path):
        os.mkdir(opt.gsave_path)
    model = GeoTr()
    reload_model(model, opt.GeoTr_path)
    model.eval()
    for img_path in img_list:
        name = img_path.split(".")[-2]
        img_path = opt.distorrted_path + img_path
        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.0
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = paddle.to_tensor(im).astype(dtype="float32").unsqueeze(axis=0)
        with paddle.no_grad():
            bm = model(im)
            print(bm.shape)
            x = bm
            perm_0 = list(range(x.ndim))
            perm_0[1] = 2
            perm_0[2] = 1
            x = x.transpose(perm=perm_0)
            perm_1 = list(range(x.ndim))
            perm_1[2] = 3
            perm_1[3] = 2
            bm = x.transpose(perm=perm_1)
            bm = bm.cpu()
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
            bm = np.stack([bm0, bm1], axis=-1)
            print(bm.shape)
            bm_ = np.reshape(bm, (1, 448, 448, 2))
            bm_ = paddle.to_tensor(bm_, dtype=paddle.float32)
            img_ = im_ori.transpose((2, 0, 1)).astype(np.float32) / 255.0
            img_ = np.expand_dims(img_, 0)
            img_ = paddle.to_tensor(img_)
            uw = F.grid_sample(img_, bm_)
            uw = uw[0].numpy().transpose((1, 2, 0))
            gt_uw = cv2.cvtColor(uw, cv2.COLOR_RGB2BGR)
            gt_uw = cv2.normalize(
                gt_uw,
                dst=None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            cv2.imwrite(opt.gsave_path + name + "_geo" + ".png", gt_uw)
        print("Done: ", img_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distorrted_path", default="./distorted/")
    parser.add_argument("--gsave_path", default="./rectified/")
    parser.add_argument("--GeoTr_path", default="./model_pretrained/DocTrP.pth")
    opt = parser.parse_args()
    rec(opt)


if __name__ == "__main__":
    main()
