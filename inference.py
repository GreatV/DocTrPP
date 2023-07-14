import argparse
import os
import warnings

import cv2
import numpy as np
import paddle
from paddle import nn
from PIL import Image

from GeoTr import GeoTr

warnings.filterwarnings("ignore")


class GeoTrP(nn.Layer):
    def __init__(self):
        super(GeoTrP, self).__init__()
        self.GeoTr = GeoTr()

    def forward(self, x):
        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        return bm


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = paddle.load(path)
        print(len(pretrained_dict.keys()))
        pretrained_dict = {
            k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict
        }
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.set_state_dict(model_dict)
        return model


def rec(opt):
    img_list = os.listdir(opt.distorrted_path)
    if not os.path.exists(opt.gsave_path):
        os.mkdir(opt.gsave_path)
    model = GeoTrP()
    reload_model(model.GeoTr, opt.GeoTr_path)
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
            bm = bm.cpu()
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = paddle.to_tensor(np.stack([bm0, bm1], axis=2)).unsqueeze(axis=0)
            out = nn.functional.grid_sample(
                paddle.to_tensor(im_ori)
                .transpose(perm=[2, 0, 1])
                .unsqueeze(axis=0)
                .astype(dtype="float32"),
                grid=lbl,
                align_corners=True,
            )
            img_geo = (
                (out[0] * 255)
                .transpose(perm=[1, 2, 0])
                .numpy()[:, :, ::-1]
                .astype(np.uint8)
            )
            cv2.imwrite(opt.gsave_path + name + "_geo" + ".png", img_geo)
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
