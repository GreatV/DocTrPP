import collections
import os
import random

import cv2
import hdf5storage as h5
import imageio.v2 as imageio
import numpy as np
import paddle
from PIL import Image


class Doc3dDataset(paddle.io.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=512):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.files = collections.defaultdict(list)
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        for split in ["train", "val"]:
            path = os.path.join(self.root, split + ".txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        # os.path.join(self.root, "img", "{0}.png".format(im_name))
        img_foldr, file_name = os.path.split(im_name)
        recon_foldr = "chess48"
        wc_path = os.path.join(self.root, "wc", "{0}.exr".format(im_name))
        bm_path = os.path.join(self.root, "bm", "{0}.mat".format(im_name))
        alb_path = os.path.join(
            self.root,
            "recon",
            img_foldr,
            file_name[:-4] + recon_foldr + "0001.png",
        )
        wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        bm = h5.loadmat(bm_path)["bm"]
        alb = imageio.imread(alb_path, format="RGB")
        if self.is_transform:
            im, lbl = self.transform(wc, bm, alb)
        return im, lbl

    def tight_crop(self, wc, alb):
        msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(
            np.uint8
        )
        size = msk.shape
        [y, x] = msk.nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        wc = wc[miny : maxy + 1, minx : maxx + 1, :]
        alb = alb[miny : maxy + 1, minx : maxx + 1, :]
        s = 20
        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), "constant")
        alb = np.pad(alb, ((s, s), (s, s), (0, 0)), "constant")
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1
        wc = wc[cy1:-cy2, cx1:-cx2, :]
        alb = alb[cy1:-cy2, cx1:-cx2, :]
        t = miny - s + cy1
        b = size[0] - maxy - s + cy2
        left = minx - s + cx1
        right = size[1] - maxx - s + cx2
        return wc, alb, t, b, left, right

    def transform(self, wc, bm, alb):
        wc, alb, t, b, left, right = self.tight_crop(wc, alb)
        alb = np.array(Image.fromarray(alb).resize(self.img_size))
        alb = alb[:, :, ::-1]
        alb = alb.astype(np.float64)
        if alb.shape[2] == 4:
            alb = alb[:, :, :3]
        alb = alb.astype(float) / 255.0
        alb = alb.transpose(2, 0, 1)
        msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(
            np.uint8
        ) * 255
        xmx, xmn, ymx, ymn, zmx, zmn = (
            1.2539363,
            -1.2442188,
            1.2396319,
            -1.2289206,
            0.6436657,
            -0.67492497,
        )
        wc[:, :, 0] = (wc[:, :, 0] - zmn) / (zmx - zmn)
        wc[:, :, 1] = (wc[:, :, 1] - ymn) / (ymx - ymn)
        wc[:, :, 2] = (wc[:, :, 2] - xmn) / (xmx - xmn)
        wc = cv2.bitwise_and(wc, wc, mask=msk)
        wc = np.array(
            Image.fromarray((wc * 255).astype(np.uint8)).resize(self.img_size)
        )
        wc = wc.astype(float) / 255.0
        wc = wc.transpose(2, 0, 1)
        bm = bm.astype(float)
        bm[:, :, 1] = bm[:, :, 1] - t
        bm[:, :, 0] = bm[:, :, 0] - left
        bm = bm / np.array([448.0 - left - right, 448.0 - t - b])
        bm = (bm - 0.5) * 2
        bm0 = cv2.resize(bm[:, :, 0], (self.img_size[0], self.img_size[1]))
        bm1 = cv2.resize(bm[:, :, 1], (self.img_size[0], self.img_size[1]))
        img = np.concatenate([alb, wc], axis=0)
        lbl = np.stack([bm0, bm1], axis=-1)
        img = paddle.to_tensor(data=img).astype(dtype="float32")
        lbl = paddle.to_tensor(data=lbl).astype(dtype="float32")
        return img, lbl
