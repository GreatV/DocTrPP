import collections
import os

import cv2
import hdf5storage as h5
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision import transforms


class Doc3dDataset(Dataset):
    def __init__(self, root, split="train", img_size=512):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
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
        im_path = os.path.join(self.root, "img", f"{im_name}.png")
        bm_path = os.path.join(self.root, "bm", f"{im_name}.mat")
        im = cv2.imread(im_path)
        im = self.transform(im)
        # im = im.transpose((2, 0, 1))

        bm = h5.loadmat(bm_path)["bm"]
        bm = bm / np.array([448, 448])
        bm = (bm - 0.5) * 2
        bm0 = cv2.resize(bm[:, :, 0], (self.img_size[0], self.img_size[1]))
        bm1 = cv2.resize(bm[:, :, 1], (self.img_size[0], self.img_size[1]))
        bm = np.stack([bm0, bm1], axis=-1)

        bm = paddle.to_tensor(bm, dtype="float32")

        return im, bm
