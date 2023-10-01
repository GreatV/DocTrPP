import collections
import os
import random

import cv2
import hdf5storage as h5
import numpy as np
import paddle
from paddle import io

# Set random seed
random.seed(12345678)


class Doc3dDataset(io.Dataset):
    def __init__(self, root, split="train", is_transform=False, image_size=512):
        self.root = os.path.expanduser(root)

        self.split = split
        self.is_transform = is_transform

        self.files = collections.defaultdict(list)

        self.image_size = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )

        for split in ["train", "val"]:
            path = os.path.join(self.root, split + ".txt")
            file_list = []
            with open(path, "r") as file:
                file_list = [file_id.rstrip() for file_id in file.readlines()]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        image_name = self.files[self.split][index]

        # Read image
        image_path = os.path.join(self.root, "img", image_name + ".png")
        image = cv2.imread(image_path)

        # Read 3D Coordinates
        wc_path = os.path.join(self.root, "wc", image_name + ".exr")
        wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        # Read backward map
        bm_path = os.path.join(self.root, "bm", image_name + ".mat")
        bm = h5.loadmat(bm_path)["bm"]

        if self.is_transform:
            image, bm = self.transform(wc, bm, image)

        return image, bm

    def tight_crop(self, wc: np.ndarray):
        mask = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(
            np.uint8
        )
        mask_size = mask.shape
        [y, x] = mask.nonzero()
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)

        wc = wc[min_y : max_y + 1, min_x : max_x + 1, :]
        s = 20
        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), "constant")

        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        wc = wc[cy1:-cy2, cx1:-cx2, :]

        top: int = min_y - s + cy1
        bottom: int = mask_size[0] - max_y - s + cy2
        left: int = min_x - s + cx1
        right: int = mask_size[1] - max_x - s + cx2

        top = np.clip(top, 0, mask_size[0])
        bottom = np.clip(bottom, 1, mask_size[0] - 1)
        left = np.clip(left, 0, mask_size[1])
        right = np.clip(right, 1, mask_size[1] - 1)

        return wc, top, bottom, left, right

    def transform(self, wc, bm, img):
        wc, top, bottom, left, right = self.tight_crop(wc)

        img = img[top:-bottom, left:-right, :]
        cv2.imwrite("img.png", img)

        # resize image
        img = cv2.resize(img, self.image_size)
        img = img[:, :, ::-1]
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # resize bm
        bm = bm.astype(np.float32)
        bm[:, :, 1] = bm[:, :, 1] - top
        bm[:, :, 0] = bm[:, :, 0] - left
        bm = bm / np.array([448.0 - left - right, 448.0 - top - bottom])
        bm = (bm - 0.5) * 2
        bm0 = cv2.resize(bm[:, :, 0], (self.image_size[0], self.image_size[1]))
        bm1 = cv2.resize(bm[:, :, 1], (self.image_size[0], self.image_size[1]))

        bm = np.stack([bm0, bm1], axis=-1)

        img = paddle.to_tensor(img).astype(dtype="float32")
        bm = paddle.to_tensor(bm).astype(dtype="float32")

        return img, bm


if __name__ == "__main__":
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    dataset = Doc3dDataset(
        root="~/datasets/doc3d/", split="train", is_transform=True, image_size=288
    )
    img, bm = dataset[0]
