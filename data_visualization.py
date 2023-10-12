import argparse
import os
import random

import cv2
import hdf5storage as h5
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn.functional as F

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    nargs="?",
    type=str,
    default="~/datasets/doc3d/",
    help="Path to the downloaded dataset",
)
parser.add_argument(
    "--folder", nargs="?", type=int, default=1, help="Folder ID to read from"
)
parser.add_argument(
    "--output",
    nargs="?",
    type=str,
    default="output.png",
    help="Output filename for the image",
)

args = parser.parse_args()

root = os.path.expanduser(args.data_root)
folder = args.folder
dirname = os.path.join(root, "img", str(folder))

choices = [f for f in os.listdir(dirname) if "png" in f]
fname = random.choice(choices)

##### Read Image #####
img_path = os.path.join(dirname, fname)
img = cv2.imread(img_path)

##### Read 3D Coords #####
wc_path = os.path.join(root, "wc", str(folder), fname[:-3] + "exr")
wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# scale wc
# value obtained from the entire dataset
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

##### Read Backward Map #####
bm_path = os.path.join(root, "bm", str(folder), fname[:-3] + "mat")
bm = h5.loadmat(bm_path)["bm"]

##### Read UV Map #####
uv_path = os.path.join(root, "uv", str(folder), fname[:-3] + "exr")
uv = cv2.imread(uv_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

##### Read Depth Map #####
dmap_path = os.path.join(root, "dmap", str(folder), fname[:-3] + "exr")
dmap = cv2.imread(dmap_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
#### do some clipping and scaling to display it
dmap[dmap > 30.0] = 30
dmap = 1 - ((dmap - np.min(dmap)) / (np.max(dmap) - np.min(dmap)))

##### Read Normal Map #####
norm_path = os.path.join(root, "norm", str(folder), fname[:-3] + "exr")
norm = cv2.imread(norm_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

##### Read Albedo #####
alb_path = os.path.join(root, "alb", str(folder), fname[:-3] + "png")
alb = cv2.imread(alb_path)

##### Read Checkerboard Image #####
recon_path = os.path.join(root, "recon", str(folder), fname[:-8] + "chess480001.png")
recon = cv2.imread(recon_path)

##### Display image and GTs #####

# use the backward mapping to dewarp the image
# scale bm to -1.0 to 1.0
bm_ = bm / np.array([448, 448])
bm_ = (bm_ - 0.5) * 2
bm_ = np.reshape(bm_, (1, 448, 448, 2))
bm_ = paddle.to_tensor(bm_, dtype="float32")
img_ = alb.transpose((2, 0, 1)).astype(np.float32) / 255.0
img_ = np.expand_dims(img_, 0)
img_ = paddle.to_tensor(img_, dtype="float32")
uw = F.grid_sample(img_, bm_)
uw = uw[0].numpy().transpose((1, 2, 0))

f, axrr = plt.subplots(2, 5)
for ax in axrr:
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

axrr[0][0].imshow(img)
axrr[0][0].title.set_text("image")
axrr[0][1].imshow(wc)
axrr[0][1].title.set_text("3D coords")
axrr[0][2].imshow(bm[:, :, 0])
axrr[0][2].title.set_text("bm 0")
axrr[0][3].imshow(bm[:, :, 1])
axrr[0][3].title.set_text("bm 1")
if uv is None:
    uv = np.zeros_like(img)
axrr[0][4].imshow(uv)
axrr[0][4].title.set_text("uv map")
axrr[1][0].imshow(dmap)
axrr[1][0].title.set_text("depth map")
axrr[1][1].imshow(norm)
axrr[1][1].title.set_text("normal map")
axrr[1][2].imshow(alb)
axrr[1][2].title.set_text("albedo")
axrr[1][3].imshow(recon)
axrr[1][3].title.set_text("checkerboard")
axrr[1][4].imshow(uw)
axrr[1][4].title.set_text("gt unwarped")
plt.tight_layout()
plt.savefig(args.output)
