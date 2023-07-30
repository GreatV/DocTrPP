import paddle.nn.functional as F
import paddle_msssim
from paddle import nn


def unwarp(img, bm):
    n, c, h, w = img.shape
    x = bm
    perm_4 = list(range(x.ndim))
    perm_4[3] = 2
    perm_4[2] = 3
    x = x.transpose(perm=perm_4)
    perm_5 = list(range(x.ndim))
    perm_5[2] = 1
    perm_5[1] = 2
    bm = x.transpose(perm=perm_5)
    bm = F.upsample(x=bm, size=(h, w), mode="bilinear")
    x = bm
    perm_6 = list(range(x.ndim))
    perm_6[1] = 2
    perm_6[2] = 1
    x = x.transpose(perm=perm_6)
    perm_7 = list(range(x.ndim))
    perm_7[2] = 3
    perm_7[3] = 2
    bm = x.transpose(perm=perm_7)
    img = img.astype(dtype="float64")
    res = F.grid_sample(x=img, grid=bm)
    return res


class Unwarploss(nn.Layer):
    def __init__(self):
        super(Unwarploss, self).__init__()
        self.xmx, self.xmn, self.ymx, self.ymn = 0.0, 0.0, 0.0, 0.0

    def forward(self, inp, pred, label):
        n, c, h, w = inp.shape
        inp_img = inp[:, :3, :, :]
        pred = pred.astype(dtype="float64")
        label = label.astype(dtype="float64")
        uwpred = unwarp(inp_img, pred)
        uworg = unwarp(inp_img, label)
        loss_fn = nn.MSELoss()
        ssim_loss = paddle_msssim.ssim
        uloss = loss_fn(uwpred, uworg)
        ssim = 1 - ssim_loss(uwpred, uworg)
        del pred
        del label
        del inp
        return (
            uloss.astype(dtype="float32"),
            ssim.astype(dtype="float32"),
            uworg.astype(dtype="float32"),
            uwpred.astype(dtype="float32"),
        )
