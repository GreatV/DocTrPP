import argparse
import inspect
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from loguru import logger
from paddle.io import DataLoader
from paddle.nn import functional as F
from paddle_msssim import ms_ssim, ssim

from doc3d_dataset import Doc3dDataset
from GeoTr import GeoTr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
RANK = int(os.getenv("RANK", -1))


def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    if deterministic:
        os.environ["FLAGS_cudnn_deterministic"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code,
    # i.e.  colorstr('blue', 'hello world')

    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string

    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }

    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    logger.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path,
    # i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def train(args):
    save_dir = Path(args.save_dir)

    # Directories
    weights_dir = save_dir / "weights"
    weights_dir.parent.mkdir(parents=True, exist_ok=True)

    last = weights_dir / "last.ckpt"
    best = weights_dir / "best.ckpt"

    # Hyperparameters

    # Config
    init_seeds(args.seed)

    # Train loader
    train_dataset = Doc3dDataset(
        args.data_root,
        split="train",
        is_transform=True,
        image_size=args.img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    # Validation loader
    val_dataset = Doc3dDataset(
        args.data_root,
        split="val",
        is_transform=True,
        image_size=args.img_size,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.workers
    )

    # Model
    model = GeoTr()

    # Data Parallel Mode
    if RANK == -1 and paddle.device.cuda.device_count() > 1:
        model = paddle.DataParallel(model)

    # Scheduler
    scheduler = optim.lr.OneCycleLR(
        max_learning_rate=args.lr,
        total_steps=args.epochs * len(train_loader),
        phase_pct=0.1,
    )

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=scheduler,
        parameters=model.parameters(),
    )

    # loss function
    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if args.resume:
        ckpt = paddle.load(args.resume)
        model.set_state_dict(ckpt["model"])
        optimizer.set_state_dict(ckpt["optimizer"])
        scheduler.set_state_dict(ckpt["scheduler"])
        best_fitness = ckpt["best_fitness"]
        start_epoch = ckpt["epoch"] + 1

    # Train
    for epoch in range(start_epoch, args.epochs):
        model.train()

        for i, (img, target) in enumerate(train_loader):
            optimizer.clear_grad()

            img = paddle.to_tensor(img)
            target = paddle.to_tensor(target)

            pred = model(img)
            pred_nhwc = pred.transpose([0, 2, 1, 3]).transpose([0, 1, 3, 2])

            loss = l1_loss_fn(pred_nhwc, target)
            mse_loss = mse_loss_fn(pred_nhwc, target)

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if i % 10 == 0:
                logger.info(
                    f"[TRAIN MODE] Epoch: {epoch}, Iter: {i}, L1 Loss: {float(loss)}, "
                    f"MSE Loss: {float(mse_loss)}, LR: {float(scheduler.get_lr())}"
                )

        # Validation
        model.eval()

        with paddle.no_grad():
            fitness = paddle.zeros([])

            for i, (img, target) in enumerate(val_loader):
                img = paddle.to_tensor(img)
                target = paddle.to_tensor(target)

                pred = model(img)
                pred_nhwc = pred.transpose([0, 2, 1, 3]).transpose([0, 1, 3, 2])

                # predict image
                out = F.grid_sample(img, pred_nhwc)
                out_gt = F.grid_sample(img, target)

                # calculate ssim
                ssim_val = ssim(out, out_gt, data_range=1.0)
                ms_ssim_val = ms_ssim(out, out_gt, data_range=1.0)
                # calculate fitness
                fitness += ms_ssim_val

                loss = l1_loss_fn(pred_nhwc, target)
                mse_loss = mse_loss_fn(pred_nhwc, target)

                if i % 10 == 0:
                    logger.info(
                        f"[VAL MODE] Epoch: {epoch}, VAL Iter: {i}, "
                        f"L1 Loss: {float(loss)} MSE Loss: {float(mse_loss)}, "
                        f"MS-SSIM: {float(ms_ssim_val)}, SSIM: {float(ssim_val)}"
                    )

            fitness /= len(val_loader)

        # Save
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_fitness": best_fitness,
            "epoch": epoch,
        }

        paddle.save(ckpt, str(last))

        if best_fitness < fitness:
            best_fitness = fitness
            paddle.save(ckpt, str(best))


def main(args):
    print_args(vars(args))

    args.save_dir = str(
        increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
    )

    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--data-root",
        nargs="?",
        type=str,
        default="~/datasets/doc3d",
        help="The root path of the dataset",
    )
    parser.add_argument(
        "--img-size",
        nargs="?",
        type=int,
        default=288,
        help="The size of the input image",
    )
    parser.add_argument(
        "--epochs",
        nargs="?",
        type=int,
        default=65,
        help="The number of training epochs",
    )
    parser.add_argument(
        "--batch-size", nargs="?", type=int, default=12, help="Batch Size"
    )
    parser.add_argument(
        "--lr", nargs="?", type=float, default=1e-04, help="Learning Rate"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        default=None,
        help="Path to previous saved model to restart from",
    )
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument(
        "--project", default=ROOT / "runs/train", help="save to project/name"
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    args = parser.parse_args()
    main(args)
