import argparse
import glob
import os
import random
from pathlib import Path

from loguru import logger

random.seed(1234567)


def run(args):
    data_root = os.path.expanduser(args.data_root)
    ratio = args.train_ratio

    data_path = os.path.join(data_root, "img", "*", "*.png")
    img_list = glob.glob(data_path, recursive=True)
    sorted(img_list)
    random.shuffle(img_list)

    train_size = int(len(img_list) * ratio)

    train_text_path = os.path.join(data_root, "train.txt")
    with open(train_text_path, "w") as file:
        for item in img_list[:train_size]:
            parts = Path(item).parts
            item = os.path.join(parts[-2], parts[-1])
            file.write("%s\n" % item.split(".png")[0])

    val_text_path = os.path.join(data_root, "val.txt")
    with open(val_text_path, "w") as file:
        for item in img_list[train_size:]:
            parts = Path(item).parts
            item = os.path.join(parts[-2], parts[-1])
            file.write("%s\n" % item.split(".png")[0])

    logger.info(f"TRAIN LABEL: {train_text_path}")
    logger.info(f"VAL LABEL: {val_text_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="~/datasets/doc3d",
        help="Data path to load data",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of training data"
    )

    args = parser.parse_args()

    logger.info(args)

    run(args)
