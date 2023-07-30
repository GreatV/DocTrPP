import argparse
import glob
import os
import pathlib
import random

random.seed(20230730)


def run(args):
    data_root = args.data_root
    data_path = os.path.join(data_root, "img", "*", "*.png")
    img_list = glob.glob(data_path, recursive=True)
    sorted(img_list)
    random.shuffle(img_list)
    ratio = args.train_ratio
    train_size = int(len(img_list) * ratio)
    with open(os.path.join(data_root, "train.txt"), "w") as file:
        for item in img_list[:train_size]:
            parts = pathlib.Path(item).parts
            item = os.path.join(parts[-2], parts[-1])
            file.write("%s\n" % item.split(".png")[0])
    with open(os.path.join(data_root, "val.txt"), "w") as file:
        for item in img_list[train_size:]:
            parts = pathlib.Path(item).parts
            item = os.path.join(parts[-2], parts[-1])
            file.write("%s\n" % item.split(".png")[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="", help="Data path to load data"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of training data"
    )

    args = parser.parse_args()
    run(args)
