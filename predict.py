import argparse

import cv2
import paddle

from GeoTr import GeoTr
from utils import to_image, to_tensor, unwarp


def run(args):
    image_path = args.image
    model_path = args.model
    output_path = args.output

    checkpoint = paddle.load(model_path)
    state_dict = checkpoint["model"]
    model = GeoTr()
    model.set_state_dict(state_dict)
    model.eval()

    img_org = cv2.imread(image_path)
    y = to_tensor(img_org)
    img = cv2.resize(img_org, (288, 288))
    x = to_tensor(img)

    bm = model(x)
    out = unwarp(y, bm)

    out = to_image(out)
    cv2.imwrite(output_path, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict")

    parser.add_argument(
        "--image",
        "-i",
        nargs="?",
        type=str,
        default="",
        help="The path of image",
    )

    parser.add_argument(
        "--model",
        "-m",
        nargs="?",
        type=str,
        default="",
        help="The path of model",
    )

    parser.add_argument(
        "--output",
        "-o",
        nargs="?",
        type=str,
        default="",
        help="The path of output",
    )

    args = parser.parse_args()

    print(args)

    run(args)
