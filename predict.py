import argparse

import cv2
import paddle
import paddle.nn.functional as F

from GeoTr import GeoTr


def run(args):
    image_path = args.image
    model_path = args.model
    output_path = args.output

    checkpoint = paddle.load(model_path)
    state_dict = checkpoint["model"]
    model = GeoTr()
    model.set_state_dict(state_dict)
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.resize(img, (288, 288))
    img = img[:, :, ::-1]
    img = img.astype("float32") / 255.0
    img = img.transpose(2, 0, 1)

    x = paddle.to_tensor(img)
    x = paddle.unsqueeze(x, axis=0)
    bm = model(x)
    bm = bm.transpose([0, 2, 1, 3]).transpose([0, 1, 3, 2])
    out = F.grid_sample(x, bm)

    out = out[0].numpy().transpose((1, 2, 0))
    out = out[:, :, ::-1]
    out = out * 255.0
    out = out.astype("uint8")
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
