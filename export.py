import argparse
import os

import paddle

from GeoTr import GeoTr


def export(args):
    model_path = args.model
    imgsz = args.imgsz
    format = args.format

    model = GeoTr()
    checkpoint = paddle.load(model_path)
    model.set_state_dict(checkpoint["model"])
    model.eval()

    dirname = os.path.dirname(model_path)
    if format == "static" or format == "onnx":
        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(shape=[1, 3, imgsz, imgsz], dtype="float32")
            ],
            full_graph=True,
        )
        paddle.jit.save(model, os.path.join(dirname, "model"))

    if format == "onnx":
        onnx_path = os.path.join(dirname, "model.onnx")
        os.system(
            f"paddle2onnx --model_dir {dirname}"
            " --model_filename model.pdmodel"
            " --params_filename model.pdiparams"
            f" --save_file {onnx_path}"
            " --opset_version 11"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export model")

    parser.add_argument(
        "--model",
        "-m",
        nargs="?",
        type=str,
        default="",
        help="The path of model",
    )

    parser.add_argument(
        "--imgsz", type=int, default=288, help="The size of input image"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="static",
        help="The format of exported model, which can be static or onnx",
    )

    args = parser.parse_args()

    export(args)
