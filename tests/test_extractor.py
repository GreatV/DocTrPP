import unittest

import numpy as np
import paddle
import torch
from extractor_base import BasicEncoderBase, ResidualBlockBase
from padiff import auto_diff, create_model

from extractor import BasicEncoder, ResidualBlock


def test_residual_block():
    module = create_model(ResidualBlockBase(320, 320))
    module.auto_layer_map("base")
    layer = create_model(ResidualBlock(320, 320))
    layer.auto_layer_map("raw")

    input = np.random.randn(1, 320, 320, 3).astype("float32")
    inp = ({"x": torch.as_tensor(input)}, {"x": paddle.to_tensor(input)})
    assert (
        auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
    ), "Failed. expected success."


def test_basic_encoder():
    module = create_model(BasicEncoderBase())
    module.auto_layer_map("base")
    layer = create_model(BasicEncoder())
    layer.auto_layer_map("raw")

    input = np.random.randn(1, 3, 320, 320).astype("float32")
    inp = ({"x": torch.as_tensor(input)}, {"x": paddle.to_tensor(input)})
    assert (
        auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
    ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
