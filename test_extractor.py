import unittest

import numpy as np
import paddle
import torch
from padiff import auto_diff, create_model

from extractor import BasicEncoder, ResidualBlock
from extractor_ref import BasicEncoderRef, ResidualBlockRef


class TestResidualBlock(unittest.TestCase):
    def setUp(self):
        self.in_channels = 320
        self.out_channels = 320
        self.norm_fn = "group"

        self.input = np.random.randn(4, 320, 320, 3).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(
            ResidualBlockRef(self.in_channels, self.out_channels, self.norm_fn)
        )
        module.auto_layer_map("base")

        layer = create_model(
            ResidualBlock(self.in_channels, self.out_channels, self.norm_fn)
        )
        layer.auto_layer_map("raw")

        inp = ({"x": torch.as_tensor(self.input)}, {"x": paddle.to_tensor(self.input)})
        assert (
            auto_diff(module, layer, inp, auto_weights=True) is True
        ), "Failed. expected success."


class TestResidualBlock2(TestResidualBlock):
    def init_params(self):
        self.norm_fn = "batch"


class TestResidualBlock3(TestResidualBlock):
    def init_params(self):
        self.norm_fn = "instance"


class TestResidualBlock4(TestResidualBlock):
    def init_params(self):
        self.norm_fn = "none"


class TestBasicEncoder(unittest.TestCase):
    def setUp(self):
        self.output_dim = 128
        self.norm_fn = "batch"

        self.input = np.random.randn(4, 3, 320, 320).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(BasicEncoderRef(self.output_dim, self.norm_fn))
        module.auto_layer_map("base")

        layer = create_model(BasicEncoder(self.output_dim, self.norm_fn))
        layer.auto_layer_map("raw")

        inp = ({"x": torch.as_tensor(self.input)}, {"x": paddle.to_tensor(self.input)})
        assert (
            auto_diff(module, layer, inp, auto_weights=True) is True
        ), "Failed. expected success."


class TestBasicEncoder2(TestBasicEncoder):
    def init_params(self):
        self.norm_fn = "group"


class TestBasicEncoder3(TestBasicEncoder):
    def init_params(self):
        self.norm_fn = "instance"


class TestBasicEncoder4(TestBasicEncoder):
    def init_params(self):
        self.norm_fn = "none"


if __name__ == "__main__":
    unittest.main()
