import unittest

import numpy as np
import paddle
import torch
from padiff import auto_diff, create_model

from GeoTr import TransDecoder, TransEncoder
from GeoTr_ref import TransDecoderRef, TransEncoderRef


class TestTransEncoder(unittest.TestCase):
    def setUp(self):
        self.num_attn_layers = 2
        self.hidden_dim = 128

        self.input = np.random.randn(3, 128, 36, 36).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(TransEncoderRef(self.num_attn_layers, self.hidden_dim))
        module.auto_layer_map("base")

        layer = create_model(TransEncoder(self.num_attn_layers, self.hidden_dim))
        layer.auto_layer_map("raw")

        inp = (
            {"imgf": torch.as_tensor(self.input)},
            {"image": paddle.to_tensor(self.input)},
        )

        assert (
            auto_diff(module, layer, inp, auto_weights=True) is True
        ), "Failed. expected success."


class TestTransDecoder(unittest.TestCase):
    def setUp(self):
        self.num_attn_layers = 2
        self.hidden_dim = 128

        self.input = np.random.randn(4, 128, 36, 36).astype("float32")
        self.query_embed = np.random.randn(1296, 4, 128).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(TransDecoderRef(self.num_attn_layers, self.hidden_dim))
        module.auto_layer_map("base")

        layer = create_model(TransDecoder(self.num_attn_layers, self.hidden_dim))
        layer.auto_layer_map("raw")

        inp = (
            {
                "imgf": torch.as_tensor(self.input),
                "query_embed": torch.as_tensor(self.query_embed),
            },
            {
                "image": paddle.to_tensor(self.input),
                "query_embed": paddle.to_tensor(self.query_embed),
            },
        )

        assert (
            auto_diff(module, layer, inp, auto_weights=True) is True
        ), "Failed. expected success."
