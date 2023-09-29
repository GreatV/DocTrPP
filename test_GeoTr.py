import unittest

import numpy as np
import paddle
import torch
from padiff import auto_diff, create_model

# isort: off
from GeoTr import (
    FlowHead,
    OverlapPatchEmbed,
    TransDecoder,
    TransEncoder,
    UpdateBlock,
    coords_grid,
    upflow8,
    GeoTr,
)
from GeoTr_ref import (
    FlowHeadRef,
    OverlapPatchEmbedRef,
    TransDecoderRef,
    TransEncoderRef,
    UpdateBlockRef,
    GeoTrRef,
)

# isort: on

from GeoTr_ref import coords_grid as coords_grid_ref
from GeoTr_ref import upflow8 as upflow8_ref


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


class TestFlowHead(unittest.TestCase):
    def setUp(self):
        self.input_dim = 128
        self.hidden_dim = 128

        self.input = np.random.randn(4, 128, 320, 320).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(FlowHeadRef(self.input_dim, self.hidden_dim))
        module.auto_layer_map("base")

        layer = create_model(FlowHead(self.input_dim, self.hidden_dim))
        layer.auto_layer_map("raw")

        inp = (
            {"x": torch.as_tensor(self.input)},
            {"x": paddle.to_tensor(self.input)},
        )

        assert (
            auto_diff(module, layer, inp, auto_weights=True) is True
        ), "Failed. expected success."


class TestUpdateBlock(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 128

        self.input = np.random.randn(256, 128, 3, 3).astype("float32")
        self.coords = np.random.randn(256, 2, 3, 3).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(UpdateBlockRef(self.hidden_dim))
        module.auto_layer_map("base")

        layer = create_model(UpdateBlock(self.hidden_dim))
        layer.auto_layer_map("raw")

        inp = (
            {
                "imgf": torch.as_tensor(self.input),
                "coords1": torch.as_tensor(self.coords),
            },
            {
                "image": paddle.to_tensor(self.input),
                "coords": paddle.to_tensor(self.coords),
            },
        )

        assert (
            auto_diff(module, layer, inp, auto_weights=True) is True
        ), "Failed. expected success."


class TestCoordsGrid(unittest.TestCase):
    def setUp(self):
        self.batch = 4
        self.ht = 320
        self.wd = 280

        self.init_params()

    def init_params(self):
        pass

    def test_coords_grid(self):
        out_ref = coords_grid_ref(self.batch, self.ht, self.wd)
        out = coords_grid(self.batch, self.ht, self.wd)

        np.testing.assert_allclose(out_ref.numpy(), out.numpy(), rtol=1e-7, atol=1e-7)


class TestUpflow8(unittest.TestCase):
    def setUp(self):
        self.mode = "bilinear"
        self.input = np.random.randn(4, 2, 320, 320).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_upflow8(self):
        out_ref = upflow8_ref(torch.as_tensor(self.input), self.mode)
        out = upflow8(paddle.to_tensor(self.input), self.mode)

        np.testing.assert_allclose(out_ref.numpy(), out.numpy(), rtol=1e-7, atol=1e-7)


class TestOverlapPatchEmbed(unittest.TestCase):
    def setUp(self):
        self.image_size = 320
        self.patch_size = 7
        self.stride = 4
        self.in_channels = 3
        self.embed_dim = 768

        self.input = np.random.randn(4, 3, 320, 320).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(
            OverlapPatchEmbedRef(
                self.image_size,
                self.patch_size,
                self.stride,
                self.in_channels,
                self.embed_dim,
            )
        )
        module.auto_layer_map("base")

        layer = create_model(
            OverlapPatchEmbed(
                self.image_size,
                self.patch_size,
                self.stride,
                self.in_channels,
                self.embed_dim,
            )
        )
        layer.auto_layer_map("raw")

        inp = (
            {"x": torch.as_tensor(self.input)},
            {"x": paddle.to_tensor(self.input)},
        )

        assert (
            auto_diff(module, layer, inp, auto_weights=True) is True
        ), "Failed. expected success."


class TestGeoTr(unittest.TestCase):
    def setUp(self):
        self.input = np.random.randn(4, 3, 288, 288).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(GeoTrRef())
        module.auto_layer_map("base")

        layer = create_model(GeoTr())
        layer.auto_layer_map("raw")

        inp = (
            {
                "image1": torch.as_tensor(self.input),
            },
            {
                "image": paddle.to_tensor(self.input),
            },
        )

        assert (
            auto_diff(module, layer, inp, auto_weights=True, atol=0.01, rtol=0.01)
            is True
        ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
