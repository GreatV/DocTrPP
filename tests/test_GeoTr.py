import random
import unittest

import numpy as np
import paddle
import torch
from GeoTr_base import (FlowHeadBase, GeoTrBase, OverlapPatchEmbedBase,
                        TransDecoderBase, TransEncoderBase, UpdateBlockBase)
from padiff import auto_diff, create_model

from GeoTr import (FlowHead, GeoTr, OverlapPatchEmbed, TransDecoder,
                   TransEncoder, UpdateBlock)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
paddle.seed(0)


def test_TransEncoder():
    module = create_model(TransEncoderBase(1))
    module.auto_layer_map("base")
    layer = create_model(TransEncoder(1))
    layer.auto_layer_map("raw")

    x = np.random.randn(3, 128, 36, 36).astype("float32")
    inp = ({"imgf": torch.as_tensor(x)}, {"imgf": paddle.to_tensor(x)})

    assert (
        auto_diff(module, layer, inp, auto_weights=True, auto_init=True, atol=1e-4)
        is True
    ), "Failed. expected success."


def test_TransDecoder():
    module = create_model(TransDecoderBase(1))
    module.auto_layer_map("base")
    layer = create_model(TransDecoder(1))
    layer.auto_layer_map("raw")

    x = np.random.randn(3, 128, 36, 36).astype("float32")
    y = np.random.randn(1296, 3, 128).astype("float32")
    inp = (
        {"imgf": torch.as_tensor(x), "query_embed": torch.as_tensor(y)},
        {"imgf": paddle.to_tensor(x), "query_embed": paddle.to_tensor(y)},
    )

    assert (
        auto_diff(module, layer, inp, auto_weights=True, auto_init=True, atol=1e-4)
        is True
    ), "Failed. expected success."


def test_FlowHead():
    module = create_model(FlowHeadBase())
    module.auto_layer_map("base")
    layer = create_model(FlowHead())
    layer.auto_layer_map("raw")

    x = np.random.randn(3, 128, 36, 36).astype("float32")
    inp = ({"x": torch.as_tensor(x)}, {"x": paddle.to_tensor(x)})

    assert (
        auto_diff(module, layer, inp, auto_weights=True, auto_init=True, atol=1e-4)
        is True
    ), "Failed. expected success."


def test_UpdateBlock():
    module = create_model(UpdateBlockBase())
    module.auto_layer_map("base")
    layer = create_model(UpdateBlock())
    layer.auto_layer_map("raw")

    x = np.random.randn(3, 128, 36, 36).astype("float32")
    y = np.random.randn(3, 2, 36, 36).astype("float32")

    inp = (
        {
            "imgf": torch.as_tensor(x),
            "coords1": torch.as_tensor(y),
        },
        {
            "imgf": paddle.to_tensor(x),
            "coords1": paddle.to_tensor(y),
        },
    )

    assert (
        auto_diff(module, layer, inp, auto_weights=True, auto_init=True, atol=1e-4)
        is True
    ), "Failed. expected success."


def test_OverlapPatchEmbed():
    module = create_model(OverlapPatchEmbedBase())
    module.auto_layer_map("base")
    layer = create_model(OverlapPatchEmbed())
    layer.auto_layer_map("raw")

    x = np.random.randn(3, 3, 224, 224).astype("float32")
    inp = ({"x": torch.as_tensor(x)}, {"x": paddle.to_tensor(x)})

    assert (
        auto_diff(module, layer, inp, auto_weights=True, auto_init=True, atol=1e-4)
        is True
    ), "Failed. expected success."


def test_GeoTr():
    module = create_model(GeoTrBase())
    # module.eval()
    module.auto_layer_map("base")
    layer = create_model(GeoTr())
    # layer.eval()
    layer.auto_layer_map("raw")

    x = np.random.randn(3, 3, 288, 288).astype("float32")
    inp = ({"image1": torch.as_tensor(x)}, {"image1": paddle.to_tensor(x)})

    assert (
        auto_diff(module, layer, inp, auto_weights=True, auto_init=True, atol=0.00001)
        is True
    ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
