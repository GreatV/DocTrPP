import unittest

import numpy as np
import paddle
import torch
from GeoTr_base import GeoTrBase
from padiff import auto_diff, create_model

from GeoTr import GeoTr


def test_GeoTr():
    module = create_model(GeoTrBase())
    module.auto_layer_map("base")
    layer = create_model(GeoTr())
    layer.auto_layer_map("raw")

    x = np.random.randn(1, 3, 288, 288).astype("float32")
    inp = ({"image1": torch.as_tensor(x)}, {"image1": paddle.to_tensor(x)})
    assert (
        auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
    ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
