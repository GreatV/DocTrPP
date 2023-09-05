import unittest

import numpy as np
import paddle
import torch
from padiff import auto_diff, create_model
from position_encoding_base import (NestedTensorBase,
                                    PositionEmbeddingLearnedBase)

from position_encoding import NestedTensor, PositionEmbeddingLearned

# def test_PositionEmbeddingSine():

#     module = create_model(PositionEmbeddingSineBase())
#     module.auto_layer_map("base")
#     layer = create_model(PositionEmbeddingSine())
#     layer.auto_layer_map("raw")

#     input = np.random.randn(64, 320, 320).astype("float32")
#     inp = ({"mask": torch.as_tensor(input)}, {"mask": paddle.to_tensor(input)})
#     assert (
#         auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
#     ), "Failed. expected success."


def test_PositionEmbeddingLearned():
    module = create_model(PositionEmbeddingLearnedBase())
    module.auto_layer_map("base")
    layer = create_model(PositionEmbeddingLearned())
    layer.auto_layer_map("raw")

    input = np.random.randn(50, 50).astype("float32")
    mask = np.random.randint(0, 50, (2)).astype("float32")

    inp = (
        {
            "tensor_list": NestedTensorBase(
                torch.as_tensor(input), torch.as_tensor(mask)
            )
        },
        {"tensor_list": NestedTensor(paddle.to_tensor(input), paddle.to_tensor(mask))},
    )
    assert (
        auto_diff(module, layer, inp, auto_weights=True, atol=1e-4) is True
    ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
