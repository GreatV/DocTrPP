import unittest

import numpy as np
import paddle
import torch
from padiff import auto_diff, create_model

# isort: off
from position_encoding import (
    NestedTensor,
    PositionEmbeddingLearned,
    PositionEmbeddingSine,
)
from position_encoding_ref import (
    NestedTensorRef,
    PositionEmbeddingLearnedRef,
    PositionEmbeddingSineRef,
)

# isort: on


class TestPositionEmbeddingSine(unittest.TestCase):
    def setUp(self):
        self.num_pos_feats = 64
        self.temperature = 10000
        self.normalize = False
        self.scale = None

        self.input = np.random.randn(320, 320, 64).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(
            PositionEmbeddingSineRef(
                self.num_pos_feats, self.temperature, self.normalize, self.scale
            )
        )
        module.auto_layer_map("base")

        layer = create_model(
            PositionEmbeddingSine(
                self.num_pos_feats, self.temperature, self.normalize, self.scale
            )
        )
        layer.auto_layer_map("raw")

        inp = (
            {"mask": torch.as_tensor(self.input)},
            {"mask": paddle.to_tensor(self.input)},
        )
        assert (
            auto_diff(module, layer, inp, auto_weights=True, diff_phase="forward")
            is True
        ), "Failed. expected success."


class TestPositionEmbeddingSineCase1(TestPositionEmbeddingSine):
    def init_params(self):
        self.normalize = True


class TestPositionEmbeddingSineCase2(TestPositionEmbeddingSine):
    def init_params(self):
        self.normalize = True
        self.scale = 0.5


class TestPositionEmbeddingLearned(unittest.TestCase):
    def setUp(self):
        self.num_pos_feats = 256
        self.input = np.random.randn(3, 50, 50).astype("float32")
        self.mask = np.random.randint(0, 255, (4, 3, 256, 256)).astype("float32")

        self.init_params()

    def init_params(self):
        pass

    def test_auto_diff(self):
        module = create_model(PositionEmbeddingLearnedRef(self.num_pos_feats))
        module.auto_layer_map("base")

        layer = create_model(PositionEmbeddingLearned(self.num_pos_feats))
        layer.auto_layer_map("raw")

        inp = (
            {
                "tensor_list": NestedTensorRef(
                    torch.as_tensor(self.input), torch.as_tensor(self.mask)
                )
            },
            {
                "tensor_list": NestedTensor(
                    paddle.to_tensor(self.input), paddle.to_tensor(self.mask)
                )
            },
        )
        assert (
            auto_diff(module, layer, inp, auto_weights=True) is True
        ), "Failed. expected success."


if __name__ == "__main__":
    unittest.main()
