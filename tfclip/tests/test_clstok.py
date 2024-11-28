import numpy as np
from keras.src import backend
from keras.src import layers
from keras.src import testing

from tfclip.clstok import AddClassToken
from tfclip.clstok import SplitClassToken


class TestAddClassToken(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            AddClassToken,
            init_kwargs={"first": True},
            input_shape=(2, 12, 4),
            input_dtype="float32",
            expected_output_shape=(2, 13, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            AddClassToken,
            init_kwargs={"first": False},
            input_shape=(2, 12, 4),
            input_dtype="float32",
            expected_output_shape=(2, 13, 4),
            expected_output_dtype="float32",
        )

    def test_mask(self):
        inputs = np.array([[2, 6, 1, 0], [2, 0, 0, 0]]).astype("int32")
        embeddings = layers.Embedding(8, 2, mask_zero=True)(inputs)
        mask = backend.convert_to_numpy(embeddings._keras_mask)
        self.assertTrue((mask == inputs.astype("bool")).all())

        embeddings = AddClassToken()(embeddings)
        expected = np.array([[1, 2, 6, 1, 0], [1, 2, 0, 0, 0]]).astype("bool")
        mask = backend.convert_to_numpy(embeddings._keras_mask)
        self.assertTrue((mask == expected).all())


class TestSplitClassToken(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            SplitClassToken,
            init_kwargs={"first": True},
            input_shape=(2, 7**2 + 1, 8),
            input_dtype="float32",
            expected_output_shape=((2, 8), (2, 7**2, 8)),
            expected_output_dtype=("float32", "float32"),
        )
        self.run_layer_test(
            SplitClassToken,
            init_kwargs={"first": False},
            input_shape=(2, 7**2 + 1, 8),
            input_dtype="float32",
            expected_output_shape=((2, 8), (2, 7**2, 8)),
            expected_output_dtype=("float32", "float32"),
        )
