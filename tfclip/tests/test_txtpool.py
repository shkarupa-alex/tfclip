import numpy as np
from keras.src import backend
from keras.src import testing

from tfclip.txtpool import TextGlobalPool


class TestTextGlobalPool(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            TextGlobalPool,
            init_kwargs={"mode": "first"},
            input_shape=((3, 4, 2), (3, 4)),
            input_dtype=("float32", "int64"),
            expected_output_shape=((3, 2), (3, 3, 2)),
            expected_output_dtype=("float32", "float32"),
        )
        self.run_layer_test(
            TextGlobalPool,
            init_kwargs={"mode": "last"},
            input_shape=((3, 4, 2), (3, 4)),
            input_dtype=("float32", "int64"),
            expected_output_shape=((3, 2), (3, 3, 2)),
            expected_output_dtype=("float32", "float32"),
        )
        self.run_layer_test(
            TextGlobalPool,
            init_kwargs={"mode": "argmax"},
            input_shape=((3, 4, 2), (3, 4)),
            input_dtype=("float32", "int64"),
            expected_output_shape=((3, 2), (3, 4, 2)),
            expected_output_dtype=("float32", "float32"),
        )
        self.run_layer_test(
            TextGlobalPool,
            init_kwargs={"mode": "none"},
            input_shape=((3, 4, 2), (3, 4)),
            input_dtype=("float32", "int64"),
            expected_output_shape=((3, 4, 2), (3, 4, 2)),
            expected_output_dtype=("float32", "float32"),
        )

    def test_value(self):
        inputs = np.arange(3 * 4 * 2).reshape([3, 4, 2]).astype("float32")
        texts = np.any(inputs < 20, axis=-1).astype("int64")

        result = TextGlobalPool(mode="first")([inputs, texts])
        self.assertTrue(
            (backend.convert_to_numpy(result[0]) == inputs[:, 0]).all()
        )

        result = TextGlobalPool(mode="last")([inputs, texts])
        self.assertTrue(
            (backend.convert_to_numpy(result[0]) == inputs[:, -1]).all()
        )

        result = TextGlobalPool(mode="argmax")([inputs, texts])
        self.assertTrue(
            (backend.convert_to_numpy(result[0][:2]) == inputs[:2, -1]).all()
        )
        self.assertTrue(
            (backend.convert_to_numpy(result[0][2]) == inputs[2, 1]).all()
        )

        result = TextGlobalPool(mode="none")([inputs, texts])
        self.assertTrue((backend.convert_to_numpy(result[0]) == inputs).all())
