from keras.src import testing

from tfclip.itsim import ImageTextSimilarity


class TestImageTextSimilarity(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            ImageTextSimilarity,
            init_kwargs={"scale_init": 0.1, "bias_init": None},
            input_shape=((2, 8), (2, 8)),
            input_dtype=("float32", "float32"),
            expected_output_shape=(2, 2),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ImageTextSimilarity,
            init_kwargs={"scale_init": 0.1, "bias_init": 0.1},
            input_shape=((2, 8), (2, 8)),
            input_dtype=("float32", "float32"),
            expected_output_shape=(2, 2),
            expected_output_dtype="float32",
        )
