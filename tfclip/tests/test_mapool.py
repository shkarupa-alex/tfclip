from keras.src import testing

from tfclip.mapool import MultiheadAttentionPooling


class TestMultiheadAttentionPooling(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            MultiheadAttentionPooling,
            init_kwargs={"heads": 8},
            input_shape=(2, 50, 768),
            input_dtype="float32",
            expected_output_shape=(2, 1, 768),
            expected_output_dtype="float32",
        )
