from keras.src import testing

from tfclip.attnpool import AttentionalPooler


class TestAttentionalPooler(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            AttentionalPooler,
            init_kwargs={"units": 512, "heads": 8, "queries": 256},
            input_shape=(2, 50, 768),
            input_dtype="float32",
            expected_output_shape=(2, 256, 512),
            expected_output_dtype="float32",
        )
