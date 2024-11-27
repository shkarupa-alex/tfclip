from keras.src import testing

from tfclip.evattn import EvaMultiHeadAttention


class TestEvaMultiHeadAttention(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            EvaMultiHeadAttention,
            init_kwargs={
                "num_heads": 12,
                "key_dim": 64,
                "value_dim": 48,
                "norm_epsilon": 1e-6,
                "rpe_pretrain": 16,
            },
            input_shape={
                "query_shape": (2, 64, 768),
                "value_shape": (2, 64, 768),
            },
            input_dtype={"query_shape": "float32", "value_shape": "float32"},
            expected_output_shape=(2, 64, 768),
            expected_output_dtype="float32",
            run_training_check=False,
        )
