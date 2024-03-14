from tf_keras.src.testing_infra import test_combinations, test_utils
from tfclip.mapool import MultiheadAttentionPooling


@test_combinations.run_all_keras_modes
class TestMultiheadAttentionPooling(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            MultiheadAttentionPooling,
            kwargs={'heads': 8},
            input_shape=[2, 50, 768],
            input_dtype='float32',
            expected_output_shape=[None, 1, 768],
            expected_output_dtype='float32'
        )
