from tf_keras.src.testing_infra import test_combinations, test_utils
from tfclip.attnpool import AttentionalPooler


@test_combinations.run_all_keras_modes
class TestAttentionalPooler(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            AttentionalPooler,
            kwargs={'units': 512, 'heads': 8, 'queries': 256},
            input_shape=[2, 50, 768],
            input_dtype='float32',
            expected_output_shape=[None, 256, 512],
            expected_output_dtype='float32'
        )
