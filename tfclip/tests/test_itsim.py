import tensorflow as tf
from tf_keras.src.testing_infra import test_combinations
from tfclip.itsim import ImageTextSimilarity
from testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestImageTextSimilarity(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            ImageTextSimilarity,
            kwargs={'scale_init': 0.1, 'bias_init': None},
            input_shapes=[(2, 8)] * 2,
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, None)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            ImageTextSimilarity,
            kwargs={'scale_init': 0.1, 'bias_init': 0.1},
            input_shapes=[(2, 8)] * 2,
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, None)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
