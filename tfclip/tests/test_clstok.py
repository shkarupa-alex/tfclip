import numpy as np
import tensorflow as tf
from tf_keras import layers
from tf_keras.src.testing_infra import test_combinations, test_utils
from tfclip.clstok import AddClassToken, SplitClassToken
from testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestAddClassToken(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            AddClassToken,
            kwargs={'first': True},
            input_shape=[2, 12, 4],
            input_dtype='float32',
            expected_output_shape=[None, 13, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            AddClassToken,
            kwargs={'first': False},
            input_shape=[2, 12, 4],
            input_dtype='float32',
            expected_output_shape=[None, 13, 4],
            expected_output_dtype='float32'
        )

    def test_mask(self):
        inputs = np.array([[2, 6, 1, 0], [2, 0, 0, 0]]).astype('int32')
        embeddings = layers.Embedding(8, 2, mask_zero=True)(inputs)
        result = self.evaluate(embeddings._keras_mask)
        self.assertAllEqual(result, inputs.astype('bool'))

        embeddings = AddClassToken()(embeddings)
        expected = np.array([[1, 2, 6, 1, 0], [1, 2, 0, 0, 0]]).astype('bool')
        result = self.evaluate(embeddings._keras_mask)
        self.assertAllEqual(result, expected)


@test_combinations.run_all_keras_modes
class TestSplitClassToken(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            SplitClassToken,
            kwargs={'first': True},
            input_shapes=[(2, 7 ** 2 + 1, 8)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, 8), (None, 7 ** 2, 8)],
            expected_output_dtypes=['float32'] * 2
        )
        layer_multi_io_test(
            SplitClassToken,
            kwargs={'first': False},
            input_shapes=[(2, 7 ** 2 + 1, 8)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, 8), (None, 7 ** 2, 8)],
            expected_output_dtypes=['float32'] * 2
        )


if __name__ == '__main__':
    tf.test.main()
