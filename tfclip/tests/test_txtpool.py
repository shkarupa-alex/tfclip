import numpy as np
import tensorflow as tf
from keras.saving import custom_object_scope
from keras.src.testing_infra import test_combinations
from tfclip.txtpool import TextGlobalPool
from testing_utils import layer_multi_io_test


class TextGlobalPoolMasked(TextGlobalPool):
    def call(self, inputs, *args, mask=None, **kwargs):
        mask = tf.reduce_any(inputs < 20, axis=-1)

        return super().call(inputs, mask=mask)


@test_combinations.run_all_keras_modes
class TestTextGlobalPool(test_combinations.TestCase):
    def test_layer(self):
        inputs = np.arange(3 * 4 * 2).reshape([3, 4, 2]).astype('float32')

        result = layer_multi_io_test(
            TextGlobalPool,
            kwargs={'mode': 'first'},
            input_datas=[inputs],
            expected_output_shapes=[(None, 2), (None, 3, 2)],
            expected_output_dtypes=['float32'] * 2
        )
        self.assertAllEqual(result[0], inputs[:, 0])

        result = layer_multi_io_test(
            TextGlobalPool,
            kwargs={'mode': 'last'},
            input_datas=[inputs],
            expected_output_shapes=[(None, 2), (None, 3, 2)],
            expected_output_dtypes=['float32'] * 2
        )
        self.assertAllEqual(result[0], inputs[:, -1])

        with custom_object_scope({'TextGlobalPoolMasked': TextGlobalPoolMasked}):
            result = layer_multi_io_test(
                TextGlobalPoolMasked,
                kwargs={'mode': 'argmax'},
                input_datas=[inputs],
                expected_output_shapes=[(None, 2), (None, 4, 2)],
                expected_output_dtypes=['float32'] * 2
            )
            self.assertAllEqual(result[0][:2], inputs[:2, -1])
            self.assertAllEqual(result[0][2], inputs[2, 1])

        result = layer_multi_io_test(
            TextGlobalPool,
            kwargs={'mode': 'none'},
            input_datas=[inputs],
            expected_output_shapes=[(None, 4, 2)] * 2,
            expected_output_dtypes=['float32'] * 2
        )
        self.assertAllEqual(result[0], inputs)


if __name__ == '__main__':
    tf.test.main()
