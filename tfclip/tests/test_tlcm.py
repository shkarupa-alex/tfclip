import numpy as np
import tensorflow as tf
from tf_keras.src.testing_infra import test_combinations
from tfclip.tlcm import TokenLastCausalMask


@test_combinations.run_all_keras_modes
class TestTokenLastCausalMask(test_combinations.TestCase):
    def test_layer(self):
        inputs = np.array([[1, 1, 0, 0]], dtype='int64')
        outputs = TokenLastCausalMask()(inputs)
        outputs = self.evaluate(outputs)
        self.assertAllEqual(outputs, np.array(
            [[[True, False, False, False, False],
              [True, True, False, False, False],
              [True, True, True, False, False],
              [True, True, True, True, False],
              [True, True, True, False, False]]]))


if __name__ == '__main__':
    tf.test.main()
