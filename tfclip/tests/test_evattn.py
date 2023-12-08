import tensorflow as tf
from keras.src.testing_infra import test_combinations
from tfclip.evattn import EvaMultiHeadAttention


@test_combinations.run_all_keras_modes
class TestEvaMultiHeadAttention(test_combinations.TestCase):
    def test_layer(self):
        inputs = tf.random.uniform(shape=(2, 197, 768))
        layer = EvaMultiHeadAttention(12, 64, norm_epsilon=1e-6, rpe_pretrain=16)

        result = layer(inputs, inputs)
        result = self.evaluate(result)

        self.assertTupleEqual(result.shape[1:], (197, 768))


if __name__ == '__main__':
    tf.test.main()
