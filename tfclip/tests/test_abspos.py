import tensorflow as tf
from keras.src.testing_infra import test_combinations, test_utils
from tfclip.abspos import ImagePositionEmbedding, TextPositionEmbedding


@test_combinations.run_all_keras_modes
class TestImagePositionEmbedding(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            ImagePositionEmbedding,
            kwargs={'patch_size': 16, 'pretrain_size': 224, 'cls_tok': True},
            input_shape=[2, 14 ** 2 + 1, 8],
            input_dtype='float32',
            expected_output_shape=[None, 14 ** 2 + 1, 8],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            ImagePositionEmbedding,
            kwargs={'patch_size': 16, 'pretrain_size': 224, 'cls_tok': False},
            input_shape=[2, 14 ** 2, 8],
            input_dtype='float32',
            expected_output_shape=[None, 14 ** 2, 8],
            expected_output_dtype='float32'
        )

    def test_resize(self):
        layer224 = ImagePositionEmbedding(32, 224)
        layer224.build([None, 7 ** 2 + 1, 16])

        layer384 = ImagePositionEmbedding(32, 384)
        layer384.build([None, 12 ** 2 + 1, 16])
        layer384.set_weights(layer224.get_weights())


@test_combinations.run_all_keras_modes
class TestTextPositionEmbedding(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            TextPositionEmbedding,
            kwargs={'context': 16},
            input_shape=[2, 14, 8],
            input_dtype='float32',
            expected_output_shape=[None, 14, 8],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            TextPositionEmbedding,
            kwargs={'context': 16},
            input_shape=[2, 32, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 8],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
