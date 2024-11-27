from keras.src import testing

from tfclip.abspos import ImagePositionEmbedding
from tfclip.abspos import TextPositionEmbedding


class TestImagePositionEmbedding(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            ImagePositionEmbedding,
            init_kwargs={
                "patch_size": 16,
                "pretrain_size": 224,
                "cls_tok": True,
            },
            input_shape=(2, 14**2 + 1, 8),
            input_dtype="float32",
            expected_output_shape=(2, 14**2 + 1, 8),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ImagePositionEmbedding,
            init_kwargs={
                "patch_size": 16,
                "pretrain_size": 224,
                "cls_tok": False,
            },
            input_shape=(2, 14**2, 8),
            input_dtype="float32",
            expected_output_shape=(2, 14**2, 8),
            expected_output_dtype="float32",
        )

    def test_resize(self):
        layer224 = ImagePositionEmbedding(32, 224)
        layer224.build([None, 7**2 + 1, 16])

        layer384 = ImagePositionEmbedding(32, 384)
        layer384.build([None, 12**2 + 1, 16])
        layer384.set_weights(layer224.get_weights())


class TestTextPositionEmbedding(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            TextPositionEmbedding,
            init_kwargs={"context": 16},
            input_shape=(2, 14, 8),
            input_dtype="float32",
            expected_output_shape=(2, 14, 8),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            TextPositionEmbedding,
            init_kwargs={"context": 16},
            input_shape=(2, 32, 8),
            input_dtype="float32",
            expected_output_shape=(2, 16, 8),
            expected_output_dtype="float32",
        )
