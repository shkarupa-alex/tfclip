import numpy as np
import tensorflow as tf
from open_clip.tokenizer import SimpleTokenizer as TargetSimpleTokenizer, HFTokenizer
from tfclip.tokenizer import basic_clean, SimpleTokenizer, SentencePieceTokenizer, TensorflowHubTokenizer


class TestSimpleTokenizer(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = SimpleTokenizer()
        self.target_tokenizer = TargetSimpleTokenizer()

    def test_easy(self):
        inputs = [basic_clean(i) for i in INPUTS_EASY]

        expected = self.target_tokenizer(INPUTS_EASY).numpy()
        result = self.evaluate(self.tokenizer(inputs))

        self.assertAllEqual(expected, result)

    def test_simple(self):
        inputs = [basic_clean(i) for i in INPUTS_SIMPLE]

        expected = self.target_tokenizer(INPUTS_SIMPLE).numpy()
        result = self.evaluate(self.tokenizer(inputs))

        self.assertAllEqual(expected, result)

    def test_normal(self):
        inputs = [basic_clean(i) for i in INPUTS_NORMAL]

        expected = self.target_tokenizer(INPUTS_NORMAL).numpy()
        result = self.evaluate(self.tokenizer(inputs))

        self.assertAllEqual(expected, result)

    def test_hard(self):
        inputs = [basic_clean(i) for i in INPUTS_HARD]

        expected = self.target_tokenizer(INPUTS_HARD).numpy()
        result = self.evaluate(self.tokenizer(inputs))

        self.assertAllEqual(expected, result)

    def test_long_sentence(self):
        inputs = ['0', ' '.join(['diagram'] * 200)]
        inputs_ = [basic_clean(i) for i in inputs]

        expected = self.target_tokenizer(inputs).numpy()
        result = self.evaluate(self.tokenizer(inputs_))

        self.assertAllEqual(expected, result)

    def test_long_word(self):
        inputs = ['0', 'ðŁĺįðŁĺįðŁĺįðŁĺįðŁĺįðŁĺįðŁĺįðŁĺį' * 2]
        inputs_ = [basic_clean(i) for i in inputs]

        expected = self.target_tokenizer(inputs).numpy()
        result = self.evaluate(self.tokenizer(inputs_))

        self.assertAllEqual(expected, result)


class TestSentencePieceTokenizer(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = SentencePieceTokenizer(
            'sentencepiece_c4_en_32k__1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec',
            clean='canonicalize')
        self.target_tokenizer = HFTokenizer('timm/ViT-B-16-SigLIP', clean='canonicalize')

    def test_easy(self):
        inputs = [basic_clean(i) for i in INPUTS_EASY]

        expected = self.target_tokenizer(INPUTS_EASY).numpy()
        result = self.evaluate(self.tokenizer(inputs))
        self.assertAllEqual(expected, result)

    def test_simple(self):
        inputs = [basic_clean(i) for i in INPUTS_SIMPLE]

        expected = self.target_tokenizer(INPUTS_SIMPLE).numpy()
        self.assertAllEqual(expected[:, :6], np.array([
            [316, 332, 1, 1, 1, 1], [260, 1, 1, 1, 1, 1], [2932, 1, 1, 1, 1, 1], [262, 2, 402, 399, 1, 1]]))

        # BigVision version with wrong (non-utf-8) lowering
        result = self.evaluate(self.tokenizer(inputs))
        self.assertAllEqual(result[:, :6], np.array([
            [316, 332, 1, 1, 1, 1], [260, 1, 1, 1, 1, 1], [2932, 1, 1, 1, 1, 1], [262, 198, 131, 402, 399, 1]]))

    def test_hard(self):
        inputs = [basic_clean(i) for i in INPUTS_HARD]

        expected = self.target_tokenizer(INPUTS_HARD).numpy()
        self.assertAllEqual(expected, np.array([
            [527, 22140, 2621, 12440, 2671, 6224, 12512, 262, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 2, 727, 7336, 2, 27610, 21005, 9214, 7336, 6512, 7703, 12022, 10189, 20351, 13840, 2277, 12512, 3284,
             2, 17523, 8857, 14100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2123, 411, 298, 298, 514, 8539, 5723, 262, 2, 268, 4101, 288, 296, 264, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 17568, 10020, 29391, 10189, 6512, 28359, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [19229, 2278, 2, 11085, 262, 2, 24852, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [5576, 278, 432, 484, 298, 264, 2259, 447, 484, 2, 264, 262, 9562, 298, 264, 262, 264, 2, 1122, 2, 303, 447,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1798, 12382, 24124, 2, 3140, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 2, 8463, 4041, 262, 3140, 2, 19881, 2, 12382, 2, 7336, 3140, 25970, 2, 2825, 2277, 262, 29264, 17739,
             3140, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [12382, 17739, 15118, 20644, 10833, 7336, 2, 12512, 262, 2277, 2, 3140, 12512, 2, 12573, 8857, 262, 21099,
             262, 7703, 2277, 15118, 10189, 2277, 22033, 12573, 455, 7336, 2, 28359, 11674, 262, 2, 15118, 22033, 21005,
             9214, 17717, 4611, 10833, 2, 8857, 2, 15118, 22033, 21005, 17568, 8857, 288, 296, 411, 2618, 288, 5469,
             262, 510, 2885, 3231, 262, 492, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))

        result = self.evaluate(self.tokenizer(inputs))
        self.assertAllEqual(result, np.array([
            [527, 22140, 2621, 12440, 2671, 6224, 12512, 262, 212, 139, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 197, 174, 727, 211, 164, 211, 148, 27610, 21005, 9214, 7336, 211, 166, 7703, 12022, 10189, 20351,
             13840, 2277, 12512, 3284, 211, 188, 17523, 8857, 14100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2123, 411, 298, 298, 514, 8539, 5723, 229, 159, 173, 367, 4334, 288, 296, 264, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 211, 157, 24168, 11674, 211, 155, 10189, 6512, 28359, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [19229, 2278, 229, 139, 149, 11085, 262, 211, 182, 24852, 212, 142, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [5576, 278, 432, 484, 298, 264, 2259, 447, 484, 199, 154, 264, 262, 9562, 298, 264, 262, 264, 199, 136,
             1122, 200, 174, 303, 447, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1798, 262, 211, 149, 24124, 211, 182, 3140, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 229, 153, 181, 262, 13373, 5919, 854, 262, 3140, 212, 138, 19881, 212, 143, 12382, 212, 142, 7336,
             3140, 25970, 211, 188, 2825, 2277, 262, 29264, 17739, 3140, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [262, 211, 149, 17739, 15118, 20644, 10833, 7336, 212, 146, 12512, 262, 2277, 211, 182, 3140, 12512, 212,
             142, 12573, 8857, 262, 211, 160, 2277, 262, 7703, 2277, 15118, 10189, 2277, 22033, 12573, 455, 211, 164,
             212, 143, 212, 145, 28359, 11674, 261, 262, 211, 167, 15118, 22033, 21005, 9214, 262, 211, 166, 4611,
             10833, 212, 143, 212, 146, 8857, 211, 167, 15118, 22033, 21005, 17568, 8857, 288, 296, 411, 2618, 288,
             5469, 262, 510, 2885, 1]]))

        # BigVision version without canonicalization and wrong (non-utf-8) lowering
        # [[527, 22140, 2621, 12440, 2671, 6224, 12512, 259, 262, 212, 139, 259, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [262, 197, 174, 727, 211, 164, 271, 211, 148, 27610, 21005, 9214, 7336, 304, 211, 166, 7703, 12022, 10189,
        #   20351, 13840, 2277, 12512, 3284, 211, 188, 17523, 8857, 14100, 259, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [2123, 1262, 411, 298, 298, 259, 514, 335, 8539, 5723, 965, 799, 229, 159, 173, 312, 965, 367, 965, 4334,
        #   288, 296, 264, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [262, 211, 157, 24168, 271, 11674, 285, 211, 155, 10189, 6512, 28359, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [19229, 2278, 229, 139, 149, 11085, 262, 211, 182, 24852, 212, 142, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [340, 3776, 278, 432, 484, 298, 264, 2259, 447, 484, 199, 154, 264, 262, 9562, 298, 264, 262, 264, 199, 136,
        #   1122, 200, 174, 303, 447, 381, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [1798, 262, 211, 149, 24124, 211, 182, 3140, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [262, 229, 153, 181, 262, 13373, 5919, 854, 297, 3140, 212, 138, 19881, 212, 143, 12382, 212, 142, 7336,
        #   3140, 25970, 211, 188, 312, 297, 3576, 271, 2277, 262, 29264, 17739, 3140, 312, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [262, 211, 149, 17739, 15118, 20644, 10833, 7336, 212, 146, 12512, 262, 2277, 211, 182, 3140, 12512, 212,
        #   142, 12573, 8857, 262, 126, 126, 211, 160, 2277, 262, 7703, 2277, 15118, 10189, 2277, 22033, 12573, 455,
        #   8098, 211, 164, 212, 143, 212, 145, 28359, 11674, 261, 262, 211, 167, 15118, 22033, 21005, 9214, 262, 211,
        #   166, 4611, 10833, 212, 143, 212, 146, 8857, 8098, 211, 167, 15118, 22033, 21005, 17568, 8857, 8098, 288,
        #   296, 8098, 411, 1]]


class TestTensorflowHubTokenizer(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = TensorflowHubTokenizer('bert_en_uncased_preprocess/3', strip_sep_token=102)
        self.target_tokenizer = HFTokenizer('bert-base-uncased', strip_sep_token=True)

    def test_easy(self):
        inputs = [basic_clean(i) for i in INPUTS_EASY]

        expected = self.target_tokenizer(INPUTS_EASY).numpy()
        result = self.evaluate(self.tokenizer(inputs))

        self.assertAllEqual(expected, result)

    def test_simple(self):
        inputs = [basic_clean(i) for i in INPUTS_SIMPLE]

        expected = self.target_tokenizer(INPUTS_SIMPLE).numpy()
        result = self.evaluate(self.tokenizer(inputs))

        self.assertAllEqual(expected, result)

    def test_normal(self):
        inputs = [basic_clean(i) for i in INPUTS_NORMAL]

        expected = self.target_tokenizer(INPUTS_NORMAL).numpy()
        result = self.evaluate(self.tokenizer(inputs))

        self.assertAllEqual(expected, result)

    def test_hard(self):
        # HF tokenizer miss NFD normalization
        inputs_hard = [i.replace('″', '') for i in INPUTS_HARD]
        inputs = [basic_clean(i) for i in inputs_hard]

        expected = self.target_tokenizer(inputs_hard).numpy()
        result = self.evaluate(self.tokenizer(inputs))

        self.assertAllEqual(expected, result)


INPUTS_EASY = ['a diagrzm', 'a diagram', 'a dog', 'a cat', 'an elephant']
INPUTS_SIMPLE = ['They\'re', 'the', 'Greatest', '\xC0bc']
INPUTS_NORMAL = tf.gather.__doc__.split('\n')
INPUTS_HARD = [
    '55°20′00″&nbsp;с.&nbsp;ш.', '«1С-Битрикс:Управление сайтом».', 'http://foo.com/unicode_(✪)_in_parens',
    'Кот-д’Ивуар', 'на 2013−2014 годы', '„Lietuvos laisvės kovos sąjūdis“', '• Всего',
    '▲ 0,834&nbsp;(очень высокий)&nbsp;(35-е место)',
    'Встретился с его сыном {{Не переведено 3|Сьюард, Фредерик Уильям|Фредериком|en|Frederick W. Seward}} — ']
