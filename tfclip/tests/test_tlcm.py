import numpy as np
from keras.src import backend
from keras.src import testing

from tfclip.tlcm import TokenLastCausalMask


class TestTokenLastCausalMask(testing.TestCase):
    def test_layer(self):
        inputs = np.array([[1, 1, 0, 0]], dtype="int64")
        outputs = TokenLastCausalMask()(inputs)
        outputs = backend.convert_to_numpy(outputs)
        self.assertTrue(
            (
                outputs
                == np.array(
                    [
                        [
                            [True, False, False, False, False],
                            [True, True, False, False, False],
                            [True, True, True, False, False],
                            [True, True, True, True, False],
                            [True, True, True, False, False],
                        ]
                    ]
                )
            ).all()
        )
