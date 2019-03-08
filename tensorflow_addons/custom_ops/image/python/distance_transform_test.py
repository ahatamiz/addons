# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for distance transform ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow_addons.custom_ops.image.python import distance_transform as distance_tranform_ops

_OUTPUT_DTYPES = [tf.float16, tf.float32, tf.float64]


class DistanceOpsTest(tf.test.TestCase):
    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_compose(self):
        for dtype in _OUTPUT_DTYPES:
            image = tf.constant(
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0],
                 [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
                dtype=tf.uint8)
            image = tf.reshape(image, [5, 5, 1])

            output = distance_tranform_ops.euclidean_dist_transform(
                image, dtype=dtype)

            output_nd = np.reshape(output.numpy(), [-1])
            expected_output = np.array([
                2, 2.23606801, 2, 2.23606801, 2, 1, 1.41421354, 1, 1.41421354,
                1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
            ])
            self.assertNDArrayNear(output_nd, expected_output, 1e-3)
            self.assertEqual(output.dtype, dtype)


if __name__ == "__main__":
    tf.test.main()
