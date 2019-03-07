/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_ADDONS_EUCLIDEAN_DISTANCE_OP_H_
#define TENSORFLOW_ADDONS_EUCLIDEAN_DISTANCE_OP_H_

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

#include <limits>

#define IDX(x, y) Eigen::array<Eigen::DenseIndex, 4> \
                  ({coords[0], x, y, coords[3]})

namespace tensorflow {

namespace generator {

using Eigen::array;
using Eigen::DenseIndex;

template <typename Device, typename T>
class EuclideanDistanceTransformGenerator {
 private:
  typename TTypes<T, 4>::ConstTensor input_;
  int64 height, width, maxElem;

 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  EuclideanDistanceTransformGenerator(typename TTypes<T, 4>::ConstTensor input)
      : input_(input) {
    height = input_.dimensions()[1];
    width = input_.dimensions()[2];
    maxElem = std::max(height, width);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const array<DenseIndex, 4>&coords) const {
    const int64 x = coords[1];
    const int64 y = coords[2];

    if (input_(coords) == T(0))
      return T(0);

    float minDistance = std::numeric_limits<T>::max();

    for (int k = 0; k < height; ++k) {
      for (int h = 0; h < width; ++h) {
        if (input_(IDX(k, h)) == T(0)) {
          float dist = std::sqrt((x - k) * (x - k) + (h - y) * (h - y));
          minDistance = std::min(minDistance, dist);
        }
      }
    }
    return T(minDistance);
  }
};

} // end namespace generator

namespace functor {

using generator::EuclideanDistanceTransformGenerator;

template <typename Device, typename T>
struct EuclideanDistanceTransformFunctor {
  typedef typename TTypes<T, 4>::ConstTensor InputType;
  typedef typename TTypes<T, 4>::Tensor OutputType;

  EuclideanDistanceTransformFunctor() {}

  EIGEN_ALWAYS_INLINE
  void operator()(const Device& device, OutputType* output,
                  const InputType& images) const {
    output->device(device) = output->generate(
        EuclideanDistanceTransformGenerator<Device, T>(images));
  }
};

} // end namespace functor

} // end namespace tensorflow

#endif // TENSORFLOW_ADDONS_EUCLIDEAN_DISTANCE_OP_H_