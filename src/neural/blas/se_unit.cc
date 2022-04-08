/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018 The LCZero Authors

 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "neural/blas/se_unit.h"
#include "neural/blas/fully_connected_layer.h"

#include <cmath>

namespace lczero {
namespace {
constexpr int kWidth = 8;
constexpr int kHeight = 8;
constexpr int kSquares = kWidth * kHeight;
}  // namespace

static void global_avg_pooling(const size_t batch_size, const size_t channels,
                               const float* M, const float* bias,
                               float* output) {
  static constexpr auto kWtiles = (kWidth + 1) / 2;  // 4
  static constexpr auto kTiles = kWtiles * kWtiles;  // 16

  for (size_t batch_index = 0; batch_index < batch_size; batch_index++) {
    const float* M_batch = &M[channels * kTiles * batch_index];

    for (size_t channel = 0; channel < channels; channel++) {
      const float* M_channel = M_batch + channel;
      auto acc = 0.0f;

      for (int block_x = 0; block_x < kWtiles; block_x++) {
        for (int block_y = 0; block_y < kWtiles; block_y++) {
          const auto b = block_y * kWtiles + block_x;
          const float* M_wtile = M_channel + channels * b;
          const auto M_incr = channels * kTiles * batch_size;

          acc += M_wtile[0] + 2 * M_wtile[M_incr] - M_wtile[3 * M_incr] +
                 2 * M_wtile[4 * M_incr] + 4 * M_wtile[5 * M_incr] -
                 2 * M_wtile[7 * M_incr] - M_wtile[12 * M_incr] -
                 2 * M_wtile[13 * M_incr] + M_wtile[15 * M_incr];
        }
      }
      *(output++) = acc / kSquares + bias[channel];
    }
  }
}

static void scale(const size_t channels, const size_t batch_size,
                  const float* scale, const float* bias, float* s, float* b) {
  const auto lambda_sigmoid = [](const auto val) {
    return 1.0f / (1.0f + std::exp(-val));
  };

  for (auto batch = size_t{0}; batch < batch_size; batch++)
    for (auto ch = size_t{0}; ch < channels; ch++) {
      auto c = batch * channels + ch;
      auto gamma = lambda_sigmoid(scale[c + batch * channels]);
      auto beta = scale[c + batch * channels + channels] + gamma * bias[ch];
      s[c] = gamma;
      b[c] = beta;
    }
}

template <bool use_eigen>
void SEUnit(const size_t batch_size, const size_t channels,
            const size_t se_fc_outputs, const float* input, const float* bias,
            const float* weights_w1, const float* weights_b1,
            const float* weights_w2, const float* weights_b2, float* s,
            float* b, const ActivationFunction activation) {
  std::vector<float> pool(2 * channels * batch_size);
  std::vector<float> fc_out1(batch_size * se_fc_outputs);

  global_avg_pooling(batch_size, channels, input, bias, pool.data());

  FullyConnectedLayer<use_eigen>::Forward1D(batch_size, channels, se_fc_outputs,
                                            pool.data(), weights_w1, weights_b1,
                                            activation,  // Activation On
                                            fc_out1.data());

  FullyConnectedLayer<use_eigen>::Forward1D(batch_size, se_fc_outputs,
                                            2 * channels, fc_out1.data(),
                                            weights_w2, weights_b2,
                                            NONE,  // Activation Off
                                            pool.data());

  scale(channels, batch_size, pool.data(), bias, s, b);
}

template void SEUnit<true>(const size_t batch_size, const size_t channels,
                           const size_t se_fc_outputs, const float* input,
                           const float* bias, const float* weights_w1,
                           const float* weights_b1, const float* weights_w2,
                           const float* weights_b2, float* s, float* b,
                           const ActivationFunction activation);
#ifdef USE_BLAS
template void SEUnit<false>(const size_t batch_size, const size_t channels,
                            const size_t se_fc_outputs, const float* input,
                            const float* bias, const float* weights_w1,
                            const float* weights_b1, const float* weights_w2,
                            const float* weights_b2, float* s, float* b,
                            const ActivationFunction activation);
#endif
}  // namespace lczero
