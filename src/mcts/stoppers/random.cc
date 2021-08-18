/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "mcts/stoppers/random.h"

#include "mcts/stoppers/alphazero.h"
#include "mcts/stoppers/legacy.h"
#include "mcts/stoppers/smooth.h"
#include "utils/random.h"

namespace lczero {

namespace {

class RandomTimeManager : public TimeManager {
 public:
  RandomTimeManager(int64_t move_overhead, const OptionsDict& params) {
    switch (Random::Get().GetInt(0, 2)) {
      case 0:
        time_manager = MakeLegacyTimeManager(move_overhead, params);
        break;
      case 1:
        time_manager = MakeAlphazeroTimeManager(move_overhead, params);
        break;
      default:
        time_manager = MakeSmoothTimeManager(move_overhead, params);
    }
  }
  std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                            const NodeTree& tree) override;

 private:
  std::unique_ptr<TimeManager> time_manager;
};

std::unique_ptr<SearchStopper> RandomTimeManager::GetStopper(
    const GoParams& params, const NodeTree& tree) {
  return time_manager->GetStopper(params, tree);
}

}  // namespace

std::unique_ptr<TimeManager> MakeRandomTimeManager(int64_t move_overhead,
                                                   const OptionsDict& params) {
  return std::make_unique<RandomTimeManager>(move_overhead, params);
}
}  // namespace lczero
