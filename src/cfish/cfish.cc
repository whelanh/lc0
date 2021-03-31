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
*/

#include "cfish/cfish.h"

extern "C" {
bool option_set_by_name(const char *name, const char *value);
}

namespace lczero {
namespace {

const OptionId kContemptId{"contempt", "Contempt",
                           "A positive contempt value to evaluate a position "
                           "more favourably the more material is left on the "
                           "board."};

const OptionId kAnalysisContemptId{"analysis-contempt", "Analysis Contempt",
                                   "Set this option to White or Black to "
                                   "analyse with contempt for that side."};
const OptionId kThreadsId{
    "threads", "Threads",
    "The number of CPU threads used for searching a position.", 't'};
const OptionId kHashId{"hash-mb", "Hash", "The size of the hash table in MB."};
const OptionId kMultiPVId{
    "multi-pv", "MultiPV",
    "Output the N best lines when searching. Leave at 1 for best performance."};
const OptionId kMoveOverheadId{
    "move-overhead", "Move Overhead",
    "Compensation for network and GUI delay (in ms)."};
const OptionId kSlowMoverId{
    "slowmover", "Slow Mover",
    "Increase to use more time, decrease to use less time."};
const OptionId kNodestimeId{"nodestime", "nodestime", ""};
const OptionId kAnalyseModeId{"analyse-mode", "UCI_AnalyseMode", ""};
const OptionId kChess960Id{"chess960", "UCI_Chess960", ""};
const OptionId kSyzygyPathId{
    "syzygy-paths", "SyzygyPath",
    "Path to the folders/directories storing the Syzygy tablebase files.", 's'};
const OptionId kSyzygyProbeDepthId{"syzygy-probe-depth", "SyzygyProbeDepth",
                                   "Minimum remaining search depth for which a "
                                   "position is probed. Increase this value to "
                                   "probe less aggressively."};
const OptionId kSyzygy50MoveRuleId{
    "syzygy-50-move-rule", "Syzygy50MoveRule",
    "Disable to let fifty-move rule draws detected by Syzygy tablebase probes "
    "count as wins or losses. This is useful for ICCF correspondence games."};
const OptionId kSyzygyProbeLimitId{"syzygy-probe-limit", "SyzygyProbeLimit",
                                   "Limit Syzygy tablebase probing to "
                                   "positions with at most this many pieces "
                                   "left (including kings and pawns)."};
const OptionId kSyzygyUseDTMId{"syzygy-use-dtm", "SyzygyUseDTM",
                               "Use Syzygy DTM tablebases (not yet released)."};
const OptionId kEvalFileId{"weights", "EvalFile", "Name of NNUE network file.",
                           'w'};
const OptionId kLargePagesId{
    "large-pages", "LargePages",
    "Control allocation of the hash table as Large Pages"};
}  // namespace

void Cfish::Run() {
  OptionsParser options;
  options.Add<IntOption>(kContemptId, -100, 100) = 24;
  std::vector<std::string> analysis_contempt = {"Off", "White", "Black"};
  options.Add<ChoiceOption>(kAnalysisContemptId, analysis_contempt) = "Off";
  options.Add<IntOption>(kThreadsId, 1, 128) = 1;
  options.Add<IntOption>(kHashId, 1, 33554432) = 16;
  options.Add<IntOption>(kMultiPVId, 1, 500) = 1;
  options.Add<IntOption>(kMoveOverheadId, 0, 5000) = 10;
  options.Add<IntOption>(kSlowMoverId, 10, 1000) = 100;
  options.Add<IntOption>(kNodestimeId, 0, 10000) = 0;
  options.Add<BoolOption>(kAnalyseModeId) = false;
  options.Add<BoolOption>(kChess960Id) = false;
  options.Add<StringOption>(kSyzygyPathId) = "<empty>";
  options.Add<IntOption>(kSyzygyProbeDepthId, 1, 100) = 1;
  options.Add<BoolOption>(kSyzygy50MoveRuleId) = true;
  options.Add<IntOption>(kSyzygyProbeLimitId, 0, 7) = 7;
  options.Add<BoolOption>(kSyzygyUseDTMId) = true;
  options.Add<StringOption>(kEvalFileId) = "nn.bin";
  options.Add<BoolOption>(kLargePagesId) = true;

  options.HideOption(kSyzygyUseDTMId);

  if (!options.ProcessAllFlags()) return;

  auto option_dict = options.GetOptionsDict();

  option_set_by_name("Contempt",
                     std::to_string(option_dict.Get<int>(kContemptId)).c_str());
  option_set_by_name("Analysis Contempt",
                     option_dict.Get<std::string>(kAnalysisContemptId).c_str());
  option_set_by_name("Threads",
                     std::to_string(option_dict.Get<int>(kThreadsId)).c_str());
  option_set_by_name("Hash",
                     std::to_string(option_dict.Get<int>(kHashId)).c_str());
  option_set_by_name("MultiPV",
                     std::to_string(option_dict.Get<int>(kMultiPVId)).c_str());
  option_set_by_name(
      "Move Overhead",
      std::to_string(option_dict.Get<int>(kMoveOverheadId)).c_str());
  option_set_by_name(
      "Slow Mover", std::to_string(option_dict.Get<int>(kSlowMoverId)).c_str());
  option_set_by_name(
      "nodestime", std::to_string(option_dict.Get<int>(kNodestimeId)).c_str());
  option_set_by_name(
      "UCI_AnalyseMode",
      std::to_string(option_dict.Get<bool>(kAnalyseModeId)).c_str());
  option_set_by_name(
      "UCI_Chess960",
      std::to_string(option_dict.Get<bool>(kChess960Id)).c_str());
  option_set_by_name("SyzygyPath",
                     option_dict.Get<std::string>(kSyzygyPathId).c_str());
  option_set_by_name(
      "SyzygyProbeDepth",
      std::to_string(option_dict.Get<int>(kSyzygyProbeDepthId)).c_str());
  option_set_by_name(
      "Syzygy50MoveRule",
      std::to_string(option_dict.Get<bool>(kSyzygy50MoveRuleId)).c_str());
  option_set_by_name(
      "SyzygyProbeLimit",
      std::to_string(option_dict.Get<int>(kSyzygyProbeLimitId)).c_str());
  option_set_by_name(
      "SyzygyUseDTM",
      std::to_string(option_dict.Get<bool>(kSyzygyUseDTMId)).c_str());
  option_set_by_name("EvalFile",
                     option_dict.Get<std::string>(kEvalFileId).c_str());
  option_set_by_name(
      "LargePages",
      std::to_string(option_dict.Get<bool>(kLargePagesId)).c_str());

  cfish_loop();
}

}  // namespace lczero
