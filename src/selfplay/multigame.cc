/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors

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

#include "selfplay/multigame.h"

namespace lczero {

class Evaluator {
 public:
  // Run before each batch before any Gather.
  virtual void Reset(const PlayerOptions& player) = 0;
  // Run for each tree.
  virtual void Gather(NodeTree* tree) = 0;
  // Run once between Gather and Move.
  virtual void Run() = 0;
  // Run for each tree in the same order as Gather.
  virtual void MakeBestMove(NodeTree* tree) = 0;
};

class PolicyEvaluator : public Evaluator {
 public:
  void Reset(const PlayerOptions& player) override {
    comp = player.network->NewComputation();
    input_format = player.network->GetCapabilities().input_format;
    transforms.clear();
    comp_idx = 0;
  }
  void Gather(NodeTree* tree) override {
    int transform;
    auto planes =
        EncodePositionForNN(input_format, tree->GetPositionHistory(), 8,
                            FillEmptyHistory::FEN_ONLY, &transform);
    transforms.push_back(transform);
    comp->AddInput(std::move(planes));
  }
  void Run() override { comp->ComputeBlocking(); }
  void MakeBestMove(NodeTree* tree) override {
    Move best;
    float max_p = std::numeric_limits<float>::lowest();
    for (auto edge : tree->GetCurrentHead()->Edges()) {
      float p = comp->GetPVal(comp_idx,
                              edge.GetMove().as_nn_index(transforms[comp_idx]));
      if (p >= max_p) {
        max_p = p;
        best = edge.GetMove(tree->GetPositionHistory().IsBlackToMove());
      }
    }
    tree->MakeMove(best);
    comp_idx++;
  }

  std::unique_ptr<NetworkComputation> comp;
  pblczero::NetworkFormat::InputFormat input_format;
  int comp_idx;
  std::vector<int> transforms;
};

MultiSelfPlayGames::MultiSelfPlayGames(PlayerOptions player1,
                                       PlayerOptions player2,
                                       const std::vector<Opening>& openings,
                                       SyzygyTablebase* syzygy_tb)
    : options_{player1, player2}, syzygy_tb_(syzygy_tb) {
  trees_.reserve(openings.size());
  for (auto opening : openings) {
    trees_.push_back(std::make_shared<NodeTree>());
    trees_.back()->ResetToPosition(opening.start_fen, {});
    results_.push_back(GameResult::UNDECIDED);

    for (Move m : opening.moves) {
      trees_.back()->MakeMove(m);
    }
  }
}

void MultiSelfPlayGames::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
}

void MultiSelfPlayGames::Play() {
  std::unique_ptr<Evaluator> evaluator = std::make_unique<PolicyEvaluator>();
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) break;
    }
    bool all_done = true;
    bool blacks_move = false;
    for (size_t i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] == GameResult::UNDECIDED) {
        if (tree->GetPositionHistory().ComputeGameResult() !=
            GameResult::UNDECIDED) {
          results_[i] = tree->GetPositionHistory().ComputeGameResult();
          continue;
        }
        if (syzygy_tb_ != nullptr) {
          auto board = tree->GetPositionHistory().Last().GetBoard();
          if (board.castlings().no_legal_castle() &&
              (board.ours() | board.theirs()).count() <=
                  syzygy_tb_->max_cardinality()) {
            auto tb_side_black = (tree->GetPlyCount() % 2) == 1;
            ProbeState state;
            const WDLScore wdl = syzygy_tb_->probe_wdl(
                tree->GetPositionHistory().Last(), &state);
            // Only fail state means the WDL is wrong, probe_wdl may produce
            // correct result with a stat other than OK.
            if (state != FAIL) {
              if (wdl == WDL_WIN) {
                results_[i] = tb_side_black ? GameResult::BLACK_WON
                                            : GameResult::WHITE_WON;
              } else if (wdl == WDL_LOSS) {
                results_[i] = tb_side_black ? GameResult::WHITE_WON
                                            : GameResult::BLACK_WON;
              } else {  // Cursed wins and blessed losses count as draws.
                results_[i] = GameResult::DRAW;
              }
              continue;
            }
          }
        }
        if (all_done) {
          all_done = false;
          blacks_move = (tree->GetPlyCount() % 2) == 1;
          // Don't break as we need to update result state for everything.
        }
      }
    }
    if (all_done) break;
    const int idx = blacks_move ? 1 : 0;
    evaluator->Reset(options_[idx]);
    for (size_t i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] != GameResult::UNDECIDED) {
        continue;
      }
      if (((tree->GetPlyCount() % 2) == 1) != blacks_move) continue;
      const auto& board = tree->GetPositionHistory().Last().GetBoard();
      auto legal_moves = board.GenerateLegalMoves();
      tree->GetCurrentHead()->CreateEdges(legal_moves);
      evaluator->Gather(tree.get());
    }
    evaluator->Run();
    for (size_t i = 0; i < trees_.size(); i++) {
      const auto& tree = trees_[i];
      if (results_[i] != GameResult::UNDECIDED) {
        continue;
      }
      if (((tree->GetPlyCount() % 2) == 1) != blacks_move) continue;
      evaluator->MakeBestMove(tree.get());
    }
  }
}

}  // namespace lczero