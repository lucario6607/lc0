/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>

#include <cassert>
#include <version>

#include "proto/lc0ex.pb.h"
#include "runtime.h"

namespace lczero::lc0ex::cuda {

// Forward declarations of CUDA types to avoid including CUDA headers in this
// header.
using Device = int;
struct ContextStruct;
using Context = ContextStruct*;
struct ModuleStruct;
using Module = ModuleStruct*;
struct StreamStruct;
using Stream = StreamStruct*;
struct EventStruct;
using Event = EventStruct*;
struct GraphExecStruct;
using GraphExec = GraphExecStruct*;
struct GraphStruct;
using Graph = GraphStruct*;

class CudaEvent;

class CudaStream {
 public:
  CudaStream() = default;
  CudaStream(StreamFlags flags);
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&& other) noexcept;
  ~CudaStream();

  explicit operator bool() const { return stream_ != nullptr; }

  bool IsIdle() const;
  void Synchronize() const;

  void WaitEvent(CudaEvent& event) const;
  void RecordEvent(CudaEvent& event) const;

  template <typename StreamType>
  operator StreamType() const;

 private:
  Stream stream_ = nullptr;
};

class CudaMemory;
class CudaHostMemory;

template <typename T>
using CudaBuffer = BufferBase<T, CudaMemory>;
template <typename T>
using CudaHostBuffer = BufferBase<T, CudaHostMemory>;

// A thin object to manage a GPU memory allocation. The main use case is a
// shared allocation which is shared between tensors.
class CudaMemory : public MemoryBase {
 public:
  CudaMemory() = default;
  CudaMemory(size_t size, size_t alignment);

  CudaMemory(const CudaMemory&) = delete;
  CudaMemory& operator=(const CudaMemory&) = delete;

  CudaMemory(CudaMemory&& other) noexcept;
  CudaMemory& operator=(CudaMemory&& other) noexcept;

  ~CudaMemory();

 protected:
  void* device_ptr_ = nullptr;
};

// A CudaHostBuffer is a CudaBuffer that allocates pinned host memory using
// cuHostAlloc. It is used to as a staging area for inputs and final destination
// for outputs. Data will be transferred to GPU memory using cuMemcpyAsync.
class CudaHostMemory : public MemoryBase {
 public:
  CudaHostMemory(size_t size, size_t alignment);
  CudaHostMemory(const CudaHostMemory&) = delete;
  CudaHostMemory& operator=(const CudaHostMemory&) = delete;
  CudaHostMemory(CudaHostMemory&& other) noexcept;
  CudaHostMemory& operator=(CudaHostMemory&& other) noexcept;
  ~CudaHostMemory();

 private:
  void* host_ptr_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() = default;
  CudaEvent(EventFlags flags);
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& other) noexcept;
  CudaEvent& operator=(CudaEvent&& other) noexcept;
  ~CudaEvent();

  explicit operator bool() const { return event_ != nullptr; }

  bool IsCompleted() const;
  void Synchronize() const;

  template <typename EventType>
  operator EventType() const;

 private:
  Event event_ = nullptr;
};

class CudaRuntime;

class ArgumentNull {
 public:
  ArgumentNull() = default;

  template <typename State>
  void operator()(void*& arg, uint64_t& value, const State& state) const;

  bool RequiresModification() const { return false; }
  std::tuple<size_t, size_t> GetOffsetAndSize() const { return {0, 0}; }
};
class ArgumentSymbol {
 public:
  ArgumentSymbol(void* symbol) : symbol_(symbol) {}

  template <typename State>
  void operator()(void*& arg, uint64_t& value, const State& state) const;

  bool RequiresModification() const { return false; }
  std::tuple<size_t, size_t> GetOffsetAndSize() const { return {0, 0}; }

 private:
  void* symbol_;
};
class ArgumentPersistentBuffer {
 public:
  ArgumentPersistentBuffer(size_t offset, size_t streamed_size)
      : offset_(offset), streamed_size_(streamed_size) {}

  template <typename State>
  void operator()(void*& arg, uint64_t& value, const State& state) const;

  bool RequiresModification() const { return false; }

  std::tuple<size_t, size_t> GetOffsetAndSize() const {
    return {offset_, streamed_size_};
  }

 private:
  size_t offset_;
  size_t streamed_size_ = 0;
};
class ArgumentExecutionBuffer {
 public:
  ArgumentExecutionBuffer(size_t offset) : offset_(offset) {}

  template <typename State>
  void operator()(void*& arg, uint64_t& value, const State& state) const;

  bool RequiresModification() const { return false; }

  std::tuple<size_t, size_t> GetOffsetAndSize() const { return {0, 0}; }

 private:
  size_t offset_;
};

class Operator;

using OpVector = absl::InlinedVector<Operator, 8>;

enum class OperatorType {
  // Binary operations
  kAdd,
  kSubtract,
  kMultiply,
  kDivide,
  // Unary operations
  kNegate,
  kAbs,
  // Variable operations
  kReadBatchSize,
  kReadTotalLegalMoves,
  // Constant value
  kNumber,
};

class Operator {
 public:
  Operator(OperatorType op, unsigned idx1 = 0, unsigned idx2 = 0)
      : op_(op), children_{idx1, idx2} {}

  Operator(OperatorType op, long value) : op_(op), value_(value) {}

  template <typename State>
  long operator()(const OpVector& other, State& state) const;

  bool RequiresModification() const {
    return op_ == OperatorType::kReadTotalLegalMoves;
  }

 private:
  struct Children {
    unsigned idx1_;
    unsigned idx2_;
  };
  OperatorType op_;
  union {
    long value_;
    Children children_;
  };
};

class CudaExecutable;

class GridFormula {
 public:
  GridFormula() = default;

  GridFormula& operator=(std::string_view formula);

  template <typename State>
  long operator()(State& state) const;

  bool RequiresModification() const;

 private:
  OpVector operators_;
};

class ArgumentParameter {
 public:
  ArgumentParameter(std::string_view formula);

  template <typename State>
  void operator()(void*& arg, uint64_t& value, const State& state) const;

  bool RequiresModification() const;
  std::tuple<size_t, size_t> GetOffsetAndSize() const { return {0, 0}; }

 private:
  OpVector operators_;
};

using ArgumentTypes =
    std::variant<ArgumentNull, ArgumentSymbol, ArgumentPersistentBuffer,
                 ArgumentExecutionBuffer, ArgumentParameter>;

class ArgumentVariant : public ArgumentTypes {
 public:
  using ArgumentTypes::variant;
};

class KernelNode : public NodeBase {
 public:
  KernelNode(const pblczero::Node& node, CudaExecutable& executable,
             const pblczero::Kernel& kernel, void* function);
  ~KernelNode() = default;

  template <typename State>
  auto operator()(State& state) const;

  bool RequiresModification() const;

  void SetDebugState(uint64_t kernel_id, uint64_t node_id) {
#ifndef _NDEBUG
    kernel_id_ = kernel_id;
    node_id_ = node_id;
#endif
  }

 private:
  absl::InlinedVector<ArgumentVariant, 8> arguments_;
  void* function_ = nullptr;
  std::array<GridFormula, 3> grid_;
  std::array<unsigned int, 3> block_;
  unsigned int dynamic_shared_memory_bytes_ = 0;
#ifndef _NDEBUG
  uint64_t kernel_id_;
  uint64_t node_id_;
#endif
};

enum class RecordEventType {
  kSleep,
  kComputeOrdering,
  kPolicyDownloadDone,
  kWdlDownloadDone,
  kMlhDownloadDone,
};

template <RecordEventType event>
class EventRecordNode : public NodeBase {
 public:
  EventRecordNode(const pblczero::Node& event_record);
  ~EventRecordNode() = default;

  template <typename State>
  auto operator()(State& state) const;
};

enum class WaitEventType {
  kComputeOrdering,
};

template <WaitEventType event>
class EventWaitNode : public NodeBase {
 public:
  EventWaitNode(const pblczero::Node& event_wait);

  template <typename State>
  auto operator()(State& state) const;
};

enum class MemcpyBuffer {
  kInputMapping,
  kInputMask,
  kInputValue,
  kOutputsPolicy,
  kOutputsWdl,
  kOutputsMlh,
};

template <MemcpyBuffer buffer>
class MemcpyNode : public NodeBase {
 public:
  MemcpyNode(const pblczero::Node& memcpy_node);

  template <typename State>
  auto operator()(State& state) const;

 private:
  ptrdiff_t gpu_offset_ = 0;
};

using NodeTypes =
    std::variant<KernelNode, EventRecordNode<RecordEventType::kSleep>,
                 EventRecordNode<RecordEventType::kComputeOrdering>,
                 EventRecordNode<RecordEventType::kPolicyDownloadDone>,
                 EventRecordNode<RecordEventType::kWdlDownloadDone>,
                 EventRecordNode<RecordEventType::kMlhDownloadDone>,
                 EventWaitNode<WaitEventType::kComputeOrdering>,
                 MemcpyNode<MemcpyBuffer::kInputMapping>,
                 MemcpyNode<MemcpyBuffer::kInputMask>,
                 MemcpyNode<MemcpyBuffer::kInputValue>,
                 MemcpyNode<MemcpyBuffer::kOutputsPolicy>,
                 MemcpyNode<MemcpyBuffer::kOutputsWdl>,
                 MemcpyNode<MemcpyBuffer::kOutputsMlh>>;

class NodeVariant : public NodeTypes {
 public:
  using NodeTypes::variant;
};

class CudaGraphExec;

class GraphExecModification {
 public:
  GraphExecModification(std::function<void(const CudaGraphExec&)> modify_func)
      : modify_func_(std::move(modify_func)) {};
  GraphExecModification(const GraphExecModification&) = delete;
  GraphExecModification& operator=(const GraphExecModification&) = delete;
  GraphExecModification(GraphExecModification&& other) noexcept;
  GraphExecModification& operator=(GraphExecModification&& other) noexcept;
  ~GraphExecModification() = default;

  void operator()(const CudaGraphExec& exec) const;

 private:
  std::function<void(const CudaGraphExec&)> modify_func_;
};

class GraphCapture {
 public:
  GraphCapture();
  GraphCapture(const GraphCapture&) = delete;
  GraphCapture& operator=(const GraphCapture&) = delete;
  GraphCapture(GraphCapture&& other) noexcept;
  GraphCapture& operator=(GraphCapture&& other) noexcept;
  ~GraphCapture();

  template <typename GraphType>
    requires std::is_pointer_v<GraphType>
  operator GraphType() const;

  std::vector<GraphExecModification> modifications_;

  Graph release() {
    Graph g = graph_;
    graph_ = nullptr;
    return g;
  }

 private:
  Graph graph_;
};

class CudaKernel;

class CudaProgram {
 public:
  CudaProgram(const pblczero::Program& source, CudaExecutable& executable,
              const std::vector<pblczero::Kernel>& kernels);
  ~CudaProgram() = default;

  size_t Size() const { return batch_size_; }

  template <typename State>
  void Run(State& state) const;

  template <typename State>
  GraphCapture Capture(GraphMode mode, State& state) const;

 private:
  std::vector<NodeVariant> nodes_;
  size_t batch_size_;
};

class CudaGraphExec {
 public:
  CudaGraphExec() = default;
  CudaGraphExec(GraphCapture& graph);
  CudaGraphExec(const CudaGraphExec&) = delete;
  CudaGraphExec& operator=(const CudaGraphExec&) = delete;
  CudaGraphExec(CudaGraphExec&& other) noexcept;
  CudaGraphExec& operator=(CudaGraphExec&& other) noexcept;
  ~CudaGraphExec();

  explicit operator bool() const { return graph_exec_ != nullptr; }

  template <typename GraphExecType>
  operator GraphExecType() const;

  void Upload(CudaStream& stream) const;
  void Launch(CudaStream& stream) const;

 private:
  GraphExec graph_exec_ = nullptr;
  Graph graph_ = nullptr;
  std::vector<GraphExecModification> modifications_;
};

class CudaExecutable {
 public:
  CudaExecutable(const pblczero::NeuralExecutable& source);
  ~CudaExecutable() = default;

  size_t GetPersistentAllocationSize() const;
  void CopyPersistentFromHost(
      std::span<const std::byte> source,
      std::optional<size_t> size_bytes = std::nullopt) const;

  size_t GetExecutionAllocationSize() const;
  size_t GetExecutionAllocationAlignment() const;

  size_t GetMaxBatchSize() const;

  template <typename T>
  void Run(size_t batch_size, ComputationState<CudaRuntime, T>& state,
           CudaEvent& compute_ordering_event);

  template <typename T>
  GraphCapture Capture(GraphMode mode, size_t batch_size,
                       ComputationState<CudaRuntime, T>& state,
                       CudaEvent& compute_ordering_event);

 protected:
  std::vector<Module> modules_;
  std::vector<CudaProgram> programs_;
  CudaMemory persistent_allocation_;
  AllocationInfo execution_allocation_;
};

class CudaRuntime {
 public:
  using Executable = CudaExecutable;
  using Stream = CudaStream;
  using Memory = CudaMemory;
  using HostMemory = CudaHostMemory;
  template <typename T>
  using Buffer = CudaBuffer<T>;
  template <typename T>
  using HostBuffer = CudaHostBuffer<T>;
  using Event = CudaEvent;
  using GraphExec = CudaGraphExec;
  CudaRuntime(int device_ordinal);
  ~CudaRuntime();

  void SetCurrent() const;

 private:
  Device device_ = 0;
  Context context_ = nullptr;
};

}  // namespace lczero::lc0ex::cuda
