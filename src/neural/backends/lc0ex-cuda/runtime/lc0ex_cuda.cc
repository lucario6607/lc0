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

#include "lc0ex_cuda.h"

#include <absl/container/inlined_vector.h>
#include <cuda.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "neural/network.h"
#include "proto/lc0ex.pb.h"
#include "proto/lc0ex_metadata.pb.h"
#include "utils/exception.h"
#include "utils/trace.h"

// TODO: Move these to a separate shared implementation file.
namespace lczero::lc0ex {
void MemoryBase::IsValidRange(std::string_view name, ptrdiff_t offset,
                              size_t size
#if __cpp_lib_source_location >= 201907L
                              ,
                              const std::source_location& location
#endif
) const {
  if (offset < 0 || size == 0 || offset + size > size_) {
    std::ostringstream oss;
    oss << "Invalid range for buffer " << name << ": offset=" << offset
        << ", size=" << size << ", buffer_size=" << size_
#if __cpp_lib_source_location >= 201907L
        << ", location=" << location.file_name() << ":" << location.line()
        << " in " << location.function_name()
#endif
        ;
    throw Exception(oss.str());
  }
}

NodeBase::NodeBase(const pblczero::Node& node)
    : dependencies_(node.dependencies().begin(), node.dependencies().end()),
      priority_(node.priority()) {}
}  // namespace lczero::lc0ex

namespace lczero::lc0ex::cuda {

namespace {
constexpr std::uint32_t kMagic = 0x1c0e;
constexpr std::uint32_t kFormat = 1;

[[noreturn]] void ThrowCuda(CUresult status, const char* expression,
                            const char* file, int line) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &description);

  std::string message = expression;
  message += " failed at ";
  message += file;
  message += ":";
  message += std::to_string(line);
  message += ": ";
  message += name ? name : "unknown CUDA error";
  if (description) {
    message += " (";
    message += description;
    message += ")";
  }
  throw Exception(message);
}

#define LC0EX_CUDA_CHECK(expression)                            \
  do {                                                          \
    const CUresult lc0ex_status = (expression);                 \
    if (lc0ex_status != CUDA_SUCCESS)                           \
      ThrowCuda(lc0ex_status, #expression, __FILE__, __LINE__); \
  } while (false)

std::size_t ElementSize(pblczero::Buffer::DataType type) {
  switch (type) {
    case pblczero::Buffer::DATA_TYPE_F32:
      return sizeof(float);
    case pblczero::Buffer::DATA_TYPE_U8:
      return sizeof(std::uint8_t);
    case pblczero::Buffer::DATA_TYPE_F16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_U32:
      return sizeof(std::uint32_t);
    case pblczero::Buffer::DATA_TYPE_U64:
      return sizeof(std::uint64_t);
    case pblczero::Buffer::DATA_TYPE_BF16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_UNKNOWN:
      break;
  }
  throw Exception("Unsupported or unknown buffer data type.");
}

// Helper class to add extra implementation functions which aren't exposed in
// the header file.
class ExecutableImpl : public CudaExecutable {
 public:
  void BuildModules(const pblczero::NeuralExecutable& source) {
    LCTRACE_FUNCTION_SCOPE;
    this->modules_.reserve(source.binaries_size());
    for (const auto& binary : source.binaries()) {
      CUmodule module = nullptr;
      LC0EX_CUDA_CHECK(cuModuleLoadData(&module, binary.data().data()));
      this->modules_.push_back(reinterpret_cast<Module>(module));
    }
  }

  void BuildPersistentAllocation(const pblczero::NeuralExecutable& source) {
    if (source.has_persistent_allocation()) {
      if (!source.persistent_allocation().has_size_bytes()) {
        throw Exception("Persistent allocation must have size_bytes.");
      }
      if (!source.persistent_allocation().has_alignment_bytes()) {
        throw Exception("Persistent allocation must have alignment_bytes.");
      }
      size_t size = source.persistent_allocation().size_bytes();
      size_t alignment = source.persistent_allocation().alignment_bytes();
      this->persistent_allocation_ = CudaMemory(size, alignment);
    }
  }

  template <typename T>
  void BuildExecutionAllocation(const T& source) {
    if (source.has_execution_allocation()) {
      if (!source.execution_allocation().has_size_bytes()) {
        throw Exception("Execution allocation must have size_bytes.");
      }
      if (!source.execution_allocation().has_alignment_bytes()) {
        throw Exception("Execution allocation must have alignment_bytes.");
      }
      if (source.execution_allocation().size_bytes() <
          execution_allocation_.size_bytes_) {
        return;
      }
      execution_allocation_.size_bytes_ =
          source.execution_allocation().size_bytes();
      execution_allocation_.alignment_bytes_ =
          source.execution_allocation().alignment_bytes();
    }
  }

  void BuildPrograms(const pblczero::NeuralExecutable& source) {
    this->programs_.reserve(source.programs_size());
    for (const auto& program : source.programs()) {
      const auto name = std::string(program.name());

      this->programs_.emplace_back(program, *this, source.kernels());
    }
  }

  CUmodule GetBinary(size_t idx) const {
    if (idx >= this->modules_.size()) {
      throw Exception("Binary index out of range.");
    }
    return reinterpret_cast<CUmodule>(this->modules_[idx]);
  }
};

}  // namespace

// The helper class to maintain state for graph node execution to a CUDA
// stream.
template <typename T>
class LaunchState {
 public:
  using ComputeType = T;
  static constexpr bool is_cuda_capturing = false;
  ComputationState<CudaRuntime, T>& cs_;
  size_t batch_size_;
  const CudaMemory& persistent_memory_;
  CudaEvent& compute_ordering_event_;

  absl::InlinedVector<uint64_t, 8> argval;
  absl::InlinedVector<void*, 8> argptr;
};

// The helper class to maintain state for graph capture using CUDA graph API. It
// can build either a linear or a DAG graph depending on the graph mode.
template <typename T>
class CaptureState : public LaunchState<T> {
 public:
  static constexpr bool is_cuda_capturing = true;
  GraphCapture graph_;
  std::vector<CUgraphNode> graph_nodes_;
  absl::InlinedVector<CUgraphNode, 4> dependencies_;
};

// Formula parsing to AST.
class Lexer {
 public:
  explicit Lexer(std::string_view input) : input_(input) {}

  enum TokenType { kIdentifier, kNumber, kOp, kEnd };

  std::tuple<TokenType, std::string_view> NextToken() {
    SkipWhitespace();
    if (pos_ >= input_.size()) {
      return {kEnd, {}};
    }
    size_t start = pos_;
    if (std::isdigit(input_[pos_]) ||
        (input_[pos_] == '-' && pos_ + 1 < input_.size() &&
         std::isdigit(input_[pos_ + 1]))) {
      ++pos_;
      while (pos_ < input_.size() && std::isdigit(input_[pos_])) {
        ++pos_;
      }
      return {kNumber, input_.substr(start, pos_ - start)};
    }
    if (input_[pos_] == '+' || input_[pos_] == '-' || input_[pos_] == '*' ||
        input_[pos_] == '/' || input_[pos_] == '(' || input_[pos_] == ')') {
      ++pos_;
      return {kOp, input_.substr(start, 1)};
    }

    if (!std::isalpha(input_[pos_]) && input_[pos_] != '_') {
      throw Exception(
          "Unexpected character in formula: " + std::string(input_) + " at " +
          std::to_string(pos_) + ": " + std::string(1, input_[pos_]));
    }

    while (pos_ < input_.size() &&
           (std::isalpha(input_[pos_]) || input_[pos_] == '_')) {
      ++pos_;
    }
    return {kIdentifier, input_.substr(start, pos_ - start)};
  }

  void SkipWhitespace() {
    while (pos_ < input_.size() && std::isspace(input_[pos_])) {
      ++pos_;
    }
  }

  std::string_view input_;
  size_t pos_ = 0;
};

class RecursiveParser {
 public:
  explicit RecursiveParser(std::string_view input) : lexer_(input) {}

  unsigned ParseIdentifier(std::string_view identifier) {
    if (identifier == "total_legal_moves") {
      result_.emplace_back(OperatorType::kReadTotalLegalMoves);
      return result_.size() - 1;
    } else if (identifier == "batch_size") {
      result_.emplace_back(OperatorType::kReadBatchSize);
      return result_.size() - 1;
    } else if (identifier == "abs") {
      std::tie(token, value) = lexer_.NextToken();
      if (token != Lexer::kOp || value != "(") {
        throw Exception("Expected '(' after 'abs' in formula.");
      }
      unsigned idx = Parse();
      if (token != Lexer::kOp || value != ")") {
        throw Exception("Expected ')' after 'abs' argument in formula.");
      }
      result_.emplace_back(OperatorType::kAbs, idx);
      return result_.size() - 1;
    } else {
      throw Exception("Unknown identifier: " + std::string(identifier));
    }
  }

  unsigned ParseNumber(std::string_view number) {
    try {
      long value = std::stol(std::string(number));
      result_.emplace_back(OperatorType::kNumber, value);
      return result_.size() - 1;
    } catch (const std::exception& e) {
      throw Exception(
          "Invalid number in formula: " + std::string(lexer_.input_) + " at " +
          std::to_string(lexer_.pos_) + std::string(number));
    }
  }

  struct BinaryOpPriority {
    char op;
    int priority;
    OperatorType type;
  };

  static constexpr std::array<BinaryOpPriority, 4> kBinaryOpPriorities = {{
      {'+', 1, OperatorType::kAdd},
      {'-', 1, OperatorType::kSubtract},
      {'*', 2, OperatorType::kMultiply},
      {'/', 2, OperatorType::kDivide},
  }};

  unsigned ParseOperator(unsigned left_idx, int priority) {
    if (token != Lexer::kOp) {
      return left_idx;
    }
    if (value == ")") {
      return left_idx;
    }
    auto iter = std::find_if(
        kBinaryOpPriorities.begin(), kBinaryOpPriorities.end(),
        [this](const BinaryOpPriority& op) { return op.op == value[0]; });
    if (iter == kBinaryOpPriorities.end()) {
      throw Exception("Unknown binary operator: " + std::string(lexer_.input_) +
                      " at " + std::to_string(lexer_.pos_) + ": " +
                      std::string(value));
    }
    auto op_type = iter->type;
    int op_priority = iter->priority;
    if (op_priority <= priority) {
      return left_idx;
    }
    unsigned right_idx = Parse(op_priority);
    result_.emplace_back(op_type, left_idx, right_idx);
    return ParseOperator(result_.size() - 1, priority);
  }

  unsigned ParseUnaryOperator(std::string_view op) {
    if (op == "-") {
      unsigned idx = Parse();
      if (idx >= result_.size()) {
        throw Exception(
            "Invalid index returned from Parse() in unary '-' formula.");
      }
      result_.emplace_back(OperatorType::kNegate, idx);
      return result_.size() - 1;
    } else if (op == "(") {
      unsigned idx = Parse();
      if (token != Lexer::kOp || value != ")") {
        throw Exception(
            "Expected ')' after '(': " + std::string(lexer_.input_) + " at " +
            std::to_string(lexer_.pos_) + ": " + std::string(value));
      }
      return idx;
    } else {
      throw Exception("Unknown unary operator: " + std::string(lexer_.input_) +
                      " at " + std::to_string(lexer_.pos_) + ": " +
                      std::string(op));
    }
  }

  unsigned Parse(int priority = 0) {
    std::tie(token, value) = lexer_.NextToken();
    unsigned left_idx = 0;
    switch (token) {
      case Lexer::kIdentifier:
        left_idx = ParseIdentifier(value);
        break;
      case Lexer::kNumber:
        left_idx = ParseNumber(value);
        break;
      case Lexer::kOp:
        left_idx = ParseUnaryOperator(value);
        break;
      case Lexer::kEnd:
        throw Exception("Unexpected end: " + std::string(lexer_.input_) +
                        " at " + std::to_string(lexer_.pos_));
    }
    std::tie(token, value) = lexer_.NextToken();
    return ParseOperator(left_idx, priority);
  }

  Lexer lexer_;
  OpVector result_;
  Lexer::TokenType token;
  std::string_view value;
};

// Kernel argument implementation. Each type implements its own kernel argument
// logic based on the Triton node arguments.
template <typename State>
void ArgumentNull::operator()(void*& arg, std::uint64_t& value,
                              const State&) const {
  arg = static_cast<void*>(&value);
  value = 0;
}

template <typename State>
void ArgumentSymbol::operator()(void*& arg, std::uint64_t& value,
                                const State&) const {
  arg = static_cast<void*>(&value);
  value = reinterpret_cast<std::uint64_t>(symbol_);
}

template <typename State>
void ArgumentPersistentBuffer::operator()(void*& arg, std::uint64_t& value,
                                          const State& state) const {
  arg = static_cast<void*>(&value);
  value = reinterpret_cast<std::uint64_t>(state.persistent_memory_.Data() +
                                          offset_);
}

template <typename State>
void ArgumentExecutionBuffer::operator()(void*& arg, std::uint64_t& value,
                                         const State& state) const {
  arg = static_cast<void*>(&value);
  value = reinterpret_cast<std::uint64_t>(state.cs_.device_memory_.Data() +
                                          offset_);
}

ArgumentParameter::ArgumentParameter(std::string_view formula) {
  RecursiveParser parser(formula);
  unsigned idx = parser.Parse();
  if (idx >= parser.result_.size()) {
    throw Exception(
        "Invalid index returned from Parse(): " + std::string(formula) +
        " at " + std::to_string(parser.lexer_.pos_) + ": " +
        std::string(parser.value));
  }
  if (parser.token != Lexer::kEnd) {
    throw Exception(
        "Unexpected token after formula: " + std::string(parser.lexer_.input_) +
        " at " + std::to_string(parser.lexer_.pos_) + ": " +
        std::string(parser.value));
  }
  operators_ = std::move(parser.result_);
}

template <typename State>
void ArgumentParameter::operator()(void*& arg, std::uint64_t& value,
                                   const State& state) const {
  arg = static_cast<void*>(&value);
  value = operators_.back()(operators_, state);
}

bool ArgumentParameter::RequiresModification() const {
  for (const auto& op : operators_) {
    if (op.RequiresModification()) {
      return true;
    }
  }
  return false;
}

// The CudaStream class implementation. It wraps a CUDA stream and provides
// methods to manage its lifecycle and operations.
CudaStream::CudaStream(StreamFlags flags) {
  int cuda_flags = CU_STREAM_NON_BLOCKING;
  if (flags != StreamFlags::DEFAULT) {
    throw Exception("Unsupported stream flags (" +
                    std::to_string(static_cast<int>(flags)) + ").");
  }
  CUstream stream = nullptr;
  LC0EX_CUDA_CHECK(cuStreamCreate(&stream, cuda_flags));
  stream_ = reinterpret_cast<Stream>(stream);
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}
CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    if (stream_ != nullptr) {
      LC0EX_CUDA_CHECK(cuStreamDestroy(*this));
    }
    stream_ = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) {
    LC0EX_CUDA_CHECK(cuStreamDestroy(*this));
  }
}

template <typename StreamType>
CudaStream::operator StreamType() const {
  static_assert(std::is_same_v<StreamType, CUstream>,
                "StreamType must be CUstream");
  return reinterpret_cast<CUstream>(stream_);
}

bool CudaStream::IsIdle() const {
  assert(stream_ != nullptr);
  CUresult status = cuStreamQuery(*this);
  if (status == CUDA_SUCCESS) return true;
  if (status == CUDA_ERROR_NOT_READY) return false;
  ThrowCuda(status, "cuStreamQuery", __FILE__, __LINE__);
}

void CudaStream::Synchronize() const {
  assert(stream_ != nullptr);
  LC0EX_CUDA_CHECK(cuStreamSynchronize(*this));
}

void CudaStream::WaitEvent(CudaEvent& event) const {
  assert(stream_ != nullptr);
  assert(event);
  LC0EX_CUDA_CHECK(cuStreamWaitEvent(*this, event, 0));
}

void CudaStream::RecordEvent(CudaEvent& event) const {
  assert(stream_ != nullptr);
  assert(event);
  LC0EX_CUDA_CHECK(cuEventRecord(event, *this));
}

// The CudaEvent class implementation. It wraps a CUDA event and provides
// methods to manage its lifecycle and operations.
CudaEvent::CudaEvent(EventFlags flags) {
  int cuda_flags = 0;
  if ((flags & EventFlags::BLOCKING) == EventFlags::BLOCKING) {
    cuda_flags |= CU_EVENT_BLOCKING_SYNC;
    flags = flags & ~EventFlags::BLOCKING;
  }
  if ((flags & EventFlags::USE_TIMING) == EventFlags::USE_TIMING) {
    flags = flags & ~EventFlags::USE_TIMING;
  } else {
    cuda_flags |= CU_EVENT_DISABLE_TIMING;
  }
  if (flags != EventFlags::DEFAULT) {
    throw Exception("Unsupported event flags (" +
                    std::to_string(static_cast<int>(flags)) + ").");
  }
  CUevent event = nullptr;
  LC0EX_CUDA_CHECK(cuEventCreate(&event, cuda_flags));
  event_ = reinterpret_cast<Event>(event);
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
  if (this != &other) {
    if (event_ != nullptr) {
      LC0EX_CUDA_CHECK(cuEventDestroy(*this));
    }
    event_ = other.event_;
    other.event_ = nullptr;
  }
  return *this;
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    LC0EX_CUDA_CHECK(cuEventDestroy(*this));
  }
}

template <typename EventType>
CudaEvent::operator EventType() const {
  static_assert(std::is_same_v<EventType, CUevent>,
                "EventType must be CUevent");
  return reinterpret_cast<CUevent>(event_);
}

void CudaEvent::Synchronize() const {
  assert(event_ != nullptr);
  LC0EX_CUDA_CHECK(cuEventSynchronize(*this));
}

bool CudaEvent::IsCompleted() const {
  assert(event_ != nullptr);
  CUresult status = cuEventQuery(*this);
  if (status == CUDA_SUCCESS) return true;
  if (status == CUDA_ERROR_NOT_READY) return false;
  ThrowCuda(status, "cuEventQuery", __FILE__, __LINE__);
}

// Parse a buffer description from the lc0ex protobuf and create a BufferInfo
// structure that contains implicit stride information if Triton omits it.
BufferInfo MakeBufferInfo(const pblczero::Buffer& buffer) {
  absl::InlinedVector<BufferInfo::ShapeAndStride, 2> shape(
      buffer.shape().begin(), buffer.shape().end());
  if (shape.empty()) {
    throw Exception("Buffer shape is empty for " + std::string(buffer.name()) +
                    ".");
  }
  if (buffer.has_layout() && buffer.layout().strides_size() > 0) {
    auto& stride = buffer.layout().strides();
    if (stride.size() != shape.size()) {
      throw Exception(
          "Buffer layout strides size does not match shape size for " +
          std::string(buffer.name()) + ".");
    }
    std::transform(shape.begin(), shape.end(), stride.begin(), shape.begin(),
                   [](BufferInfo::ShapeAndStride s, std::int64_t stride) {
                     s.stride = stride;
                     return s;
                   });
  } else {
    std::transform(shape.rbegin(), shape.rend(), shape.rbegin(),
                   [stride = 1](BufferInfo::ShapeAndStride s) mutable {
                     s.stride = stride;
                     stride *= static_cast<std::int64_t>(s.shape);
                     return s;
                   });
  }
  size_t size_bytes =
      shape[0].shape * shape[0].stride * ElementSize(buffer.data_type());
  return {
      buffer.data_type(),
      std::move(shape),
      size_bytes,
      static_cast<ptrdiff_t>(buffer.offset()),
  };
}

// Aligned device memory allocation helper.
std::pair<CUdeviceptr, CUdeviceptr> AllocateDeviceMemory(
    std::uint64_t size_bytes, std::uint64_t alignment_bytes) {
  const auto allocation_size = size_bytes + alignment_bytes - 1;

  CUdeviceptr base = 0;
  LC0EX_CUDA_CHECK(
      cuMemAlloc(&base, static_cast<std::size_t>(allocation_size)));

  const auto base_address = static_cast<std::uint64_t>(base);
  const auto aligned_address =
      (base_address + alignment_bytes - 1) & ~(alignment_bytes - 1);
  return {base, static_cast<CUdeviceptr>(aligned_address)};
}

GridFormula& GridFormula::operator=(std::string_view formula) {
  RecursiveParser parser(formula);
  unsigned idx = parser.Parse();
  if (idx >= parser.result_.size()) {
    throw Exception(
        "Invalid index returned from Parse(): " + std::string(formula) +
        " at " + std::to_string(parser.lexer_.pos_) + ": " +
        std::string(parser.value));
  }
  if (parser.token != Lexer::kEnd) {
    throw Exception(
        "Unexpected token after formula: " + std::string(parser.lexer_.input_) +
        " at " + std::to_string(parser.lexer_.pos_) + ": " +
        std::string(parser.value));
  }
  operators_ = std::move(parser.result_);
  return *this;
}

template <typename T>
long Operator::operator()(const OpVector& other, T& state) const {
  long left, right;
  switch (op_) {
    case OperatorType::kAdd:
    case OperatorType::kSubtract:
    case OperatorType::kMultiply:
    case OperatorType::kDivide:
      right = other[children_.idx2_](other, state);
    case OperatorType::kNegate:
    case OperatorType::kAbs:
      left = other[children_.idx1_](other, state);
      break;
    case OperatorType::kReadBatchSize:
      left = state.batch_size_;
      break;
    case OperatorType::kReadTotalLegalMoves:
      left = state.cs_.total_legal_moves_;
      break;
    case OperatorType::kNumber:
      left = value_;
      break;
  }
  switch (op_) {
    case OperatorType::kAdd:
      return left + right;
    case OperatorType::kSubtract:
      return left - right;
    case OperatorType::kMultiply:
      return left * right;
    case OperatorType::kDivide:
      if (right == 0) {
        throw Exception("Division by zero in formula.");
      }
      return left / right;
    case OperatorType::kNegate:
      return -left;
    case OperatorType::kAbs:
      return std::abs(left);
    case OperatorType::kReadBatchSize:
    case OperatorType::kReadTotalLegalMoves:
    case OperatorType::kNumber:
      return left;
  }
}

template <typename State>
long GridFormula::operator()(State& state) const {
  if (operators_.empty()) {
    throw Exception("GridFormula operators are empty.");
  }
  return operators_.back()(operators_, state);
}

bool GridFormula::RequiresModification() const {
  for (const auto& op : operators_) {
    if (op.RequiresModification()) {
      return true;
    }
  }
  return false;
}

// The KernelNode class implementation. It represents a kernel node in the
// lc0ex graph and provides methods to launch the kernel with the appropriate
// arguments and configuration.
KernelNode::KernelNode(const pblczero::Node& source, CudaExecutable& executable,
                       const pblczero::Kernel& kernel, void* function)
    : NodeBase(source),
      function_(function),
      dynamic_shared_memory_bytes_(source.dynamic_shared_memory_bytes()) {
  ExecutableImpl& impl = static_cast<ExecutableImpl&>(executable);
  auto shared_memory = source.dynamic_shared_memory_bytes();
  if (shared_memory != 0) {
    LC0EX_CUDA_CHECK(cuFuncSetAttribute(
        reinterpret_cast<CUfunction>(function_),
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_memory));
  }
  if (source.grid_size() != 3) {
    throw Exception("Kernel node grid size is not 3.");
  }
  if (source.block_size() != 3) {
    throw Exception("Kernel node block size is not 3.");
  }
  std::copy(source.grid().begin(), source.grid().end(), grid_.begin());
  std::copy(source.block().begin(), source.block().end(), block_.begin());
  arguments_.reserve(kernel.parameters_size());
  auto argument_iter = source.arguments().begin();
  for (const auto& type : kernel.parameters()) {
    switch (type) {
      case pblczero::ParameterType_PARAMETER_TYPE_POINTER:
      case pblczero::ParameterType_PARAMETER_TYPE_U32: {
        if (argument_iter == source.arguments().end()) {
          throw Exception(
              "Kernel parameters size does not match node arguments size.");
        }
        auto& argument = *argument_iter++;
        if (argument.has_symbol()) {
          CUdeviceptr symbol = 0;
          size_t symbol_size = 0;
          std::string name(argument.symbol().symbol_name());
          CUmodule module = impl.GetBinary(argument.symbol().binary_idx());
          LC0EX_CUDA_CHECK(cuModuleGetGlobal(&symbol, &symbol_size,
                                             reinterpret_cast<CUmodule>(module),
                                             name.c_str()));
          arguments_.emplace_back(std::in_place_type<ArgumentSymbol>,
                                  reinterpret_cast<void*>(symbol));
        } else if (argument.has_allocation()) {
          switch (argument.allocation().kind()) {
            case pblczero::Node::Argument::AllocationLocation::
                ALLOCATION_PERSISTENT:
              arguments_.emplace_back(
                  std::in_place_type<ArgumentPersistentBuffer>,
                  argument.allocation().offset(),
                  argument.allocation().streamed_size());
              break;
            case pblczero::Node::Argument::AllocationLocation::
                ALLOCATION_EXECUTION:
              arguments_.emplace_back(
                  std::in_place_type<ArgumentExecutionBuffer>,
                  argument.allocation().offset());
              break;
            case pblczero::Node::Argument::AllocationLocation::
                ALLOCATION_UNKNOWN:
              throw Exception("Kernel argument allocation kind is unknown.");
          }
        } else if (argument.has_parameter()) {
          arguments_.emplace_back(std::in_place_type<ArgumentParameter>,
                                  argument.parameter());
        } else {
          throw Exception("Kernel argument is not a symbol, allocation.");
        }

        break;
      }
      case pblczero::ParameterType_PARAMETER_TYPE_NULL_POINTER:
        arguments_.emplace_back(std::in_place_type<ArgumentNull>);
        break;
      case pblczero::ParameterType_PARAMETER_TYPE_UNKNOWN:
        throw Exception("Kernel argument type is unknown.");
        break;
    }
  }
}

template <typename State>
auto KernelNode::operator()(State& state) const {
  state.argval.resize(arguments_.size());
  state.argptr.resize(arguments_.size());
  for (size_t i = 0; i < arguments_.size(); ++i) {
    std::visit(
        [&state, i](auto&& arg) {
          return arg(state.argptr[i], state.argval[i], state);
        },
        arguments_[i]);
  }

  if constexpr (!State::is_cuda_capturing) {
    LC0EX_CUDA_CHECK(
        cuLaunchKernel(reinterpret_cast<CUfunction>(function_), grid_[0](state),
                       grid_[1](state), grid_[2](state), block_[0], block_[1],
                       block_[2], dynamic_shared_memory_bytes_,
                       state.cs_.stream_, state.argptr.data(), nullptr));

    return;
  } else {
    CUgraphNode node = nullptr;
    CUDA_KERNEL_NODE_PARAMS params{};
    params.func = reinterpret_cast<CUfunction>(function_);
    params.gridDimX = grid_[0](state);
    params.gridDimY = grid_[1](state);
    params.gridDimZ = grid_[2](state);
    params.blockDimX = block_[0];
    params.blockDimY = block_[1];
    params.blockDimZ = block_[2];
    params.sharedMemBytes = dynamic_shared_memory_bytes_;
    params.kernelParams = state.argptr.data();

    LC0EX_CUDA_CHECK(cuGraphAddKernelNode(&node, state.graph_,
                                          state.dependencies_.data(),
                                          state.dependencies_.size(), &params));

    int window_max_size = 0;
    LC0EX_CUDA_CHECK(cuDeviceGetAttribute(
        &window_max_size, CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE,
        0));
    if (window_max_size > 0) {
      std::tuple<size_t, size_t> buffer{0, 0};

      for (const auto& arg : arguments_) {
        std::visit(
            [&](auto&& arg) {
              auto b = arg.GetOffsetAndSize();
              if (std::get<1>(b) > std::get<1>(buffer)) {
                buffer = b;
              }
            },
            arg);
      }

      auto [offset, size] = buffer;

      if (size > 0) {
        CUlaunchAttributeValue policy{};
        policy.accessPolicyWindow.base_ptr =
            const_cast<void*>(reinterpret_cast<const void*>(
                state.persistent_memory_.Data() + offset));
        policy.accessPolicyWindow.num_bytes =
            std::min(size, static_cast<size_t>(window_max_size));
        policy.accessPolicyWindow.hitRatio = 1.0f;
        policy.accessPolicyWindow.hitProp =
            CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING;
        policy.accessPolicyWindow.missProp =
            CUaccessProperty::CU_ACCESS_PROPERTY_NORMAL;

        LC0EX_CUDA_CHECK(cuGraphKernelNodeSetAttribute(
            node, CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, &policy));
      }
    }

    CUlaunchAttributeValue priority{};
    priority.priority = priority_;
    LC0EX_CUDA_CHECK(cuGraphKernelNodeSetAttribute(
        node, CU_LAUNCH_ATTRIBUTE_PRIORITY, &priority));

    if (RequiresModification()) {
      state.graph_.modifications_.emplace_back(
          [node, cs = &state.cs_, batch_size = state.batch_size_,
           persistent = &state.persistent_memory_,
           this](const CudaGraphExec& exec) {
            absl::InlinedVector<void*, 8> argptr(arguments_.size());
            absl::InlinedVector<uint64_t, 8> argval(arguments_.size());
            struct ArgState {
              decltype(*cs)& cs_;
              decltype(*persistent)& persistent_memory_;
              size_t batch_size_;
            } state{*cs, *persistent, batch_size};
            for (size_t i = 0; i < arguments_.size(); ++i) {
              std::visit(
                  [&](auto&& arg) { return arg(argptr[i], argval[i], state); },
                  arguments_[i]);
            }
            CUDA_KERNEL_NODE_PARAMS params{};
            params.func = reinterpret_cast<CUfunction>(function_);
            params.gridDimX = grid_[0](state);
            params.gridDimY = grid_[1](state);
            params.gridDimZ = grid_[2](state);
            params.blockDimX = block_[0];
            params.blockDimY = block_[1];
            params.blockDimZ = block_[2];
            params.sharedMemBytes = dynamic_shared_memory_bytes_;
            params.kernelParams = argptr.data();

            LC0EX_CUDA_CHECK(
                cuGraphExecKernelNodeSetParams(exec, node, &params));
          });
    }
    return node;
  }
}

bool KernelNode::RequiresModification() const {
  for (const auto& arg : arguments_) {
    if (std::visit([](auto&& arg) { return arg.RequiresModification(); },
                   arg)) {
      return true;
    }
  }
  for (const auto& grid : grid_) {
    if (grid.RequiresModification()) {
      return true;
    }
  }
  return false;
}

// The EventRecordNode and EventWaitNode classes implementation. They represent
// event record and wait nodes in the lc0ex graph and provide methods to record
// and wait for events on the CUDA stream.
namespace {

// Input and output transfers as kernel nodes that read and write mapped pinned
// host memory. Copy-engine memcpy nodes are executed in submission order, so a
// graph's input upload queued behind the previous graph's output downloads
// cannot start before that graph finishes; that serialized consecutive batches
// even though compute ordering allows the next batch to start during the heads.
// Set LC0EX_KERNEL_COPY=0 to use memcpy nodes instead.
constexpr char kCopyKernelPtx[] = R"LC0EXPTX(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-36037853
// Cuda compilation tools, release 12.9, V12.9.86
// Based on NVVM 7.0.1
//

.version 8.8
.target sm_75
.address_size 64

	// .globl	lc0ex_copy_bytes

.visible .entry lc0ex_copy_bytes(
	.param .u64 lc0ex_copy_bytes_param_0,
	.param .u64 lc0ex_copy_bytes_param_1,
	.param .u64 lc0ex_copy_bytes_param_2
)
{
	.reg .pred 	%p<6>;
	.reg .b16 	%rs<2>;
	.reg .b32 	%r<13>;
	.reg .b64 	%rd<29>;


	ld.param.u64 	%rd15, [lc0ex_copy_bytes_param_0];
	ld.param.u64 	%rd16, [lc0ex_copy_bytes_param_1];
	ld.param.u64 	%rd13, [lc0ex_copy_bytes_param_2];
	cvta.to.global.u64 	%rd1, %rd15;
	cvta.to.global.u64 	%rd2, %rd16;
	mov.u32 	%r1, %nctaid.x;
	mov.u32 	%r2, %ntid.x;
	mul.wide.u32 	%rd3, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	mul.wide.u32 	%rd17, %r3, %r2;
	mov.u32 	%r4, %tid.x;
	cvt.u64.u32 	%rd18, %r4;
	add.s64 	%rd4, %rd17, %rd18;
	or.b64  	%rd19, %rd16, %rd15;
	and.b64  	%rd20, %rd19, 15;
	setp.ne.s64 	%p1, %rd20, 0;
	mov.u64 	%rd27, 0;
	@%p1 bra 	$L__BB0_5;

	shr.u64 	%rd5, %rd13, 4;
	setp.ge.u64 	%p2, %rd4, %rd5;
	@%p2 bra 	$L__BB0_4;

	mov.u64 	%rd26, %rd4;

$L__BB0_3:
	shl.b64 	%rd21, %rd26, 4;
	add.s64 	%rd22, %rd1, %rd21;
	add.s64 	%rd23, %rd2, %rd21;
	ld.global.nc.v4.u32 	{%r5, %r6, %r7, %r8}, [%rd23];
	st.global.v4.u32 	[%rd22], {%r5, %r6, %r7, %r8};
	add.s64 	%rd26, %rd26, %rd3;
	setp.lt.u64 	%p3, %rd26, %rd5;
	@%p3 bra 	$L__BB0_3;

$L__BB0_4:
	and.b64  	%rd27, %rd13, -16;

$L__BB0_5:
	add.s64 	%rd28, %rd27, %rd4;
	setp.ge.u64 	%p4, %rd28, %rd13;
	@%p4 bra 	$L__BB0_7;

$L__BB0_6:
	add.s64 	%rd24, %rd2, %rd28;
	ld.global.nc.u8 	%rs1, [%rd24];
	add.s64 	%rd25, %rd1, %rd28;
	st.global.u8 	[%rd25], %rs1;
	add.s64 	%rd28, %rd28, %rd3;
	setp.lt.u64 	%p5, %rd28, %rd13;
	@%p5 bra 	$L__BB0_6;

$L__BB0_7:
	ret;

}
	// .globl	lc0ex_ordering_wait
.visible .entry lc0ex_ordering_wait(
	.param .u64 lc0ex_ordering_wait_param_0,
	.param .u64 lc0ex_ordering_wait_param_1
)
{
	.reg .pred 	%p<5>;
	.reg .b32 	%r<3>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd2, [lc0ex_ordering_wait_param_0];
	ld.param.u64 	%rd3, [lc0ex_ordering_wait_param_1];
	mov.u32 	%r1, %tid.x;
	setp.ne.s32 	%p1, %r1, 0;
	@%p1 bra 	$L__BB1_4;

	// begin inline asm
	mov.u64 %rd4, %globaltimer;
	// end inline asm
	// begin inline asm
	ld.acquire.gpu.global.u64 %rd5, [%rd2];
	// end inline asm
	setp.ge.u64 	%p2, %rd5, %rd3;
	@%p2 bra 	$L__BB1_4;

$L__BB1_2:
	// begin inline asm
	mov.u64 %rd7, %globaltimer;
	// end inline asm
	sub.s64 	%rd8, %rd7, %rd4;
	setp.gt.u64 	%p3, %rd8, 20000000;
	@%p3 bra 	$L__BB1_4;

	mov.u32 	%r2, 200;
	// begin inline asm
	nanosleep.u32 %r2;
	// end inline asm
	// begin inline asm
	ld.acquire.gpu.global.u64 %rd9, [%rd2];
	// end inline asm
	setp.lt.u64 	%p4, %rd9, %rd3;
	@%p4 bra 	$L__BB1_2;

$L__BB1_4:
	ret;

}
	// .globl	lc0ex_ordering_signal
.visible .entry lc0ex_ordering_signal(
	.param .u64 lc0ex_ordering_signal_param_0,
	.param .u64 lc0ex_ordering_signal_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd1, [lc0ex_ordering_signal_param_0];
	ld.param.u64 	%rd2, [lc0ex_ordering_signal_param_1];
	mov.u32 	%r1, %tid.x;
	setp.ne.s32 	%p1, %r1, 0;
	@%p1 bra 	$L__BB2_2;

	cvta.to.global.u64 	%rd3, %rd1;
	membar.gl;
	atom.global.max.u64 	%rd4, [%rd3], %rd2;

$L__BB2_2:
	ret;

}

)LC0EXPTX";

bool KernelCopyEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("LC0EX_KERNEL_COPY");
    return value == nullptr || std::string_view(value) != "0";
  }();
  return enabled;
}

CUfunction RuntimeKernel(const char* name) {
  static std::mutex mutex;
  static std::vector<std::pair<CUcontext, CUmodule>> modules;
  CUcontext context = nullptr;
  LC0EX_CUDA_CHECK(cuCtxGetCurrent(&context));
  std::lock_guard<std::mutex> lock(mutex);
  CUmodule module = nullptr;
  for (const auto& [ctx, mod] : modules) {
    if (ctx == context) module = mod;
  }
  if (module == nullptr) {
    LC0EX_CUDA_CHECK(cuModuleLoadData(&module, kCopyKernelPtx));
    modules.emplace_back(context, module);
  }
  CUfunction function = nullptr;
  LC0EX_CUDA_CHECK(cuModuleGetFunction(&function, module, name));
  return function;
}

CUfunction CopyKernelFunction() {
  static thread_local CUcontext cached_context = nullptr;
  static thread_local CUfunction cached = nullptr;
  CUcontext context = nullptr;
  LC0EX_CUDA_CHECK(cuCtxGetCurrent(&context));
  if (context != cached_context) {
    cached = RuntimeKernel("lc0ex_copy_bytes");
    cached_context = context;
  }
  return cached;
}

// Compute ordering between executions (backend option `ordering`).
// "event" uses CUDA event record/wait nodes.
// Under WDDM a graph's event wait is only released when the recording graph
// finishes, so the next batch cannot start during the heads as intended.
// "ticket" enforces the same order on the device: every execution takes a
// ticket under the compute ordering lock, waits until the counter reaches the
// previous ticket and publishes its own ticket where the event was recorded.
// "none" drops the ordering: consecutive batches overlap on the GPU. That won
// +1.2 % on a card held at its board power limit, but lost ~10 % once the card
// was limited by core power instead. "event" is the default; the ticket wait
// gives up after 20 ms so it can never deadlock.
enum class OrderingMode { kEvent, kTicket, kNone };

struct OrderingCounter {
  const CudaEvent* key;
  CUdeviceptr counter;
  uint64_t next_ticket;
  OrderingMode mode;
};

OrderingCounter& GetOrderingCounter(const CudaEvent& key) {
  static std::mutex mutex;
  static std::vector<std::unique_ptr<OrderingCounter>> counters;
  std::lock_guard<std::mutex> lock(mutex);
  for (auto& c : counters) {
    if (c->key == &key) return *c;
  }
  CUdeviceptr counter = 0;
  LC0EX_CUDA_CHECK(cuMemAlloc(&counter, sizeof(uint64_t)));
  LC0EX_CUDA_CHECK(cuMemsetD8(counter, 0, sizeof(uint64_t)));
  counters.push_back(std::make_unique<OrderingCounter>(
      OrderingCounter{&key, counter, 0, OrderingMode::kNone}));
  return *counters.back();
}

OrderingMode GetOrderingMode(const CudaEvent& compute_ordering_event) {
  return GetOrderingCounter(compute_ordering_event).mode;
}

struct OrderingKernelArguments {
  CUdeviceptr counter;
  unsigned long long ticket;
  std::array<void*, 2> pointers{&counter, &ticket};
};

CUDA_KERNEL_NODE_PARAMS OrderingKernelParams(bool wait,
                                             OrderingKernelArguments& args) {
  static thread_local CUcontext cached_context = nullptr;
  static thread_local CUfunction wait_function = nullptr;
  static thread_local CUfunction signal_function = nullptr;
  CUcontext context = nullptr;
  LC0EX_CUDA_CHECK(cuCtxGetCurrent(&context));
  if (context != cached_context) {
    wait_function = RuntimeKernel("lc0ex_ordering_wait");
    signal_function = RuntimeKernel("lc0ex_ordering_signal");
    cached_context = context;
  }
  CUDA_KERNEL_NODE_PARAMS params{};
  params.func = wait ? wait_function : signal_function;
  params.gridDimX = params.gridDimY = params.gridDimZ = 1;
  params.blockDimX = 32;
  params.blockDimY = params.blockDimZ = 1;
  params.kernelParams = args.pointers.data();
  return params;
}

unsigned long long OrderingTicketArgument(bool wait, uint64_t ticket) {
  return wait ? (ticket > 0 ? ticket - 1 : 0) : ticket;
}

// Adds (or launches) the ordering kernel. `wait` waits for the previous
// execution's ticket, otherwise publishes this execution's ticket.
template <typename State>
auto OrderingKernel(State& state, bool wait) {
  OrderingCounter& counter = GetOrderingCounter(state.compute_ordering_event_);
  OrderingKernelArguments args{
      counter.counter, OrderingTicketArgument(wait, state.cs_.ordering_ticket_)};
  const CUDA_KERNEL_NODE_PARAMS params = OrderingKernelParams(wait, args);
  if constexpr (!State::is_cuda_capturing) {
    LC0EX_CUDA_CHECK(cuLaunchKernel(params.func, 1, 1, 1, 32, 1, 1, 0,
                                    state.cs_.stream_, params.kernelParams,
                                    nullptr));
    return;
  } else {
    CUgraphNode node = nullptr;
    LC0EX_CUDA_CHECK(cuGraphAddKernelNode(&node, state.graph_,
                                          state.dependencies_.data(),
                                          state.dependencies_.size(), &params));
    state.graph_.modifications_.emplace_back(
        [node, cs = &state.cs_, counter_ptr = counter.counter,
         wait](const CudaGraphExec& exec) {
          OrderingKernelArguments args{
              counter_ptr, OrderingTicketArgument(wait, cs->ordering_ticket_)};
          const CUDA_KERNEL_NODE_PARAMS params =
              OrderingKernelParams(wait, args);
          LC0EX_CUDA_CHECK(cuGraphExecKernelNodeSetParams(exec, node, &params));
        });
    return node;
  }
}

// Returns the device address of pinned host memory, or 0 if it is not mapped.
CUdeviceptr MappedHostPointer(void* host_ptr) {
  CUdeviceptr device_ptr = 0;
  if (cuPointerGetAttribute(&device_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                            reinterpret_cast<CUdeviceptr>(host_ptr)) !=
      CUDA_SUCCESS) {
    return 0;
  }
  return device_ptr;
}

struct CopyKernelArguments {
  CUdeviceptr dst;
  CUdeviceptr src;
  unsigned long long bytes;
  std::array<void*, 3> pointers{&dst, &src, &bytes};
};

CUDA_KERNEL_NODE_PARAMS CopyKernelParams(CopyKernelArguments& arguments) {
  CUDA_KERNEL_NODE_PARAMS params{};
  params.func = CopyKernelFunction();
  params.gridDimX = 16;
  params.gridDimY = 1;
  params.gridDimZ = 1;
  params.blockDimX = 256;
  params.blockDimY = 1;
  params.blockDimZ = 1;
  params.kernelParams = arguments.pointers.data();
  return params;
}

}  // namespace

template <RecordEventType event>
EventRecordNode<event>::EventRecordNode(const pblczero::Node& source)
    : NodeBase(source) {
  if (!source.has_record_event()) {
    throw Exception("Event record node does not have record event.");
  }
}

template <RecordEventType event>
template <typename State>
auto EventRecordNode<event>::operator()(State& state) const {
  CudaEvent* e;
  switch (event) {
    case RecordEventType::kSleep:
      e = &state.cs_.sleep_event_;
      break;
    case RecordEventType::kComputeOrdering:
      e = &state.compute_ordering_event_;
      break;
    case RecordEventType::kWdlDownloadDone:
      e = &state.cs_.wdl_download_done_;
      break;
    case RecordEventType::kMlhDownloadDone:
      e = &state.cs_.mlh_download_done_;
      break;
    case RecordEventType::kPolicyDownloadDone:
      e = &state.cs_.policy_download_done_;
      break;
  }
  const OrderingMode ordering =
      GetOrderingMode(state.compute_ordering_event_);
  if (event == RecordEventType::kComputeOrdering &&
      ordering != OrderingMode::kEvent) {
    if (ordering == OrderingMode::kTicket) {
      return OrderingKernel(state, /*wait=*/false);
    }
    if constexpr (State::is_cuda_capturing) {
      CUgraphNode node = nullptr;
      LC0EX_CUDA_CHECK(cuGraphAddEmptyNode(&node, state.graph_,
                                           state.dependencies_.data(),
                                           state.dependencies_.size()));
      return node;
    } else {
      return;
    }
  }
  if constexpr (!State::is_cuda_capturing) {
    state.cs_.stream_.RecordEvent(*e);
    return;
  } else {
    CUgraphNode node = nullptr;
    LC0EX_CUDA_CHECK(cuGraphAddEventRecordNode(&node, state.graph_,
                                               state.dependencies_.data(),
                                               state.dependencies_.size(), *e));
    return node;
  }
}

template <WaitEventType event>
EventWaitNode<event>::EventWaitNode(const pblczero::Node& source)
    : NodeBase(source) {
  if (!source.has_wait_event()) {
    throw Exception("Event wait node does not have wait event.");
  }
}

template <WaitEventType event>
template <typename State>
auto EventWaitNode<event>::operator()(State& state) const {
  CudaEvent* e;
  switch (event) {
    case WaitEventType::kComputeOrdering:
      e = &state.compute_ordering_event_;
      break;
  }
  const OrderingMode ordering =
      GetOrderingMode(state.compute_ordering_event_);
  if (event == WaitEventType::kComputeOrdering &&
      ordering != OrderingMode::kEvent) {
    if (ordering == OrderingMode::kTicket) {
      return OrderingKernel(state, /*wait=*/true);
    }
    if constexpr (State::is_cuda_capturing) {
      CUgraphNode node = nullptr;
      LC0EX_CUDA_CHECK(cuGraphAddEmptyNode(&node, state.graph_,
                                           state.dependencies_.data(),
                                           state.dependencies_.size()));
      return node;
    } else {
      return;
    }
  }
  if constexpr (!State::is_cuda_capturing) {
    state.cs_.stream_.WaitEvent(*e);
    return;
  } else {
    CUgraphNode node = nullptr;
    LC0EX_CUDA_CHECK(cuGraphAddEventWaitNode(&node, state.graph_,
                                             state.dependencies_.data(),
                                             state.dependencies_.size(), *e));
    return node;
  }
}

// The MemcpyNode class implementation. It represents a memcpy node in the lc0ex
// graph and provides methods to perform host-to-device or device-to-host memory
// copies.
template <MemcpyBuffer kind>
MemcpyNode<kind>::MemcpyNode(const pblczero::Node& source)
    : NodeBase(source), gpu_offset_{source.memcpy().gpu_offset()} {
  if (!source.has_memcpy()) {
    throw Exception("Memcpy node does not have memcpy.");
  }
}

template <MemcpyBuffer kind>
template <typename State>
auto MemcpyNode<kind>::operator()(State& state) const {
  using ComputeType = typename State::ComputeType;
  bool is_h2d = false;
  size_t bytes = 0;
  void* host_ptr = nullptr;
  CUdeviceptr device_ptr = reinterpret_cast<CUdeviceptr>(
      state.cs_.device_memory_.Data() + gpu_offset_);

  constexpr size_t kWdlSize = 3;
  constexpr size_t kMlhSize = 1;
  switch (kind) {
    case MemcpyBuffer::kInputMapping:
      host_ptr = state.cs_.input_mapping.data();
      is_h2d = true;
      bytes = state.cs_.total_legal_moves_ * sizeof(uint32_t);
      break;
    case MemcpyBuffer::kInputMask:
      host_ptr = state.cs_.input_mask.data();
      is_h2d = true;
      bytes = state.batch_size_ * sizeof(uint64_t) * kInputPlanes;
      break;
    case MemcpyBuffer::kInputValue:
      host_ptr = state.cs_.input_value.data();
      is_h2d = true;
      bytes = state.batch_size_ * sizeof(ComputeType) * kInputPlanes;
      break;
    case MemcpyBuffer::kOutputsPolicy:
      host_ptr = state.cs_.output_policy.data();
      is_h2d = false;
      bytes = state.cs_.total_legal_moves_ * sizeof(ComputeType);
      break;
    case MemcpyBuffer::kOutputsWdl:
      host_ptr = state.cs_.output_wdl.data();
      is_h2d = false;
      bytes = state.batch_size_ * sizeof(ComputeType) * kWdlSize;
      break;
    case MemcpyBuffer::kOutputsMlh:
      host_ptr = state.cs_.output_mlh.data();
      is_h2d = false;
      bytes = state.batch_size_ * sizeof(ComputeType) * kMlhSize;
      break;
  }

  CudaBuffer<ComputeType> device_buffer =
      state.cs_.device_memory_.template AsSpan<ComputeType>(gpu_offset_, bytes);
  if constexpr (!State::is_cuda_capturing) {
    if (is_h2d) {
      LC0EX_CUDA_CHECK(
          cuMemcpyHtoDAsync(device_ptr, host_ptr, bytes, state.cs_.stream_));
    } else {
      LC0EX_CUDA_CHECK(
          cuMemcpyDtoHAsync(host_ptr, device_ptr, bytes, state.cs_.stream_));
    }
    return;
  } else {
    CUgraphNode node = nullptr;
    const CUdeviceptr mapped_host =
        KernelCopyEnabled() ? MappedHostPointer(host_ptr) : 0;
    if (mapped_host != 0) {
      CopyKernelArguments arguments{is_h2d ? device_ptr : mapped_host,
                                    is_h2d ? mapped_host : device_ptr, bytes};
      const CUDA_KERNEL_NODE_PARAMS params = CopyKernelParams(arguments);
      LC0EX_CUDA_CHECK(cuGraphAddKernelNode(&node, state.graph_,
                                            state.dependencies_.data(),
                                            state.dependencies_.size(),
                                            &params));
      if (kind == MemcpyBuffer::kInputMapping ||
          kind == MemcpyBuffer::kOutputsPolicy) {
        state.graph_.modifications_.emplace_back(
            [node, cs = &state.cs_, this](const CudaGraphExec& exec) {
              const CUdeviceptr device = reinterpret_cast<CUdeviceptr>(
                  cs->device_memory_.Data() + gpu_offset_);
              CopyKernelArguments arguments{};
              if (kind == MemcpyBuffer::kInputMapping) {
                arguments.dst = device;
                arguments.src = MappedHostPointer(cs->input_mapping.data());
                arguments.bytes = cs->total_legal_moves_ * sizeof(uint32_t);
              } else {
                arguments.dst = MappedHostPointer(cs->output_policy.data());
                arguments.src = device;
                arguments.bytes = cs->total_legal_moves_ * sizeof(ComputeType);
              }
              const CUDA_KERNEL_NODE_PARAMS params = CopyKernelParams(arguments);
              LC0EX_CUDA_CHECK(
                  cuGraphExecKernelNodeSetParams(exec, node, &params));
            });
      }
      return node;
    }
    CUDA_MEMCPY3D copy_params{};
    copy_params.srcMemoryType =
        is_h2d ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
    copy_params.srcHost = is_h2d ? host_ptr : nullptr;
    copy_params.srcDevice = is_h2d ? 0 : device_ptr;
    copy_params.dstMemoryType =
        is_h2d ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
    copy_params.dstHost = is_h2d ? 0 : host_ptr;
    copy_params.dstDevice = is_h2d ? device_ptr : 0;
    copy_params.WidthInBytes = bytes;
    copy_params.Height = 1;
    copy_params.Depth = 1;
    LC0EX_CUDA_CHECK(cuGraphAddMemcpyNode(
        &node, state.graph_, state.dependencies_.data(),
        state.dependencies_.size(), &copy_params, nullptr));
    if (kind == MemcpyBuffer::kInputMapping) {
      state.graph_.modifications_.emplace_back(
          [node, cs = &state.cs_, this](const CudaGraphExec& exec) {
            size_t total_legal_moves = cs->total_legal_moves_;
            CUDA_MEMCPY3D copy_params{};
            copy_params.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy_params.srcHost = cs->input_mapping.data();
            copy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy_params.dstDevice = reinterpret_cast<CUdeviceptr>(
                cs->device_memory_.Data() + gpu_offset_);
            copy_params.WidthInBytes = total_legal_moves * sizeof(uint32_t);
            copy_params.Height = 1;
            copy_params.Depth = 1;
            LC0EX_CUDA_CHECK(cuGraphExecMemcpyNodeSetParams(
                exec, node, &copy_params, nullptr));
          });
    } else if (kind == MemcpyBuffer::kOutputsPolicy) {
      state.graph_.modifications_.emplace_back(
          [node, cs = &state.cs_, this](const CudaGraphExec& exec) {
            size_t total_legal_moves = cs->total_legal_moves_;
            CUDA_MEMCPY3D copy_params{};
            copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy_params.srcDevice = reinterpret_cast<CUdeviceptr>(
                cs->device_memory_.Data() + gpu_offset_);
            copy_params.dstMemoryType = CU_MEMORYTYPE_HOST;
            copy_params.dstHost = cs->output_policy.data();
            copy_params.WidthInBytes = total_legal_moves * sizeof(ComputeType);
            copy_params.Height = 1;
            copy_params.Depth = 1;
            LC0EX_CUDA_CHECK(cuGraphExecMemcpyNodeSetParams(
                exec, node, &copy_params, nullptr));
          });
    }
    return node;
  }
}

// The CudaProgram class implementation. It represents a program in the lc0ex
// executable and provides methods to launch the graph.
CudaProgram::CudaProgram(const pblczero::Program& source,
                         CudaExecutable& executable,
                         const std::vector<pblczero::Kernel>& kernels) {
  LCTRACE_FUNCTION_SCOPE;
  using NodeVector = decltype(nodes_);
  pblczero::ProgramMetadata metadata;
  metadata.ParseFromString(source.metadata());
  batch_size_ = metadata.batch_size();
  ExecutableImpl& impl = static_cast<ExecutableImpl&>(executable);

  if (source.has_execution_allocation()) {
    impl.BuildExecutionAllocation(source);
  }

  nodes_.reserve(source.nodes_size());
  size_t idx = 0;
  for (const auto& node : source.nodes()) {
    size_t i = idx++;
    if (node.has_kernel_idx()) {
      if (node.kernel_idx() >= kernels.size()) {
        throw Exception("Node (" + std::to_string(i) +
                        ") kernel index is out of range in " +
                        std::string(source.name()) + ".");
      }
      auto& kernel = kernels[node.kernel_idx()];
      CUmodule module = impl.GetBinary(kernel.binary_idx());
      std::string kernel_name(kernel.function());
      CUfunction function = nullptr;
      LC0EX_CUDA_CHECK(
          cuModuleGetFunction(&function, module, kernel_name.c_str()));
      if (node.grid_size() != 3) {
        throw Exception("Node (" + std::to_string(i) +
                        ") grid size is not 3 in " +
                        std::string(source.name()) + ".");
      }
      if (node.block_size() != 3) {
        throw Exception("Node (" + std::to_string(i) +
                        ") block size is not 3 in " +
                        std::string(source.name()) + ".");
      }
      if (!node.has_dynamic_shared_memory_bytes()) {
        throw Exception("Node (" + std::to_string(i) +
                        ") dynamic shared memory bytes is not set in " +
                        std::string(source.name()) + ".");
      }
      nodes_.emplace_back(std::in_place_type<KernelNode>, node, executable,
                          kernel, reinterpret_cast<void*>(function));
      std::get<KernelNode>(nodes_.back()).SetDebugState(node.kernel_idx(), i);
    } else if (node.has_record_event()) {
      static const std::map<
          std::string_view,
          std::function<void(NodeVector&, const pblczero::Node&)>>
          event_node_creators = {
              {"/event/sleep",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kSleep>>,
                     node);
               }},
              {"/event/compute_ordering",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kComputeOrdering>>,
                     node);
               }},
              {"/event/wdl_done",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kWdlDownloadDone>>,
                     node);
               }},
              {"/event/mlh_done",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kMlhDownloadDone>>,
                     node);
               }},
              {"/event/policy_done",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kPolicyDownloadDone>>,
                     node);
               }}};
      auto it = event_node_creators.find(node.record_event());
      if (it == event_node_creators.end()) {
        throw Exception("Unknown record event name (" +
                        std::string(node.record_event()) + ") in node (" +
                        std::to_string(i) + ") in " +
                        std::string(source.name()) + ".");
      }
      it->second(nodes_, node);
    } else if (node.has_wait_event()) {
      static const std::map<
          std::string_view,
          std::function<void(NodeVector&, const pblczero::Node&)>>
          wait_node_creators = {
              {"/event/compute_ordering",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventWaitNode<WaitEventType::kComputeOrdering>>,
                     node);
               }}};
      auto it = wait_node_creators.find(node.wait_event());
      if (it == wait_node_creators.end()) {
        throw Exception("Unknown wait event name (" +
                        std::string(node.wait_event()) + ") in node (" +
                        std::to_string(i) + ") in " +
                        std::string(source.name()) + ".");
      }
      it->second(nodes_, node);
    } else if (node.has_memcpy()) {
      static const std::map<
          std::string_view,
          std::function<void(NodeVector&, const pblczero::Node&)>>
          memcpy_node_creators = {
              {"/input/policy_mapping",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         MemcpyNode<MemcpyBuffer::kInputMapping>>,
                     node);
               }},
              {"/input/plane_masks",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<MemcpyNode<MemcpyBuffer::kInputMask>>,
                     node);
               }},
              {"/input/plane_values",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<MemcpyNode<MemcpyBuffer::kInputValue>>,
                     node);
               }},
              {"/output/policy",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         MemcpyNode<MemcpyBuffer::kOutputsPolicy>>,
                     node);
               }},
              {"/output/wdl",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<MemcpyNode<MemcpyBuffer::kOutputsWdl>>,
                     node);
               }},
              {"/output/mlh",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<MemcpyNode<MemcpyBuffer::kOutputsMlh>>,
                     node);
               }}};
      auto it = memcpy_node_creators.find(node.memcpy().name());
      if (it == memcpy_node_creators.end()) {
        throw Exception("Unknown memcpy buffer name (" +
                        std::string(node.memcpy().name()) + ") in node (" +
                        std::to_string(i) + ") in " +
                        std::string(source.name()) + ".");
      }
      it->second(nodes_, node);
    } else {
      throw Exception("Unknown node type (" + std::to_string(i) + ") in " +
                      std::string(source.name()) + ".");
    }
  }
}

template <typename State>
void CudaProgram::Run(State& state) const {
  for (const auto& node : nodes_) {
    std::visit([&](auto&& arg) { arg(state); }, node);
  }
}

GraphExecModification::GraphExecModification(
    GraphExecModification&& other) noexcept
    : modify_func_(std::move(other.modify_func_)) {}

GraphExecModification& GraphExecModification::operator=(
    GraphExecModification&& other) noexcept {
  if (this != &other) {
    modify_func_ = std::move(other.modify_func_);
  }
  return *this;
}

void GraphExecModification::operator()(const CudaGraphExec& exec) const {
  modify_func_(exec);
}

GraphCapture::GraphCapture() {
  CUgraph graph = nullptr;
  LC0EX_CUDA_CHECK(cuGraphCreate(&graph, 0));
  graph_ = reinterpret_cast<Graph>(graph);
}

GraphCapture::GraphCapture(GraphCapture&& other) noexcept
    : modifications_(std::move(other.modifications_)), graph_(other.graph_) {
  other.graph_ = nullptr;
}

GraphCapture& GraphCapture::operator=(GraphCapture&& other) noexcept {
  if (this != &other) {
    if (graph_) {
      LC0EX_CUDA_CHECK(cuGraphDestroy(*this));
    }
    graph_ = other.graph_;
    other.graph_ = nullptr;
  }
  return *this;
}

GraphCapture::~GraphCapture() {
  if (graph_) {
    LC0EX_CUDA_CHECK(cuGraphDestroy(*this));
  }
}

template <typename GraphType>
  requires std::is_pointer_v<GraphType>
GraphCapture::operator GraphType() const {
  static_assert(std::is_same_v<GraphType, CUgraph>,
                "GraphType must be CUgraph");
  return reinterpret_cast<GraphType>(graph_);
}

template <typename State>
GraphCapture CudaProgram::Capture(GraphMode mode, State& state) const {
  state.graph_nodes_.reserve(nodes_.size());
  for (const auto& node : nodes_) {
    state.dependencies_.clear();
    if (mode == GraphMode::kDag) {
      auto deps =
          std::visit([&](auto&& arg) { return arg.GetDependecies(); }, node);
      for (auto dep : deps) {
        state.dependencies_.push_back(state.graph_nodes_[dep]);
      }
    } else {
      state.dependencies_.push_back(state.graph_nodes_.back());
    }
    CUgraphNode graph_node =
        std::visit([&](auto&& arg) { return arg(state); }, node);
    state.graph_nodes_.push_back(graph_node);
  }
  GraphCapture rv{std::move(state.graph_)};
  return rv;
}

namespace {

[[maybe_unused]]
inline CUdevice& ToDevice(Device& device) {
  return static_cast<CUdevice&>(device);
}
[[maybe_unused]]
inline const CUdevice& ToDevice(const Device& device) {
  return static_cast<const CUdevice&>(device);
}
[[maybe_unused]]
inline CUcontext& ToContext(Context& context) {
  return reinterpret_cast<CUcontext&>(context);
}
[[maybe_unused]]
const CUcontext& ToContext(const Context& context) {
  return reinterpret_cast<const CUcontext&>(context);
}

}  // namespace

// The CudaMemory class implementation. It represents a block of device memory
// and provides methods to manage its lifecycle and operations.
CudaMemory::CudaMemory(size_t size, size_t alignment)
    : MemoryBase(size, alignment) {
  assert(size > 0);
  assert(alignment > 0);
  assert(std::has_single_bit(alignment));
  std::tie(reinterpret_cast<CUdeviceptr&>(device_ptr_),
           reinterpret_cast<CUdeviceptr&>(aligned_ptr_)) =
      AllocateDeviceMemory(size, alignment);
}

CudaMemory::CudaMemory(CudaMemory&& other) noexcept
    : MemoryBase(std::move(other)), device_ptr_(other.device_ptr_) {
  other.device_ptr_ = 0;
}

CudaMemory& CudaMemory::operator=(CudaMemory&& other) noexcept {
  if (this != &other) {
    if (device_ptr_ != 0) {
      LC0EX_CUDA_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(device_ptr_)));
    }
    MemoryBase::operator=(std::move(other));
    device_ptr_ = other.device_ptr_;
    other.device_ptr_ = 0;
  }
  return *this;
}

CudaMemory::~CudaMemory() {
  if (device_ptr_ != 0) {
    LC0EX_CUDA_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(device_ptr_)));
  }
}

// The CudaHostMemory class implementation. It represents a block of pinned host
// memory and provides methods to manage its lifecycle and operations.
CudaHostMemory::CudaHostMemory(size_t size, size_t alignment)
    : MemoryBase(size, alignment) {
  assert(size > 0);
  assert(alignment > 0);
  assert(std::has_single_bit(alignment));
  assert(size % alignment == 0);
  void* host_ptr = nullptr;
  size = (size + alignment - 1) & ~(alignment - 1);
  LC0EX_CUDA_CHECK(cuMemHostAlloc(&host_ptr, size,
                                   CU_MEMHOSTALLOC_PORTABLE |
                                       CU_MEMHOSTALLOC_DEVICEMAP));
  const auto base_address = reinterpret_cast<uintptr_t>(host_ptr);
  const auto aligned_address =
      (base_address + alignment - 1) & ~(alignment - 1);
  host_ptr_ = host_ptr;
  aligned_ptr_ = reinterpret_cast<char*>(aligned_address);
}

CudaHostMemory::CudaHostMemory(CudaHostMemory&& other) noexcept
    : MemoryBase(std::move(other)), host_ptr_(other.host_ptr_) {
  other.host_ptr_ = nullptr;
}

CudaHostMemory::~CudaHostMemory() {
  if (host_ptr_ != nullptr) {
    LC0EX_CUDA_CHECK(cuMemFreeHost(host_ptr_));
  }
}

// The CudaGraphExec class implementation. It represents a CUDA graph
// execution object and provides methods to manage its lifecycle and
// operations.
CudaGraphExec::CudaGraphExec(GraphCapture& graph)
    : graph_(graph.release()), modifications_(std::move(graph.modifications_)) {
  CUgraphExec graph_exec = nullptr;
  LC0EX_CUDA_CHECK(
      cuGraphInstantiate(&graph_exec, reinterpret_cast<CUgraph>(graph_),
                         CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY));
  graph_exec_ = reinterpret_cast<GraphExec>(graph_exec);
}

void CudaGraphExec::Upload(CudaStream& stream) const {
  LC0EX_CUDA_CHECK(
      cuGraphUpload(reinterpret_cast<CUgraphExec>(graph_exec_), stream));
}

CudaGraphExec::CudaGraphExec(CudaGraphExec&& other) noexcept
    : graph_exec_(other.graph_exec_),
      graph_(other.graph_),
      modifications_(std::move(other.modifications_)) {
  other.graph_exec_ = nullptr;
  other.graph_ = nullptr;
}

CudaGraphExec& CudaGraphExec::operator=(CudaGraphExec&& other) noexcept {
  if (this != &other) {
    if (graph_exec_) {
      LC0EX_CUDA_CHECK(cuGraphExecDestroy(*this));
    }
    if (graph_) {
      LC0EX_CUDA_CHECK(cuGraphDestroy(reinterpret_cast<CUgraph>(graph_)));
    }
    graph_ = other.graph_;
    modifications_ = std::move(other.modifications_);
    graph_exec_ = other.graph_exec_;
    other.graph_exec_ = nullptr;
    other.graph_ = nullptr;
  }
  return *this;
}

CudaGraphExec::~CudaGraphExec() {
  if (graph_exec_) {
    LC0EX_CUDA_CHECK(cuGraphExecDestroy(*this));
  }
  if (graph_) {
    LC0EX_CUDA_CHECK(cuGraphDestroy(reinterpret_cast<CUgraph>(graph_)));
  }
}

template <typename GraphExecType>
CudaGraphExec::operator GraphExecType() const {
  return reinterpret_cast<GraphExecType>(graph_exec_);
}

uint64_t NextOrderingTicket(const CudaEvent& compute_ordering_event) {
  OrderingCounter& counter = GetOrderingCounter(compute_ordering_event);
  if (counter.mode != OrderingMode::kTicket) return 0;
  return ++counter.next_ticket;
}

void SetComputeOrdering(const CudaEvent& compute_ordering_event,
                        std::string_view mode) {
  OrderingCounter& counter = GetOrderingCounter(compute_ordering_event);
  if (mode == "event") {
    counter.mode = OrderingMode::kEvent;
  } else if (mode == "ticket") {
    counter.mode = OrderingMode::kTicket;
  } else if (mode == "none") {
    counter.mode = OrderingMode::kNone;
  } else {
    throw Exception("Unknown lc0ex ordering mode: " + std::string(mode));
  }
}

void CudaGraphExec::Launch(CudaStream& stream) const {
  for (const auto& mod : modifications_) {
    mod(*this);
  }
  LC0EX_CUDA_CHECK(
      cuGraphLaunch(reinterpret_cast<CUgraphExec>(graph_exec_), stream));
}

// The CudaRuntime class implementation. It represents a CUDA  context which
// uses primary context to allow interoperability with other CUDA APIs and
// libraries.
CudaRuntime::CudaRuntime(int device_ordinal) {
  LCTRACE_FUNCTION_SCOPE;
  LC0EX_CUDA_CHECK(cuInit(0));
  LC0EX_CUDA_CHECK(cuDeviceGet(&ToDevice(device_), device_ordinal));
  LC0EX_CUDA_CHECK(cuDevicePrimaryCtxRetain(&ToContext(context_), device_));
  SetCurrent();
}

CudaRuntime::~CudaRuntime() {
  if (context_) LC0EX_CUDA_CHECK(cuDevicePrimaryCtxRelease(ToDevice(device_)));
}

void CudaRuntime::SetCurrent() const {
  LC0EX_CUDA_CHECK(cuCtxSetCurrent(ToContext(context_)));
}

// The CudaExecutable class implementation. It implements and NeuralExecutable
// which can have many programs using shared kernels and buffers. It provides
// methods to run or capture graphs.
CudaExecutable::CudaExecutable(const pblczero::NeuralExecutable& source) {
  if (source.magic() != kMagic) throw Exception("Invalid lc0ex magic.");
  if (source.format() != kFormat) {
    throw Exception("Unsupported lc0ex format generation.");
  }

  ExecutableImpl* impl = static_cast<ExecutableImpl*>(this);

  impl->BuildModules(source);
  impl->BuildPersistentAllocation(source);
  impl->BuildExecutionAllocation(source);
  impl->BuildPrograms(source);
}

size_t CudaExecutable::GetPersistentAllocationSize() const {
  return persistent_allocation_.Size();
}

size_t CudaExecutable::GetExecutionAllocationSize() const {
  return execution_allocation_.size_bytes_;
}

size_t CudaExecutable::GetExecutionAllocationAlignment() const {
  return execution_allocation_.alignment_bytes_;
}

size_t CudaExecutable::GetMaxBatchSize() const {
  return programs_.back().Size();
}

void CudaExecutable::CopyPersistentFromHost(
    std::span<const std::byte> source,
    std::optional<std::size_t> size_bytes) const {
  const std::size_t copy_size = size_bytes.value_or(source.size());
  if (copy_size == 0 || !persistent_allocation_) return;

  LC0EX_CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)persistent_allocation_.Data(),
                                source.data(), copy_size));
}

template <typename T>
void CudaExecutable::Run(size_t batch_size,
                         ComputationState<CudaRuntime, T>& cs,
                         CudaEvent& compute_ordering_event) {
  auto program =
      std::lower_bound(programs_.begin(), programs_.end(), batch_size,
                       [](const CudaProgram& program, size_t batch_size) {
                         return program.Size() < batch_size;
                       });
  if (program == programs_.end()) {
    throw Exception("Batch size exceeds maximum supported by executable.");
  }
  LaunchState<T> state{
      cs, batch_size, persistent_allocation_, compute_ordering_event, {}, {}};
  program->Run(state);
}

template void CudaExecutable::Run<float>(
    size_t batch_size, ComputationState<CudaRuntime, float>& state,
    CudaEvent& compute_ordering_event);
template void CudaExecutable::Run<Half>(
    size_t batch_size, ComputationState<CudaRuntime, Half>& state,
    CudaEvent& compute_ordering_event);

template <typename T>
GraphCapture CudaExecutable::Capture(GraphMode mode, size_t batch_size,
                                     ComputationState<CudaRuntime, T>& cs,
                                     CudaEvent& compute_ordering_event) {
  auto program =
      std::lower_bound(programs_.begin(), programs_.end(), batch_size,
                       [](const CudaProgram& program, size_t batch_size) {
                         return program.Size() < batch_size;
                       });
  if (program == programs_.end()) {
    throw Exception("Batch size exceeds maximum supported by executable.");
  }
  CaptureState<T> state{
      {cs, batch_size, persistent_allocation_, compute_ordering_event, {}, {}},
      {},
      {},
      {}};
  return program->Capture(mode, state);
}

template GraphCapture CudaExecutable::Capture<float>(
    GraphMode mode, size_t batch_size,
    ComputationState<CudaRuntime, float>& state,
    CudaEvent& compute_ordering_event);
template GraphCapture CudaExecutable::Capture<Half>(
    GraphMode mode, size_t batch_size,
    ComputationState<CudaRuntime, Half>& state,
    CudaEvent& compute_ordering_event);

}  // namespace lczero::lc0ex::cuda
