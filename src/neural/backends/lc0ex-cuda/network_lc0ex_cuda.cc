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

#include <absl/container/inlined_vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <future>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "network_fingerprint.h"
#include "neural/backend.h"
#include "neural/backends/lc0ex-cuda/runtime/runtime.h"
#include "neural/encoder.h"
#include "neural/loader.h"
#include "neural/onnx/converter.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "neural/tables/attention_policy_map.h"
#include "proto/lc0ex.pb.h"
#include "runtime/lc0ex_cuda.h"
#include "utils/atomic_vector.h"
#include "utils/exception.h"
#include "utils/fastmath.h"
#include "utils/logging.h"
#include "utils/trace.h"

namespace lczero {
namespace {

constexpr std::string_view kBackendName = "lc0ex-cuda";
constexpr std::size_t kNumOutputPolicy = 218;
constexpr std::size_t kNumWdlOutputs = 3;

FillEmptyHistory ParseHistoryFill(const std::string& value) {
  if (value == "fen_only") return FillEmptyHistory::FEN_ONLY;
  if (value == "always") return FillEmptyHistory::ALWAYS;
  if (value == "no") return FillEmptyHistory::NO;
  throw Exception("Unknown history fill mode '" + value + "'.");
}

std::uint64_t DataTypeSize(pblczero::Buffer::DataType data_type) {
  switch (data_type) {
    case pblczero::Buffer::DATA_TYPE_F32:
      return sizeof(float);
    case pblczero::Buffer::DATA_TYPE_U8:
      return sizeof(std::uint8_t);
    case pblczero::Buffer::DATA_TYPE_F16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_U64:
      return sizeof(std::uint64_t);
    case pblczero::Buffer::DATA_TYPE_BF16:
      return sizeof(std::uint16_t);
    default:
      throw Exception("Unsupported or unknown lc0ex buffer data type.");
  }
}

pblczero::NeuralExecutable LoadExecutableFile(const std::string& path) {
  LCTRACE_FUNCTION_SCOPE;
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    throw Exception("Cannot read lc0ex executable from " + path + ".");
  }

  std::string serialized((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  if (file.bad()) {
    throw Exception("Error while reading lc0ex executable from " + path + ".");
  }

  pblczero::NeuralExecutable executable;
  executable.ParseFromString(serialized);
  return executable;
}

void CheckNetworkFingerprint(const WeightsFile& weights,
                             const pblczero::NeuralExecutable& executable) {
  LCTRACE_FUNCTION_SCOPE;
  pblczero::Net executable_fingerprint;
  executable_fingerprint.ParseFromString(executable.metadata());

  const auto network_fingerprint = lc0ex::BuildNetworkFingerprint(weights);
  if (network_fingerprint.OutputAsString() !=
      executable_fingerprint.OutputAsString()) {
    throw Exception(
        "The lc0ex executable was created for a different network "
        "architecture.");
  }
}

bool HasMatchingShape(const pblczero::Buffer& buffer,
                      const pblczero::TensorProto& initializer) {
  if (initializer.dims_size() != buffer.shape().size()) {
    return false;
  }
  for (std::size_t i = 0; i < initializer.dims_size(); ++i) {
    const auto dimension = initializer.dims(i);
    if (dimension < 0 ||
        static_cast<std::uint64_t>(dimension) != buffer.shape()[i]) {
      return false;
    }
  }
  return true;
}

void ValidateInitializer(const pblczero::Buffer& buffer,
                         const pblczero::TensorProto& initializer) {
  if (static_cast<int>(buffer.data_type()) !=
      static_cast<int>(initializer.data_type())) {
    throw Exception("Data type mismatch for lc0ex buffer '" +
                    std::string(buffer.name()) + "'.");
  }
  if (!HasMatchingShape(buffer, initializer)) {
    throw Exception("Shape mismatch for lc0ex buffer '" +
                    std::string(buffer.name()) + "'.");
  }
  size_t buffer_size =
      std::accumulate(buffer.shape().begin(), buffer.shape().end(), 1ull,
                      std::multiplies<std::uint64_t>()) *
      DataTypeSize(buffer.data_type());
  if (static_cast<std::uint64_t>(initializer.raw_data().size()) !=
      buffer_size) {
    throw Exception("Size mismatch for lc0ex buffer '" +
                    std::string(buffer.name()) + "'.");
  }
}

WeightsToOnnxConverterOptions MakeConverterOptions(
    const OptionsDict& backend_options) {
  WeightsToOnnxConverterOptions converter_options;
  converter_options.opset = backend_options.GetOrDefault<int>("opset", 17);
  converter_options.ir = backend_options.GetOrDefault<int>("ir", -1);
  converter_options.alt_mish =
      backend_options.GetOrDefault<bool>("alt_mish", false);
  converter_options.alt_layernorm =
      backend_options.GetOrDefault<bool>("alt_layernorm", false);
  converter_options.no_shape =
      backend_options.GetOrDefault<bool>("no_shape", false);
  converter_options.policy_head =
      backend_options.GetOrDefault<std::string>("policy_head", "vanilla");
  converter_options.value_head =
      backend_options.GetOrDefault<std::string>("value_head", "winner");
  converter_options.no_wdl_softmax = true;

  std::string datatype;
  if (backend_options.Exists<std::string>("datatype")) {
    datatype = backend_options.Get<std::string>("datatype");
  } else {
    const bool fp16 = backend_options.GetOrDefault<bool>("fp16", true);
    datatype = fp16 ? "f16" : "f32";
  }
  converter_options.data_type =
      WeightsToOnnxConverterOptions::StringToDataType(datatype);
  return converter_options;
}

void CopyStridedHostTensor(const std::vector<std::uint64_t>& shape,
                           const std::vector<std::int64_t>& dst_strides,
                           const std::vector<std::int64_t>& src_strides,
                           std::size_t elem_size, const std::byte* src,
                           std::byte* dst) {
  if (shape.empty()) {
    std::memcpy(dst, src, elem_size);
    return;
  }
  std::size_t contiguous_dim = shape.size();
  std::size_t contiguous_bytes = elem_size;
  while (contiguous_dim > 0) {
    std::size_t dim = contiguous_dim - 1;
    if (dim == shape.size() - 1) {
      if (dst_strides[dim] == 1 && src_strides[dim] == 1) {
        contiguous_bytes *= shape[dim];
        contiguous_dim = dim;
      } else {
        break;
      }
    } else {
      if (dst_strides[dim] == dst_strides[dim + 1] *
                                  static_cast<std::int64_t>(shape[dim + 1]) &&
          src_strides[dim] == src_strides[dim + 1] *
                                  static_cast<std::int64_t>(shape[dim + 1])) {
        contiguous_bytes *= shape[dim];
        contiguous_dim = dim;
      } else {
        break;
      }
    }
  }

  auto copy_dim = [&](auto& self, std::size_t dim, const std::byte* s,
                      std::byte* d) -> void {
    if (dim >= contiguous_dim) {
      std::memcpy(d, s, contiguous_bytes);
      return;
    }
    for (std::uint64_t i = 0; i < shape[dim]; ++i) {
      self(self, dim + 1, s + i * src_strides[dim] * elem_size,
           d + i * dst_strides[dim] * elem_size);
    }
  };
  copy_dim(copy_dim, 0, src, dst);
}

std::vector<std::int64_t> DefaultStrides(
    const std::vector<std::uint64_t>& shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  if (shape.empty()) return strides;
  for (std::size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * static_cast<std::int64_t>(shape[i]);
  }
  return strides;
}

std::vector<std::int64_t> GetStride(const pblczero::Buffer& buffer) {
  if (buffer.has_layout()) {
    return {buffer.layout().strides()};
  }
  return DefaultStrides(buffer.shape());
}

void CopyTensorToHostStaging(const pblczero::Buffer& buffer,
                             std::span<const std::byte> source,
                             std::span<std::byte> destination_staging) {
  const auto elem_size = DataTypeSize(buffer.data_type());
  const auto src_strides = DefaultStrides(buffer.shape());

  if (buffer.offset() >= destination_staging.size()) {
    throw Exception("Buffer '" + std::string(buffer.name()) +
                    "' offset exceeds persistent allocation bounds.");
  }

  CopyStridedHostTensor(buffer.shape(), GetStride(buffer), src_strides,
                        elem_size, source.data(),
                        destination_staging.data() + buffer.offset());
}

template <typename BackendType, typename ExecutableType>
void UploadWeights(BackendType& backend, const WeightsFile& weights,
                   ExecutableType& executable,
                   const pblczero::NeuralExecutable& executable_proto,
                   const OptionsDict& backend_options) {
  LCTRACE_FUNCTION_SCOPE;
  std::optional<WeightsFile> converted_weights;
  if (!weights.has_onnx_model()) {
    CERR << "Converting weights to ONNX first.";
    CERR << "HINT: Loading can be made faster using leela2onnx and onnx2leela "
            "commands. You can decompress the resulting file for a little "
            "faster loading.";
    CheckNetworkFingerprint(weights, executable_proto);
    converted_weights =
        ConvertWeightsToOnnx(weights, MakeConverterOptions(backend_options));
  }

  const auto& onnx_weights = converted_weights ? *converted_weights : weights;

  backend.InitializeBackendAttributes(onnx_weights);

  pblczero::ModelProto onnx;
  onnx.ParseFromString(onnx_weights.onnx_model().model());

  std::unordered_map<std::string_view, std::tuple<const pblczero::TensorProto*,
                                                  const pblczero::Buffer*>>
      initializers_by_name;
  initializers_by_name.reserve(onnx.graph().initializer_size());
  std::string duplicate_initializer;
  const bool unique_initializers =
      absl::c_all_of(onnx.graph().initializer(), [&](const auto& initializer) {
        const auto name = initializer.name();
        decltype(initializers_by_name)::value_type::second_type value(
            &initializer, nullptr);
        if (!initializers_by_name.emplace(name, value).second) {
          duplicate_initializer = name;
          return false;
        }
        return true;
      });
  if (!unique_initializers) {
    throw Exception("The ONNX model contains duplicate initializer '" +
                    duplicate_initializer + "'.");
  }

  std::string_view missing_buffer;
  if (absl::c_any_of(executable_proto.buffers(), [&](const auto& buffer) {
        auto iter = initializers_by_name.find(buffer.name());
        if (iter == initializers_by_name.end()) {
          missing_buffer = buffer.name();
          return true;
        }
        std::get<1>(iter->second) = &buffer;
        return false;
      })) {
    throw Exception("The lc0ex buffer '" + std::string(missing_buffer) +
                    "' has no corresponding ONNX initializer.");
  }

  CERR << "Uploading ONNX initializers to the lc0ex runtime.";
  std::vector<std::byte> staging(executable.GetPersistentAllocationSize(),
                                 std::byte{0});

  for (const auto& initializer : onnx.graph().initializer()) {
    auto iter = initializers_by_name.find(initializer.name());
    if (iter == initializers_by_name.end() || !std::get<1>(iter->second)) {
      CERR << "WARNING: ONNX initializer '" << initializer.name()
           << "' has no corresponding lc0ex buffer.";
      continue;
    }

    ValidateInitializer(*std::get<1>(iter->second), initializer);
    const auto source = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(initializer.raw_data().data()),
        initializer.raw_data().size());
    CopyTensorToHostStaging(*std::get<1>(iter->second), source, staging);
  }

  executable.CopyPersistentFromHost(staging);
}

BackendAttributes MakeBackendAttributes() {
  return {
      .has_mlh = true,
      .has_wdl = true,
      .runs_on_cpu = false,
      .suggested_num_search_threads = 1,
      .recommended_batch_size = 0,
      .maximum_batch_size = 0,
  };
}

template <typename RuntimeType, typename ComputeType>
class Lc0exBackend;
template <typename RuntimeType, typename ComputeType>
class Lc0exBackendComputation;

// Persitent computation cache per thread GPU resources for efficient concurrent
// inference.
template <typename RuntimeType, typename ComputeType>
class Lc0exPersistentComputation {
 public:
  using Memory = typename RuntimeType::Memory;
  using HostMemory = typename RuntimeType::HostMemory;
  using Stream = typename RuntimeType::Stream;
  using Event = typename RuntimeType::Event;
  using GraphExec = typename RuntimeType::GraphExec;
  template <typename T>
  using Buffer = typename RuntimeType::template Buffer<T>;
  template <typename T>
  using HostBuffer = typename RuntimeType::template HostBuffer<T>;
  using Executable = typename RuntimeType::Executable;
  using Backend = Lc0exBackend<RuntimeType, ComputeType>;

  Lc0exPersistentComputation(Backend& backend, Executable& executable);

  size_t Size() const { return entries_.size(); }

  void Clear() {
    std::fill(total_legal_moves_.begin(), total_legal_moves_.end(), -1);
    entries_.clear();
  }

  auto& GetState() { return state_; }

  template <typename CaptureType>
  void InitialGraphCapture(size_t batch_idx, CaptureType& graph) {
    graphs_[batch_idx] = graph;
    graphs_[batch_idx].Upload(state_.stream_);
  }

  void DecodeWdl() const {
    for (size_t i = 0; i < entries_.size(); ++i) {
      auto logits_input =
          state_.output_wdl.subspan(i * kNumWdlOutputs, kNumWdlOutputs);
      auto& result = entries_[i].result;
      std::array<float, 3> logits;
      std::copy(logits_input.begin(), logits_input.end(), logits.begin());
      const float maximum = std::max({logits[0], logits[1], logits[2]});
      const float win = std::exp(logits[0] - maximum);
      const float draw = std::exp(logits[1] - maximum);
      const float loss = std::exp(logits[2] - maximum);
      const float scale = 1.0f / (win + draw + loss);

      if (result.q) *result.q = (win - loss) * scale;
      if (result.d) *result.d = draw * scale;
    }
  }

  void DecodePolicy(float inverse_policy_temperature) const {
    for (size_t i = 0; i < entries_.size(); ++i) {
      DecodePolicy(i, state_.output_policy, inverse_policy_temperature);
    }
  }

  void DecodeMlh() const {
    for (size_t i = 0; i < entries_.size(); ++i) {
      auto& result = entries_[i].result;
      if (result.m) {
        *result.m = state_.output_mlh[i];
      }
    }
  }

 private:
  template <typename T>
  struct SameSizeInt {
    using type =
        std::conditional_t<sizeof(T) == 2, uint16_t,
                           std::conditional_t<sizeof(T) == 4, uint32_t, void>>;
  };
  template <typename T>
  using SameSizeIntT = typename SameSizeInt<T>::type;
  struct Entry {
    EvalResultPtr result;
  };

  void DecodePolicy(size_t index, std::span<const ComputeType> logits,
                    float inverse_policy_temperature) const;

  static constexpr size_t kGpuAlignment = 256;
  static size_t AlignTo(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  }

  static size_t GetHostAllocationSize(size_t max_batch) {
    return std::max(
        AlignTo(sizeof(uint32_t) * max_batch * kNumOutputPolicy,
                kGpuAlignment) +
            AlignTo(sizeof(uint64_t) * max_batch * kInputPlanes,
                    kGpuAlignment) +
            AlignTo(sizeof(ComputeType) * max_batch * kInputPlanes,
                    kGpuAlignment),
        AlignTo(sizeof(ComputeType) * max_batch * kNumOutputPolicy,
                kGpuAlignment) +
            AlignTo(sizeof(ComputeType) * max_batch * kNumWdlOutputs,
                    kGpuAlignment) +
            AlignTo(sizeof(ComputeType) * max_batch, kGpuAlignment));
  }

  AtomicVector<Entry> entries_;
  std::vector<int> total_legal_moves_;
  std::unique_ptr<GraphExec[]> graphs_;

  HostMemory host_memory_;
  lc0ex::ComputationState<RuntimeType, ComputeType> state_;

  friend class Lc0exBackendComputation<RuntimeType, ComputeType>;
};

// A thin wrapper computation to access the persistent computation.
template <typename RuntimeType, typename ComputeType>
class Lc0exBackendComputation final : public BackendComputation {
 public:
  using Persistent = Lc0exPersistentComputation<RuntimeType, ComputeType>;
  using Backend = Lc0exBackend<RuntimeType, ComputeType>;
  explicit Lc0exBackendComputation(Backend* backend,
                                   std::unique_ptr<Persistent>&& persistent);
  ~Lc0exBackendComputation() override;

  size_t UsedBatchSize() const override { return persistent_->Size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override;

  void ComputeBlocking() override;

 private:
  void ExecuteProgram(std::size_t actual_batch);

  Backend* backend_;
  std::unique_ptr<Persistent> persistent_;
};

// `auto` is the DAG graph (R34). R22 keyed this on the configured concurrency,
// because the graph then measured +25 % with one execution slot in flight and
// -9 % with two. That penalty was the graph serialising against *synchronous*
// host-to-device copies, not against the second slot: once the copies moved
// onto the slot's non-blocking stream the sign flipped, and the graph is now
// +7.2 % at two threads and batch 16 and never negative in a search. `linear`
// and `off` remain for diagnosis -- `off` in particular, because a
// graph-launched backend needs `nsys --cuda-graph-trace=node` to show its
// kernels at all.
lc0ex::GraphMode ResolveGraphMode(const std::string& value) {
  if (value == "dag" || value == "on") return lc0ex::GraphMode::kDag;
  if (value == "linear") return lc0ex::GraphMode::kLinear;
  if (value == "off") return lc0ex::GraphMode::kOff;
  if (value != "auto") {
    throw Exception("Unknown lc0ex graph mode '" + value +
                    "'; expected auto, dag, linear or off.");
  }
  return lc0ex::GraphMode::kDag;
}
using ExecutableVariant =
    std::variant<const lc0ex::cuda::CudaRuntime::Executable*>;

template <typename RuntimeTypeParam, typename ComputeType>
class Lc0exBackend final : public Backend {
 public:
  using RuntimeType = RuntimeTypeParam;
  using Persistent = Lc0exPersistentComputation<RuntimeType, ComputeType>;
  using FreePersistentList =
      absl::InlinedVector<std::unique_ptr<Persistent>, 3>;

  Lc0exBackend(const OptionsDict& options, const OptionsDict& backend_options,
               const pblczero::NeuralExecutable& executable_proto,
               std::promise<ExecutableVariant>& promise)
      : attributes_(MakeBackendAttributes()),
        graph_mode_(ResolveGraphMode(
            backend_options.GetOrDefault<std::string>("graph", "auto"))),
        runtime_(backend_options.GetOrDefault<int>("gpu", 0)),
        executable_(executable_proto),
        compute_ordering_event_(lc0ex::EventFlags::DEFAULT),
        backend_options_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
        weights_path_(
            options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    LCTRACE_FUNCTION_SCOPE;
    lc0ex::cuda::SetComputeOrdering(
        compute_ordering_event_,
        backend_options.GetOrDefault<std::string>("ordering", "event"));
    // Notify weight loading thread that it can start using cuda executable now.
    promise.set_value(&executable_);
    UpdateConfiguration(options);

    auto max_batch_size = executable_.GetMaxBatchSize();
    opt_batch_ = backend_options.GetOrDefault<int>("opt_batch", max_batch_size);
    max_batch_ = backend_options.GetOrDefault<int>("max_batch", max_batch_size);
    max_batch_ = std::clamp<int>(max_batch_, 1, max_batch_size);
    opt_batch_ = std::clamp(opt_batch_, 1, max_batch_);
    attributes_.maximum_batch_size = max_batch_;
    attributes_.recommended_batch_size = opt_batch_;
  }

  void CaptureGraphs(const OptionsDict& backend_options)
      NO_THREAD_SAFETY_ANALYSIS {
    if (graph_mode_ == lc0ex::GraphMode::kOff) {
      return;
    }
    LCTRACE_FUNCTION_SCOPE;
    size_t number_of_graphs =
        backend_options.GetOrDefault("capture_graphs_onload", 2);
    for (size_t i = 0; i < number_of_graphs; ++i) {
      persistent_.emplace_back(
          std::make_unique<Persistent>(*this, executable_));
    }
    for (size_t i = 0; i < static_cast<size_t>(max_batch_); ++i) {
      for (auto& persistent : persistent_) {
        auto graph =
            executable_.Capture(graph_mode_, i + 1, persistent->GetState(),
                                compute_ordering_event_);
        persistent->InitialGraphCapture(i, graph);
      }
    }
  }

  ~Lc0exBackend() override { runtime_.SetCurrent(); }

  BackendAttributes GetAttributes() const override { return attributes_; }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    runtime_.SetCurrent();

    Mutex::Lock lock(persistent_lock_);

    if (!persistent_.empty()) {
      auto rv =
          std::make_unique<Lc0exBackendComputation<RuntimeType, ComputeType>>(
              this, std::move(persistent_.back()));
      persistent_.pop_back();

      return rv;
    }
    lock.unlock();
    return std::make_unique<Lc0exBackendComputation<RuntimeType, ComputeType>>(
        this, std::make_unique<Persistent>(*this, executable_));
  }

  void PushPersistent(std::unique_ptr<Persistent>&& persistent) {
    Mutex::Lock lock(persistent_lock_);
    persistent_.push_back(std::move(persistent));
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    if (backend_options_ !=
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId)) {
      return NEED_RESTART;
    }
    if (weights_path_ !=
        options.Get<std::string>(SharedBackendParams::kWeightsId)) {
      return NEED_RESTART;
    }

    inverse_policy_temperature_ =
        1.0f / options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    fill_empty_history_ = ParseHistoryFill(
        options.Get<std::string>(SharedBackendParams::kHistoryFill));
    return UPDATE_OK;
  }

  // compute ordering mutex offloads batch ordering to GPU. It makes sure that
  // subsequent batch starts only when it won't any more cause performance
  // problems. If Triton produced an optimised graph for multi_stream
  // configuration this mutex could be conditionally disabled to allow all
  // threads submit GPU work concurrently. Triton side would require major
  // changes to optimise kernels for concurrent execution.
  std::unique_lock<Mutex> GetComputeOrderingLock() {
    return std::unique_lock{compute_ordering_mutex_};
  }

  // Called from the weight loading thread to initialize backend attributes
  // after the weights have been converted to ONNX and the executable has been
  // loaded. SetCurrent is required to allow weight upload thread to do weight
  // upload directly.
  void InitializeBackendAttributes(const WeightsFile& weights) {
    runtime_.SetCurrent();
    const auto& onnx = weights.onnx_model();
    const auto& format = weights.format().network_format();
    if (onnx.has_output_wdl() || onnx.has_output_value() ||
        onnx.has_output_mlh()) {
      attributes_.has_mlh = onnx.has_output_mlh();
      attributes_.has_wdl = onnx.has_output_wdl();
    } else {
      // A carrier ONNX model only ships initializers, with no graph outputs
      // to name the heads; take them from the network format instead.
      attributes_.has_mlh =
          format.moves_left() == pblczero::NetworkFormat::MOVES_LEFT_V1;
      attributes_.has_wdl =
          format.value() == pblczero::NetworkFormat::VALUE_WDL;
    }
    input_format_ = format.input();
  }

 private:
  BackendAttributes attributes_;
  int opt_batch_ = 0;
  int max_batch_ = 0;
  float inverse_policy_temperature_ = 1.0f;
  lc0ex::GraphMode graph_mode_ = lc0ex::GraphMode::kDag;
  RuntimeType runtime_;
  RuntimeType::Executable executable_;
  RuntimeType::Event compute_ordering_event_;
  Mutex compute_ordering_mutex_;
  const std::string backend_options_;
  const std::string weights_path_;
  pblczero::NetworkFormat::InputFormat input_format_;
  FillEmptyHistory fill_empty_history_ = FillEmptyHistory::NO;
  Mutex persistent_lock_;
  FreePersistentList persistent_ GUARDED_BY(persistent_lock_);

  friend class Lc0exBackendComputation<RuntimeType, ComputeType>;
  friend class Lc0exPersistentComputation<RuntimeType, ComputeType>;
};

template <typename RuntimeType, typename ComputeType>
Lc0exPersistentComputation<RuntimeType, ComputeType>::
    Lc0exPersistentComputation(Backend& backend, Executable& executable)
    : entries_(backend.GetAttributes().maximum_batch_size),
      total_legal_moves_(backend.GetAttributes().maximum_batch_size),
      graphs_(std::make_unique<GraphExec[]>(
          backend.GetAttributes().maximum_batch_size)),
      host_memory_(
          GetHostAllocationSize(backend.GetAttributes().maximum_batch_size),
          kGpuAlignment),
      state_{
          .stream_ = {lc0ex::StreamFlags::DEFAULT},
          .device_memory_ = {executable.GetExecutionAllocationSize(),
                             executable.GetExecutionAllocationAlignment()},
          .input_mapping = {host_memory_.template AsSpan<std::uint32_t>(
              0, sizeof(std::uint32_t) *
                     backend.GetAttributes().maximum_batch_size *
                     kNumOutputPolicy)},
          .input_mask = {host_memory_.template AsSpan<uint64_t>(
              AlignTo(sizeof(std::uint32_t) *
                          backend.GetAttributes().maximum_batch_size *
                          kNumOutputPolicy,
                      kGpuAlignment),
              sizeof(uint64_t) * backend.GetAttributes().maximum_batch_size *
                  kInputPlanes)},
          .input_value = {host_memory_.template AsSpan<ComputeType>(
              AlignTo(sizeof(std::uint32_t) *
                          backend.GetAttributes().maximum_batch_size *
                          kNumOutputPolicy,
                      kGpuAlignment) +
                  AlignTo(sizeof(uint64_t) *
                              backend.GetAttributes().maximum_batch_size *
                              kInputPlanes,
                          kGpuAlignment),
              sizeof(ComputeType) * backend.GetAttributes().maximum_batch_size *
                  kInputPlanes)},
          .output_policy = {host_memory_.template AsSpan<ComputeType>(
              0, sizeof(ComputeType) *
                     backend.GetAttributes().maximum_batch_size *
                     kNumOutputPolicy)},
          .output_wdl = {host_memory_.template AsSpan<ComputeType>(
              AlignTo(sizeof(ComputeType) *
                          backend.GetAttributes().maximum_batch_size *
                          kNumOutputPolicy,
                      kGpuAlignment),
              sizeof(ComputeType) * backend.GetAttributes().maximum_batch_size *
                  kNumWdlOutputs)},
          .output_mlh = {host_memory_.template AsSpan<ComputeType>(
              AlignTo(sizeof(ComputeType) *
                          backend.GetAttributes().maximum_batch_size *
                          kNumOutputPolicy,
                      kGpuAlignment) +
                  AlignTo(sizeof(ComputeType) *
                              backend.GetAttributes().maximum_batch_size *
                              kNumWdlOutputs,
                          kGpuAlignment),
              sizeof(ComputeType) *
                  backend.GetAttributes().maximum_batch_size)},
          .sleep_event_ = {lc0ex::EventFlags::BLOCKING},
          .wdl_download_done_ = {lc0ex::EventFlags::DEFAULT},
          .mlh_download_done_ = {lc0ex::EventFlags::DEFAULT},
          .policy_download_done_ = {lc0ex::EventFlags::DEFAULT},
          .total_legal_moves_ = {1},
      }

{}

template <typename RuntimeType, typename ComputeType>
BackendComputation::AddInputResult
Lc0exBackendComputation<RuntimeType, ComputeType>::AddInput(
    const EvalPosition& pos, EvalResultPtr result) {
  int transform = 0;
  const InputPlanes input =
      EncodePositionForNN(backend_->input_format_, pos.pos, kMoveHistory,
                          backend_->fill_empty_history_, &transform);

  typename Persistent::Entry entry{
      .result = result,
  };
  size_t idx = persistent_->entries_.emplace_back(std::move(entry));
  int start_legal_moves = 0;
  if (idx > 0) {
    std::atomic_ref<int> previous_total_legal_moves(
        persistent_->total_legal_moves_[idx - 1]);
    while ((start_legal_moves = previous_total_legal_moves.load(
                std::memory_order_relaxed)) == -1) {
      SpinloopPause();
    }
  }
  std::atomic_ref<int> total_legal_moves(persistent_->total_legal_moves_[idx]);
  total_legal_moves.store(
      start_legal_moves + static_cast<int>(pos.legal_moves.size()),
      std::memory_order_relaxed);
  const size_t base = idx * kInputPlanes;
  auto mask = persistent_->state_.input_mask;
  auto value = persistent_->state_.input_value;
  auto policy_map = persistent_->state_.input_mapping;
  std::transform(pos.legal_moves.begin(), pos.legal_moves.end(),
                 policy_map.begin() + start_legal_moves,
                 [transform, idx](const Move move) {
                   return MoveToPremapIndex(move, transform) +
                          idx * std::size(kAttnPolicyMap);
                 });
  for (std::size_t i = 0; i < kInputPlanes; ++i) {
    mask[base + i] = input[i].mask;
    value[base + i] = input[i].value;
  }
  return ENQUEUED_FOR_EVAL;
}

template <typename RuntimeType, typename ComputeType>
Lc0exBackendComputation<RuntimeType, ComputeType>::Lc0exBackendComputation(
    Backend* backend, std::unique_ptr<Persistent>&& persistent)
    : backend_(backend), persistent_(std::move(persistent)) {
  persistent_->Clear();
}

template <typename RuntimeType, typename ComputeType>
Lc0exBackendComputation<RuntimeType, ComputeType>::~Lc0exBackendComputation() {
  backend_->PushPersistent(std::move(persistent_));
}

template <typename RuntimeType, typename ComputeType>
void Lc0exPersistentComputation<RuntimeType, ComputeType>::DecodePolicy(
    size_t idx, std::span<const ComputeType> logits,
    float inverse_policy_temperature) const {
  const auto& entry = entries_[idx];
  if (entry.result.p.empty()) return;
  size_t legal_moves = total_legal_moves_[idx];
  size_t start_index = 0;
  if (idx > 0) {
    start_index = total_legal_moves_[idx - 1];
    legal_moves -= start_index;
  }
  assert(legal_moves == entry.result.p.size());
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < legal_moves; ++i) {
    entry.result.p[i] = logits[start_index + i];
    maximum = std::max(maximum, entry.result.p[i]);
  }

  float total = 0.0f;
  for (float& value : entry.result.p) {
    value = FastExp((value - maximum) * inverse_policy_temperature);
    total += value;
  }
  const float scale = total > 0.0f ? 1.0f / total : 1.0f;
  for (float& value : entry.result.p) value *= scale;
}

template <typename RuntimeType, typename ComputeType>
void Lc0exBackendComputation<RuntimeType, ComputeType>::ExecuteProgram(
    size_t actual_batch) {
  backend_->executable_.Run(actual_batch, persistent_->state_,
                            backend_->compute_ordering_event_);
}

template <typename RuntimeType, typename ComputeType>
void Lc0exBackendComputation<RuntimeType, ComputeType>::ComputeBlocking() {
  const size_t actual_batch = persistent_->entries_.size();
  if (actual_batch == 0) return;
  LCTRACE_FUNCTION_SCOPE;

  size_t legal_moves = persistent_->total_legal_moves_[actual_batch - 1];
  if (legal_moves == 0) {
    legal_moves = 1;
    persistent_->state_.input_mapping[0] = 0;
  }
  persistent_->state_.total_legal_moves_ = legal_moves;

  if (persistent_->graphs_[actual_batch - 1]) {
    std::unique_lock<Mutex> lock(backend_->GetComputeOrderingLock());
    persistent_->state_.ordering_ticket_ =
        lc0ex::cuda::NextOrderingTicket(backend_->compute_ordering_event_);
    persistent_->graphs_[actual_batch - 1].Launch(persistent_->state_.stream_);
  } else {
    std::unique_lock<Mutex> lock(backend_->GetComputeOrderingLock());
    persistent_->state_.ordering_ticket_ =
        lc0ex::cuda::NextOrderingTicket(backend_->compute_ordering_event_);
    ExecuteProgram(actual_batch);

    if (backend_->graph_mode_ != lc0ex::GraphMode::kOff) {
      // Keep lock for graph construction because all cuda driver calls would
      // have lock contention if other thread is submitting work or capturing
      // graphs. External lock avoids thread bouncing slowdown.
      auto graph = backend_->executable_.Capture(
          backend_->graph_mode_, actual_batch, persistent_->state_,
          backend_->compute_ordering_event_);

      // We allow other thread to submit work after the graph construction
      // because following instantiation is slow and suffers less from
      // concurrent GPU submissions.
      if (lock.owns_lock()) {
        lock.unlock();
      }

      // Graph instantiation happens when it is copied to the exec holder.
      persistent_->graphs_[actual_batch - 1] = graph;
      // We trigger upload now to avoid potential slower graph launch when there
      // is likely less time left before deadline or we are already late.
      persistent_->graphs_[actual_batch - 1].Upload(
          persistent_->state_.stream_);
    }
  }

  // Do blocking wait on the GPU which triggers early to account for wake up
  // latency. Blocking sleep makes only sense if the batch is large enough to
  // let heads take longer than the wakeup latency.
  if (static_cast<int>(actual_batch * 5) >
      backend_->GetAttributes().recommended_batch_size) {
    persistent_->state_.sleep_event_.Synchronize();
  }

  constexpr size_t kWdlHead = 0x1;
  constexpr size_t kPolicyHead = 0x2;
  constexpr size_t kMlhHead = 0x4;
  size_t pending_heads = kWdlHead | kPolicyHead | kMlhHead;

  while (pending_heads) {
    if ((pending_heads & kWdlHead) != 0) {
      bool found = pending_heads == kWdlHead;
      if (found) {
        persistent_->state_.wdl_download_done_.Synchronize();
      } else {
        found = persistent_->state_.wdl_download_done_.IsCompleted();
      }
      if (found) {
        pending_heads ^= kWdlHead;
        persistent_->DecodeWdl();
      }
    }
    if ((pending_heads & kPolicyHead) != 0) {
      bool found = pending_heads == kPolicyHead;
      if (found) {
        persistent_->state_.policy_download_done_.Synchronize();
      } else {
        found = persistent_->state_.policy_download_done_.IsCompleted();
      }
      if (found) {
        pending_heads ^= kPolicyHead;
        persistent_->DecodePolicy(backend_->inverse_policy_temperature_);
      }
    }
    if ((pending_heads & kMlhHead) != 0) {
      bool found = pending_heads == kMlhHead;
      if (found) {
        persistent_->state_.mlh_download_done_.Synchronize();
      } else {
        found = persistent_->state_.mlh_download_done_.IsCompleted();
      }
      if (found) {
        pending_heads ^= kMlhHead;
        persistent_->DecodeMlh();
      }
    }
  }
}

class Lc0exBackendFactory final : public BackendFactory {
 public:
  int GetPriority() const override { return 1; }
  std::string_view GetName() const override { return kBackendName; }

  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    std::unique_ptr<Backend> backend;
    pblczero::NeuralExecutable executable_proto;
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId));

    std::future<bool> weights;

    {
      using BackendVariant =
          std::variant<Lc0exBackend<lc0ex::cuda::CudaRuntime, lc0ex::Half>*,
                       Lc0exBackend<lc0ex::cuda::CudaRuntime, float>*>;
      std::promise<BackendVariant> backend_promise;
      auto backend_future = backend_promise.get_future();
      std::promise<ExecutableVariant> executable_promise;
      auto executable_future = executable_promise.get_future();
      std::promise<const pblczero::NeuralExecutable*> executable_proto_promise;
      auto executable_proto_future = executable_proto_promise.get_future();
      // Load the weights in a separate thread to allow cuda initialization
      // and graph capture for free.
      weights = std::async(
          std::launch::async,
          [backend_future = std::move(backend_future),
           executable_future = std::move(executable_future),
           executable_proto_future = std::move(executable_proto_future),
           &options, &backend_options]() mutable {
            const std::string weights_path =
                options.Get<std::string>(SharedBackendParams::kWeightsId);
            const std::optional<WeightsFile> weights =
                LoadWeights(weights_path);
            if (!weights) {
              throw Exception(
                  "The lc0ex-cuda backend requires a network file.");
            }

            auto backend_variant = backend_future.get();
            auto source = executable_proto_future.get();
            auto executable_variant = executable_future.get();

            // Call correct backend variant.
            std::visit(
                [&](auto* backend) {
                  using RuntimeType =
                      typename std::decay_t<decltype(*backend)>::RuntimeType;
                  using ExecutableType = typename RuntimeType::Executable;
                  auto exec =
                      std::get<const ExecutableType*>(executable_variant);
                  UploadWeights(*backend, *weights, *exec, *source,
                                backend_options);
                },
                backend_variant);
            return true;
          });

      const std::string lc0ex_path = backend_options.Get<std::string>("lc0ex");
      if (lc0ex_path.empty()) {
        throw Exception("The lc0ex-cuda backend requires an lc0ex path.");
      }

      executable_proto = LoadExecutableFile(lc0ex_path);

      executable_proto_promise.set_value(&executable_proto);

      if (executable_proto.io_data_type() ==
          pblczero::Buffer_DataType_DATA_TYPE_F16) {
        auto typed_backend =
            std::make_unique<Lc0exBackend<lc0ex::cuda::CudaRuntime, lc0ex::Half>>(
                options, backend_options, executable_proto, executable_promise);
        backend_promise.set_value(typed_backend.get());
        auto typed_ptr = typed_backend.get();
        backend = std::move(typed_backend);
        typed_ptr->CaptureGraphs(backend_options);
      } else if (executable_proto.io_data_type() ==
                 pblczero::Buffer_DataType_DATA_TYPE_F32) {
        auto typed_backend =
            std::make_unique<Lc0exBackend<lc0ex::cuda::CudaRuntime, float>>(
                options, backend_options, executable_proto, executable_promise);
        backend_promise.set_value(typed_backend.get());
        auto typed_ptr = typed_backend.get();
        backend = std::move(typed_backend);
        typed_ptr->CaptureGraphs(backend_options);
      } else {
        backend_promise.set_exception(std::make_exception_ptr(
            Exception("Unsupported data type for lc0ex-cuda backend.")));
      }
      backend_options.CheckAllOptionsRead(std::string(kBackendName));
    }

    // Wait for the weights to be uploaded before returning the backend.
    weights.get();
    return backend;
  }
};

REGISTER_BACKEND(Lc0exBackendFactory)

}  // namespace
}  // namespace lczero
