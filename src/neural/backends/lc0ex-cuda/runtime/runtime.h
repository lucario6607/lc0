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

#include <absl/container/inlined_vector.h>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if __cpp_lib_source_location >= 201907L
#include <source_location>
#endif

#include "proto/lc0ex.pb.h"
#include "utils/fp16_utils.h"

namespace lczero::lc0ex {

// Host-side storage type for FP16 network IO. MSVC has no _Float16.
#if defined(_MSC_VER) && !defined(__clang__)
struct Half {
  uint16_t bits;
  Half() = default;
  Half(float f) : bits(FP32toFP16(f)) {}
  operator float() const { return FP16toFP32(bits); }
};
#else
using Half = _Float16;
#endif
static_assert(sizeof(Half) == 2);

// How an Execution issues its kernel launch loop.
//   kOff    - a plain launch loop, one cuLaunchKernel per node.
//   kDag    - upstream's behaviour: a CUDA graph whose edges are the node
//             dependencies, so independent kernels may run concurrently.
//   kLinear - stream capture of the plain launch loop into a linear graph:
//             the launch-overhead saving without the cross-kernel concurrency.
// R22 measured kDag at +25 % with one execution slot in flight and -9 % with
// two, so the choice has to be made by the caller, not baked in.
enum class GraphMode { kOff, kDag, kLinear };


// A computation per thread computation state which will be passed to an
// executable when running or capturing a graph.
template <typename R, typename C>
struct ComputationState {
  typename R::Stream stream_;
  typename R::Memory device_memory_;
  typename R::template HostBuffer<uint32_t> input_mapping;
  typename R::template HostBuffer<uint64_t> input_mask;
  typename R::template HostBuffer<C> input_value;
  typename R::template HostBuffer<C> output_policy;
  typename R::template HostBuffer<C> output_wdl;
  typename R::template HostBuffer<C> output_mlh;
  typename R::Event sleep_event_;
  typename R::Event wdl_download_done_;
  typename R::Event mlh_download_done_;
  typename R::Event policy_download_done_;
  size_t total_legal_moves_;
  uint64_t ordering_ticket_ = 0;
};

// Handles and descriptor references returned by an Executable remain valid
// until that Executable is destroyed. An Execution and its buffers must not
// outlive their Executable. Distinct Executions may be used concurrently; an
// individual Execution must only be accessed by one host thread at a time.
struct TargetInfo {
  pblczero::Target::Vendor vendor = pblczero::Target::VENDOR_UNKNOWN;
  std::string architecture;
};

struct BufferInfo {
  struct ShapeAndStride {
    std::uint64_t shape = 0;
    std::int64_t stride = 0;
  };
  pblczero::Buffer::DataType data_type = pblczero::Buffer::DATA_TYPE_UNKNOWN;
  absl::InlinedVector<ShapeAndStride, 2> shape;
  size_t size_bytes = 0;
  ptrdiff_t offset_bytes = 0;
};

struct AllocationInfo {
  size_t size_bytes_ = 0;
  size_t alignment_bytes_ = 0;
};

class MemoryBase {
 public:
  MemoryBase() = default;
  MemoryBase(size_t size, size_t alignment)
      : size_(size), alignment_(alignment) {}
  ~MemoryBase() = default;

  MemoryBase(const MemoryBase&) = delete;
  MemoryBase& operator=(const MemoryBase&) = delete;
  MemoryBase(MemoryBase&& other) noexcept
      : aligned_ptr_(other.aligned_ptr_),
        size_(other.size_),
        alignment_(other.alignment_) {
    other.aligned_ptr_ = nullptr;
    other.size_ = 0;
    other.alignment_ = 0;
  }
  MemoryBase& operator=(MemoryBase&& other) noexcept {
    if (this != &other) {
      aligned_ptr_ = other.aligned_ptr_;
      size_ = other.size_;
      alignment_ = other.alignment_;
      other.aligned_ptr_ = nullptr;
      other.size_ = 0;
      other.alignment_ = 0;
    }
    return *this;
  }
  
  void IsValidRange(
      std::string_view name, ptrdiff_t offset, size_t size
#if __cpp_lib_source_location >= 201907L
      ,
      const std::source_location& location = std::source_location::current()
#endif
  ) const;

  explicit operator bool() const { return aligned_ptr_ != nullptr; }

  template <typename T>
  std::span<T> AsSpan(ptrdiff_t offset, size_t size) const noexcept {
    assert(size % sizeof(T) == 0);
    assert(IsValidRange(offset, size));
    return {reinterpret_cast<T*>(aligned_ptr_ + offset), size / sizeof(T)};
  }

  std::byte* Data() noexcept {
    return reinterpret_cast<std::byte*>(aligned_ptr_);
  }

  const std::byte* Data() const noexcept {
    return reinterpret_cast<const std::byte*>(aligned_ptr_);
  }

  size_t Size() const noexcept { return size_; }
  size_t Alignment() const noexcept { return alignment_; }

 protected:
  bool IsValidRange(ptrdiff_t offset, size_t size) const noexcept {
    return offset >= 0 && static_cast<size_t>(offset) + size <= size_;
  }
  char* aligned_ptr_ = nullptr;
  size_t size_ = 0;
  size_t alignment_ = 0;
};

template <typename T, typename MemoryType>
class BufferBase : public std::span<T> {
  using Base = std::span<T>;

  Base Init(std::string_view name, MemoryType& memory, ptrdiff_t offset,
            size_t size) {
    memory->IsValidRange(name, offset, size);
    return memory->AsSpan(T{}, offset, size);
  }

 public:
  BufferBase() = default;
  BufferBase(std::string_view name, MemoryType& memory, ptrdiff_t offset,
             size_t size)
      : Base(Init(name, memory, offset, size)) {}
  BufferBase(const std::span<T>& span) : Base(span) {}
};

enum class StreamFlags : int {
  DEFAULT = 0,
};

enum class EventFlags : int {
  DEFAULT = 0,
  BLOCKING = 1 << 0,
  USE_TIMING = 1 << 1,
};

inline EventFlags operator&(EventFlags lhs, EventFlags rhs) {
  return static_cast<EventFlags>(static_cast<int>(lhs) & static_cast<int>(rhs));
}

inline EventFlags operator|(EventFlags lhs, EventFlags rhs) {
  return static_cast<EventFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline EventFlags operator~(EventFlags flag) {
  return static_cast<EventFlags>(~static_cast<int>(flag));
}

class NodeBase {
 public:
  NodeBase(const pblczero::Node& node);
  ~NodeBase() = default;

  std::span<const uint32_t> GetDependecies() const noexcept {
    return {dependencies_.data(), dependencies_.size()};
  }

  int GetPriority() const noexcept { return priority_; }

 protected:
  absl::InlinedVector<uint32_t, 4> dependencies_;
  int priority_ = 0;
};

}  // namespace lczero::lc0ex
