# lc0ex on RTX 5080 (sm_120) — BT4-tf13g-nes-it325

Branch base: `Menkib64/lc0` `lc0ex_graph_and_ownership_rework` (158b5ca).
Builder base: `Menkib64/lczero-triton` `graph_memcpy_events_and_priorities` (8ab2641);
`git am lc0ex-5080/builder/*.patch` on that branch.

All numbers: `lc0 backendbench --threads=2`, one batch size per run, started from a
cooled GPU (<= 45 C), interleaved A/B, correctness checked with `backendcompare`
against `cuda-fp16` on 2000 random positions (policy KL ~1e-5, same class as onnx-trt).

## Runtime changes (this branch, `src/neural/backends/lc0ex-cuda`)

* **Builds with MSVC on Windows.** `_Float16` is replaced by a portable `lc0ex::Half`
  (F16C conversions) and `GraphCapture`'s conversion operator is constrained to
  pointer types so MSVC picks the `CudaGraphExec` constructor.
* **Input/output transfers as kernel nodes.** The graph's H2D/D2H memcpy nodes are
  replaced by a tiny copy kernel over mapped pinned memory
  (`LC0EX_KERNEL_COPY=0` restores memcpy nodes). Copy-engine work runs in submission
  order, so the next batch's upload queued behind the previous batch's downloads.
* **`ordering=` backend option** (`event` default, `ticket`, `none`). `none` lets
  consecutive batches overlap: +1.2 % at stock 360 W, but ~-10 % once the card is
  limited by core power. `ticket` is a device-side ordering; its wait gives up after
  20 ms so it cannot deadlock.

## Builder changes (`builder/`, apply in order with `git am`)

* `LC0EX_CUTLASS_{QKV,OUTPROJ,FFN1,FFN2}=1`: CUTLASS fp16-accumulate encoder GEMMs.
  Tile sweep now uses random operands and long runs (zeroed operands draw less power
  and pick the wrong tiles on a power-capped card) plus sm_120 candidates.
* `LC0EX_TRITON_PIN=<json>`: pin Triton GEMM configs. Triton's 3-repetition cold
  autotune picks different tiles on every build (up to 45 % slower on one GEMM).
  `triton-pins-b84.json` holds the configs that measured best in-graph at batch 84.
* `LC0EX_CUTLASS_SHAPES` / `LC0EX_CUTLASS_MIN_M`, `LC0EX_CUTLASS_SWEEP_SET=small`.

Tried and rejected (not included): LayerNorm fused into the out-projection/FFN2
epilogue through an 8-CTA cluster (-6.6 %: the serial LN tail idles the tensor cores
and cluster co-scheduling stalls), Smolgen compress+dense1 fused and split over
squares (-1.4 %: shortens a chain that only waits for SMs held by QKV), a 2-heads-per-
program attention kernel, CUTLASS for every GEMM, microbenchmark-tuned Triton pins.

Recipe:

```bash
LC0EX_CUTLASS_QKV=1 LC0EX_CUTLASS_OUTPROJ=1 LC0EX_CUTLASS_FFN1=1 LC0EX_CUTLASS_FFN2=1 \
LC0EX_TRITON_PIN=triton-pins-b84.json LC0EX_CUTLASS_SWEEP_SET=small LC0EX_TRITON_CONFIG_REUSE=1 \
uv run --package lczero-triton lczero-triton graph --network BT4-tf13g-nes-it325-fp16.pb.gz \
    --output bt4.lc0ex --batch-size 32,36,40,42,44,48,56,64,72,76,80,84
```

(`cutlass_matmul.py` expects nvcc at `/usr/local/cuda-12.9` and CUTLASS headers at
`~/spsa/cutlass/include`.)

## Results (nps)

| | batch 84 | batch 40 |
|---|---:|---:|
| onnx-trt, stock 360 W | 8,529 | |
| branch as downloaded, stock 360 W | 9,704 | |
| this branch, stock 360 W | 10,360 | |
| onnx-trt, 550 W + OC | ~9,800 | |
| this branch, 550 W, sys-clock OC removed, undervolt (sustained) | 11,338 | **11,464** |

At 550 W the card is held by an internal core-power cap ("SW Power Cap" active, board
limit not): lc0ex's full-rate fp16 tensor work sits at ~2.7 GHz (b84) / ~3.0 GHz (b40).
Core undervolting helps; a system-clock overclock hurt batch 84.

**Best batch: 40** — ties 84 in short runs, slightly ahead sustained, and has none of
the occasional slow batches batch 84 shows. 42 is ~1.5 % behind 40.

## Int8 carrier nets (`bt4_carrier_R45`) — builder patches 0005-0007

A carrier net is the same BT4 net plus an embedded ONNX model with no graph that
carries, per encoder, `q1/{attn_in,attn_out,ffn_in}/{w,scale,bias,r}`:
symmetric int8 weights stored as uint8 bytes in [out, in] order, a per-output
FP32 `scale`, the FP32 layer `bias` and a per-input FP32 `r`:

    y[o] = scale[o] * sum_i clamp(round(x[i] * r[i]), +-127) * w[o, i] + bias[o]

`attn_in` is the fused QKV, `attn_out` the out-projection (with the residual
`alpha` folded into its scale and bias), `ffn_in` FFN1. FFN2 has no int8 copy.

`LC0EX_INT8=1` runs those three GEMMs on int8 tensor cores (Triton, int32
accumulator) and quantizes their inputs inside the LayerNorm / attention kernels
that produce them. `LC0EX_INT8_LAYERS=qkv,out,ffn1` picks a subset. The FP16
BT4 kernels are untouched; the existing FP16 artifacts also load carrier nets.

```bash
LC0EX_INT8=1 LC0EX_CUTLASS_QKV=1 LC0EX_CUTLASS_OUTPROJ=1 LC0EX_CUTLASS_FFN1=1 \
LC0EX_CUTLASS_FFN2=1 LC0EX_TRITON_PIN=triton-pins-b84.json \
uv run --package lczero-triton lczero-triton graph --network bt4_carrier_R45.pb.gz \
    --output bt4_carrier_int8.lc0ex --batch-size 84
```

Batch 84, 2 threads, cooled GPU, accuracy vs `cuda-fp16` on 2000 positions:

| int8 layers | nps | vs FP16 | policy KL mean | top-1 |
|---|---:|---:|---:|---:|
| none (FP16) | 10,240 | | 1.1e-5 | 99.80 % |
| qkv | 11,865 | +15.9 % | 2.4e-4 | 98.85 % |
| qkv, ffn1 | 12,654 | +23.6 % | 3.9e-4 | 98.05 % |
| qkv, out, ffn1 | **13,775** | **+34.5 %** | 6.0e-4 | 97.25 % |

The int8 kernels match the exact int8 arithmetic; the accuracy cost is the net's
quantization. Best batch for int8 is 84 (batch 40: 13,040, 42: 12,930).
