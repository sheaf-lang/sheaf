# Key Concepts

Sheaf is a Lisp dialect for machine learning that compiles to native code. Programs are lowered to [StableHLO](https://openxla.org/stablehlo), the same intermediate representation used by JAX and TensorFlow, then compiled and executed via [IREE](https://iree.dev). The entire framework ships as a standalone lightweight binary that runs training and inference on GPU.

This page describes the concepts behind the system: what happens when you run a program, how differentiation works, and how the runtime manages GPU execution.

## Design Principles

Sheaf follows a small set of principles that inform every design decision:

**Pure functions compile automatically.** Any function that takes tensors and returns tensors, without side effects, is eligible for GPU compilation. There are no `@jit` decorator nor `torch.compile()`. The compiler decides what to compile based on purity analysis. Functions that perform I/O or print are interpreted; everything else is compiled.

**No Python in the execution path.** The compiler, runtime, and IREE execution engine are statically linked into a single binary. Models ship as `.shf` source files alongside cached `.vmfb` compiled artifacts. A standalone deployment consists of the Sheaf binary, the model files (code and data), and its compiled artefacts.

**Functional PRNG.** Random number generation follows the JAX model: keys are explicit values, not implicit global state. `(random-key 42)` produces a key; `(random-split key n)` derives sub-keys deterministically. This makes random operations pure (key in, result out), which means they compile to GPU code like any other operation.

## Parameter Trees

Neural network parameters in Sheaf are plain nested dicts:

```clojure
(defn init-params [key]
  (let [[k1 k2] (random-split key 2)]
    {:layer1 {:W (/ (random-normal k1 '[784 256]) (sqrt 784))
              :b (zeros '[256])}
     :layer2 {:W (/ (random-normal k2 '[256 10]) (sqrt 256))
              :b (zeros '[10])}}))
```

When a dict is passed to a compiled function, the compiler flattens it to positional tensor arguments with a stable field ordering. A dict `{:W tensor<768x768xf32>, :b tensor<768xf32>}` becomes two MLIR arguments `(%arg0: tensor<768x768xf32>, %arg1: tensor<768xf32>)`. The field-to-index mapping is recorded in a manifest so the runtime can reconstruct the dict on output.

This flattening is recursive: nested dicts like `{:transformer {:h [{:attn ...} {:attn ...}]}}` produce a flat sequence of tensor arguments ordered by depth-first traversal. The user never sees this. `(get params :W)` and `(get-in params [:transformer :h])` work identically whether the function is interpreted or compiled.

For a complete reference on dict operations, see [Dictionary operations](reference.md#dictionary-operations).

## Automatic Differentiation

Sheaf implements reverse-mode automatic differentiation at the expression level. When you write `(value-and-grad f)`, the compiler generates a single function that computes both the forward pass and the parameter gradients in one execution:

```clojure
(defn loss [params x y]
  (mean (** (- (@ x (get params :W)) y) 2)))

;; Sample data
(let [params {:W (random-normal (random-key 0) '[4 1])}
      x      (random-normal (random-key 1) '[8 4])
      y      (random-normal (random-key 2) '[8 1])
      ;; Compute loss and gradients in a single call
      ;; value-and-grad differentiates w.r.t. the first argument (params)
      [loss grads] ((value-and-grad (fn [p] (loss p x y))) params)]
  (print "loss:" loss)
  (print "dW shape:" (shape (get grads :W))))
```

The gradients have the same structure as the input parameters: if `params` is a nested dict, `grads` is a nested dict with identical keys and shapes.

### How It Works

Before differentiation, the forward expression is converted to Administrative Normal Form (ANF), where every intermediate computation is bound to a name:

```clojure
;; Original
(mean (** (- (@ x W) y) 2))

;; ANF: flat sequence of named bindings
(let [a0 (@ x W)
      a1 (- a0 y)
      a2 (** a1 2)
      a3 (mean a2)]
  a3)
```

The reverse-mode pass then walks these bindings backward, accumulating adjoint contributions. For each binding `v = f(a, b)`, it computes `dL/da += dL/dv * df/da`. Both forward and backward bindings are fused into a single MLIR function. Intermediate activations computed in the forward pass are directly available for the backward pass without recomputation.

### Differentiable Operations

| Category    | Operations                                                  |
| ----------- | ----------------------------------------------------------- |
| Arithmetic  | `+`, `-`, `*`, `/`, `**`, `neg`                             |
| Matrix      | `@` (matmul), `einsum`, `transpose`                         |
| Shape       | `reshape`, `swapaxes`, `slice`                              |
| Activations | `relu`, `gelu`, `sigmoid`, `tanh`, `softmax`, `log-softmax` |
| Reductions  | `sum`, `mean`, `var`                                        |
| Elementwise | `exp`, `log`, `sqrt`, `abs`                                 |
| Control     | `where`, `maximum`, `minimum`                               |
| Iteration   | `reduce`, `scan`                                            |

`reduce` and `scan` are unrolled before differentiation when the collection size is statically known. This enables differentiating through transformer layers expressed as `(reduce block x layers)`.

### Structural Lowering

Some operations are structural rather than numerical. Before differentiation, they are lowered to pure tensor operations:

- **Dict access**: `(get params :W)` becomes `GetTupleElement(params, [0])` with a static index
- **Iteration**: `(reduce f init items)` is unrolled to a chain of let-bindings when the number of items is known at compile time

After lowering, the expression contains only tensor operations that the reverse-mode pass can differentiate.

## Compilation Pipeline

When you call a function, the compiler transforms it through several intermediate representations:

```
Source (.shf)
    |
 [Parse]        ->  SheafValue AST
    |
 [Compile]      ->  CompiledExpr IR (macros expanded, symbols resolved)
    |
 [Codegen]      ->  StableHLO MLIR (typed, shape-inferred)
    |
 [iree-compile] ->  VMFB bytecode (fused kernels, memory planned)
    |
 [IREE Runtime] ->  CUDA / Metal / Vulkan / CPU execution
```

### JIT Compilation

Functions are compiled transparently on first call. The runtime detects when a function lacks compiled code and triggers compilation automatically:

1. **Type inference** - examines runtime arguments to determine tensor shapes
2. **Dict lowering** - converts dict parameters to tuples with stable field ordering
3. **Inlining** - expands calls to user-defined functions into the caller's body
4. **Codegen** - generates StableHLO MLIR from the inlined expression
5. **Compilation** - invokes `iree-compile` with the appropriate backend flags
6. **Caching** - stores the resulting VMFB by content hash

Subsequent calls with the same argument shapes dispatch directly to the cached VMFB. If the source code changes, the content hash changes, and the function is recompiled automatically.

### Backend Selection

Sheaf probes available hardware in preference order: CUDA, Metal, Vulkan, CPU. The first available backend is used.

You can override this with the `--device` flag:

```bash
sheaf model.shf --device cuda     # Force CUDA backend
sheaf model.shf --device metal    # Force Metal backend
```

### Cache Layout

Compiled artifacts are cached in the `__sheaf__` directory alongside your source files:

```
__sheaf__/
  manifest.json           # Maps function names to content hashes
  forward.cuda.vmfb       # Compiled forward pass (CUDA backend)
  loss-vag.cuda.vmfb      # Compiled value-and-gradient closure
```

The manifest tracks content hashes to detect staleness. Delete the `__sheaf__` directory to force recompilation.

A model can be distributed as `.shf` source files alongside its `__sheaf__/` cache. The recipient runs training or inference with the Sheaf binary alone; the JIT compiler toolchain is only needed if the source changes. This makes on-device deployment a file copy.

### StableHLO Code Generation

The code generator translates the internal IR to StableHLO MLIR. A Sheaf function:

```clojure
(defn forward [params x]
  (+ (@ x (get params :W)) (get params :b)))
```

produces typed MLIR with explicit broadcasting and flattened dict arguments:

```mlir
func.func @forward(%arg0: tensor<768x768xf32>,   // params[:W]
                   %arg1: tensor<768xf32>,       // params[:b]
                   %arg2: tensor<4x768xf32>)     // x
    -> tensor<4x768xf32> {
  %0 = stablehlo.dot_general %arg2, %arg0,
       contracting_dims = [1] x [0]
       : (tensor<4x768xf32>, tensor<768x768xf32>) -> tensor<4x768xf32>
  %1 = stablehlo.broadcast_in_dim %arg1, dims = [1]
       : (tensor<768xf32>) -> tensor<4x768xf32>
  %2 = stablehlo.add %0, %1 : tensor<4x768xf32>
  return %2 : tensor<4x768xf32>
}
```

Scalar values and dict fields known at compile time are propagated through the expression. Shape-dependent operations like `reshape` and `slice` require static dimensions. If a value cannot be resolved at compile time, compilation fails with an explicit error.

Dict and tuple parameters use virtual tuples: internal bookkeeping that tracks which MLIR arguments correspond to which fields, without materializing `stablehlo.tuple` operations (which are being phased out of StableHLO).

## Observability

Sheaf includes built-in instrumentation for understanding program behavior. These tools are available as CLI flags with zero overhead when not enabled.

### Profiling (`--blame`)

```bash
sheaf train.shf --blame
```

Prints a hierarchical timing report showing wall time, per-function self time, call counts, and a call tree. This is useful for identifying which functions dominate execution time and whether they are compiled or interpreted.

### Call Tracing (`--trace`)

```bash
sheaf train.shf --trace                 # Trace all functions
sheaf train.shf --trace forward,loss    # Trace specific functions
```

The tracer logs every function call with argument shapes, return shapes, tensor statistics, and timing.

### Runtime Guards (`--guard`)

```bash
sheaf train.shf --guard no-nan      # Halt on any NaN or Inf
```

Guards check tensor values at function boundaries without modifying the program. When a guard triggers, Sheaf reports the function, the tensor, and a backtrace of the call hierarchy.

## IREE Runtime

The runtime interfaces with [IREE](https://iree.dev) via FFI. IREE loads compiled VMFBs and dispatches them to the appropriate hardware backend (CUDA, Metal, Vulkan, or CPU).

### Device Memory Management

Tensor data lives on the host by default. When a tensor is passed to a compiled function, it is transferred to device memory. The `DeviceBuffer` type represents tensors that remain on device between calls, avoiding repeated host-to-device transfers in training loops:

```
Step 1: params (host) -> DeviceBuffer (H2D transfer)
Step 2: DeviceBuffer -> forward -> DeviceBuffer (no transfer)
Step 3: DeviceBuffer -> backward -> DeviceBuffer (no transfer)
...
Step N: DeviceBuffer -> host (D2H transfer, only for checkpointing)
```

This process is automatic: the runtime tracks which tensors are already on device and skips redundant transfers.

### Buffer Cache

Input tensors are cached by pointer identity (`Arc::ptr_eq`): if the same tensor is passed to the same argument position, the runtime reuses the existing device buffer. This is O(1) with no false positives, and handles the common case where model parameters do not change between training steps.

Each argument position holds up to 8 cached buffers with most-recently-used ordering. This prevents thrashing when transformer layers share a compiled function but pass different weight tensors.

## Known Limitations

- **Shape specialization.** Functions are compiled for the tensor shapes seen at the first call. A different shape (e.g., a different batch size) triggers automatic recompilation and caching. This is the same tracing model as `jax.jit`.
- **Control flow.** `if` expressions inside compiled functions cause fallback to the interpreter. Conditional logic must live outside the hot path, or use `where`, which compiles to `stablehlo.select`.
- **Backend maturity.** IREE backend performance varies by hardware. Benchmarks on your target device are recommended.
- **Memory management.** No automatic model sharding or gradient checkpointing. Large models may exceed device memory.
