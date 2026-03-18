---
hide:
  - toc
---

<div style="text-align: center; margin-bottom: 3rem;" class="intro-text">

<div style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 2rem;">
  <img src="images/sheaf-dark.png" alt="Sheaf" style="height: 5.5rem;">
  <h1 style="margin: 0; font-size: 4.5rem; font-weight: 500; color: black;">Sheaf</h1>
</div>

<h2>A Functional Language for Differentiable Computation</h2>

<p>Inspired by Clojure. Designed from the ground up for machine learning. Compiles to native GPU code.</p>

</div>

<div class="flex-section flex-section-header" markdown="1">

<div markdown="1">

## For ML Researchers

- **Single binary framework** — Models train and run on GPU with no environment to configure
- **Functional parameters** — Models are data structures that can be inspected, transformed, and composed
- **Runtime Observability** — Guards and traces expose tensor stability and failure modes

</div>

<div markdown="1">

## For Agentic AI

- **Context Density** — 60-75% fewer tokens than equivalent Python for the same architecture
- **Uniform Syntax** — Single syntactic form for all operations reduces ambiguity and generation errors
- **Immediate Onboarding** — Built-in context generator for Claude Code, Cursor, and Copilot

</div>

</div>

## **Neural Networks as Math**

<div class="flex-section" markdown="1">
<div markdown="1">

Neural networks in Sheaf are written as mathematical transformations.

Layers, activations, and parameter bindings form explicit data flow, without imperative state management.

Because the language is functionally pure, compilation and differentiation require no decorators or annotations. `value-and-grad` differentiates any pure function; the JIT resolves shapes and compiles to GPU code automatically.

</div>
<div markdown="1">

=== "Atomic"

    ```clojure
    (defn forward [x p]
      (as-> x h
        (with-params [p :l1] (relu    (+ (@ h W) b)))
        (with-params [p :l2] (softmax (+ (@ h W) b)))))








    ```

=== "Composite"

    ```clojure
    (defn transformer-block [x layer-p config]
      (as-> x h
        (-> h
            (layer-norm (get layer-p :ln1) 2)
            (multi-head-attention layer-p config)
            (first)
            (+ h))   ;; residual

        (-> h
            (layer-norm (get layer-p :ln2) 2)
            (mlp (get layer-p :mlp))
            (+ h))))

    ```

</div>

</div>

## **Models as Data**

<div class="flex-section" markdown="1">
<div markdown="1">

In Sheaf, model parameters are nested dictionaries.

There are no module classes, or registration, or parameter groups. Structural operations like pruning, freezing, or weight sharing are expressed as regular data transformations.

Compile-time macros generate architecture variants from a single template. The same primitives extend to neuro-symbolic pipelines where logic and learning are jointly differentiable.

</div>

<div markdown="1">

```clojure
;; Freeze a layer: zero out its gradients
(defn freeze [grad layer-key]
  (assoc grad layer-key
    (tree-map (fn [g] (zeros (shape g)))
              (get grad layer-key))))

;; Apply weight decay to all parameters at once
(defn weight-decay [params rate]
  (tree-map (fn [w] (* w (- 1.0 rate))) params))

```

</div>
</div>

## **Observability**

<div class="flex-section" markdown="1">
<div markdown="1">

Every function call, tensor shape, and numerical statistic is observable at runtime.

Three modes expose different aspects of execution: a tracer reconstructs the full call hierarchy with tensor statistics, guards halt on numerical invariants like NaN or range violations, and a profiler attributes wall time to each function in the call tree.

</div>

<div markdown="1">

=== "Tracer"

    ```css
    ├─ [train-step] dict(keys:["l1", "l2"]), f32[4x2] [min:0.00e0 max:1.00e0] (32B), f32[4x1] [min:0.00e0 max:1.00e0] (16B), 0.700000
    │ ├─ [forward] f32[4x2] [min:0.00e0 max:1.00e0] (32B), dict(keys:["l1", "l2"])
    │ │ ├─ [relu] f32[4x8] [min:-1.37e0 max:2.33e0] (128B)
    │ │ └─ ← f32[4x8] [min:0.00e0 max:2.33e0] (128B) (0.8μs)
    │ │ ├─ [sigmoid] f32[4x1] [min:-5.48e-2 max:1.18e0] (16B)
    │ │ └─ ← f32[4x1] [min:4.86e-1 max:7.66e-1] (16B) (1.8μs)
    │ └─ ← f32[4x1] [min:4.86e-1 max:7.66e-1] (16B) (0.0μs)
    │ ├─ [sgd-step] dict(keys:["l1", "l2"]), dict(keys:["l1", "l2"]), 0.700000
    │ └─ ← dict(keys:["l1", "l2"]) (8.9μs)
    └─ ← dict(keys:["loss", "p"]) (0.0μs)
    ```

=== "Guards"

    ```css
    $ sheaf train.shf --guard no-nan
    Step 1 | Loss: 0.306990
    Step 2 | Loss: 0.500000

    /!\ Guard Breached: NoNan
    Function: sigmoid
    Tensor contains NaN or Inf values: f32[4x1] [min:inf max:-inf]

    Backtrace (last 26 operations):

    ├─ [train-step] dict(keys:["l1", "l2"]), f32[4x2], f32[4x1], 1000.0
    │ ├─ [forward] f32[4x2], dict(keys:["l1", "l2"])
    │ │ ├─ [relu] f32[4x8] [min:-2.67e0 max:1.73e0]
    │ │ └─ ← f32[4x8] [min:0.00e0 max:1.73e0] (0.6μs)
    │ │ ├─ [sigmoid] f32[4x1] [min:inf max:-inf] [NaN DETECTED]
    ...
    ```

=== "Profiler"

    ```css
    Profiler: 5.63s wall

      Function                          Calls      Total       Self   Avg/call
      ------------------------------------------------------------------------
      gpt-forward                         100      3.72s      3.72s    37.23ms
      reshape                             301   900.57ms   900.57ms     2.99ms
      choice                              100   622.85ms   622.85ms     6.23ms
      softmax                             100   158.67ms   158.67ms     1.59ms
      generate-token                      100      5.56s   158.37ms    55.65ms
      io                                    4    45.11ms    45.11ms    11.28ms
      <lambda>                            102      5.58s    12.88ms    54.71ms
      ... 23 others                      1728                5.42ms

      Call tree:

      ├── generate (5.58s, 1 call)
      │   ├── reduce (5.58s, 1 call)
      │   │   └── <lambda> (5.58s, 101 calls)
      │   │       ├── generate-token (5.56s, 100 calls)
      │   │       │   ├── gpt-forward (3.72s, 100 calls)
      │   │       │   ├── reshape (900.16ms, 100 calls)
      │   │       │   ├── choice (622.85ms, 100 calls)
      │   │       │   ├── softmax (158.67ms, 100 calls)
      │   │       │   └── ... 8 others (1.47ms, 900 calls)
      │   │       └── ... 5 others (3.37ms, 1002 calls)
      │   └── ... 2 others (1.9μs, 2 calls)
      └── ... 7 others (45.64ms, 19 calls)
    ```

</div>
</div>

## **Resource Efficiency**

<div class="flex-section">
<div markdown="1">

A complete GPT-2 124M implementation in Sheaf is 1,908 tokens, while the equivalent PyTorch is 7,486. Sheaf's uniform syntax keeps the code concise and close to the math.
<br/><br/>

Context usage counts GPT-4 tokens (tiktoken) across model, training, and sampling code. Deploy size is the minimal runtime required to train and run a model on a CUDA GPU.

</div>
<div>
<div>
<div style="margin-bottom: 1.5rem;">
  <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.4rem;"><span><strong>Code size</strong> (GPT-4 tokens)</span></div>
  <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.2rem;">
    <span style="font-size: 0.78rem; width: 80px; color: #666;">Sheaf</span>
    <div style="flex: 1; background: #e8e8e8; border-radius: 6px; height: 10px; overflow: hidden;"><div style="background: linear-gradient(90deg, #612156, #8a3a7a); width: 25%; height: 100%; border-radius: 6px;"></div></div>
    <span style="font-size: 0.78rem; width: 80px; text-align: right;">1,908</span>
  </div>
  <div style="display: flex; align-items: center; gap: 0.8rem;">
    <span style="font-size: 0.78rem; width: 80px; color: #666;">PyTorch</span>
    <div style="flex: 1; background: #e8e8e8; border-radius: 6px; height: 10px; overflow: hidden;"><div style="background: #c0c0c0; width: 100%; height: 100%; border-radius: 6px;"></div></div>
    <span style="font-size: 0.78rem; width: 80px; text-align: right;">7,486</span>
  </div>
</div>
<div style="margin-bottom: 1.5rem;">
  <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.4rem;"><span><strong>Deploy size</strong></span></div>
  <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.2rem;">
    <span style="font-size: 0.78rem; width: 80px; color: #666;">Sheaf</span>
    <div style="flex: 1; background: #e8e8e8; border-radius: 6px; height: 10px; overflow: hidden;"><div style="background: linear-gradient(90deg, #612156, #8a3a7a); width: 2%; height: 100%; border-radius: 6px; min-width: 6px;"></div></div>
    <span style="font-size: 0.78rem; width: 80px; text-align: right;">4 MB</span>
  </div>
  <div style="display: flex; align-items: center; gap: 0.8rem;">
    <span style="font-size: 0.78rem; width: 80px; color: #666;">PyTorch</span>
    <div style="flex: 1; background: #e8e8e8; border-radius: 6px; height: 10px; overflow: hidden;"><div style="background: #c0c0c0; width: 100%; height: 100%; border-radius: 6px;"></div></div>
    <span style="font-size: 0.78rem; width: 80px; text-align: right;">~2.4 GB</span>
  </div>
</div>
<div style="font-size: 0.7rem; color: #888; margin-top: 0.5rem;">GPT-2 124M · model + training loop + sampler · token count via tiktoken · Sheaf binary includes GPU runtime.</div>
</div>
</div>
</div>

## **Clean Implementation**

<div class="flex-section" markdown="1">
<div markdown="1">

Sheaf is written in Rust. The complete runtime with GPU backends ships
as a single 4 MB executable.

Sheaf code is JIT-compiled to optimized GPU kernels for CUDA and Metal through StableHLO and IREE.

The compiler toolchain is downloaded on first use and is not required to run a compiled model.

</div>

<div markdown="1">

```
$ du -h * # Standalone, self-contained deployment
128K	__sheaf__                 # compiled model
3.2M	data
4.0K	model.shf
164M	out-shakespeare-char
3.6M	sheaf                     # runtime
4.0K	train.shf

$ ./sheaf train.shf
Loading model...
Loaded: 6 layers, 384 dim, 65 vocab
Training data: 1115394 tokens
Training for 100 steps (batch_size=4 block_size=256)...
Step 100 | Loss: 1.4573
```

</div>
</div>

---

<div style="text-align: center; margin-bottom: 3rem;" class="intro-text"; markdown="1">
[Get started](quickstart.md){ .md-button }
</div>
