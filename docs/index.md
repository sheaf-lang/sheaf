---
hide:
  - navigation
  - toc
  - footer
---

<div class="hero" markdown="1">

<div class="hero-title">
  <img src="images/sheaf-dark.png#only-light" alt="">
  <img src="images/sheaf-light.png#only-dark" alt="">
  <h1>Sheaf</h1>
</div>

<h2>A functional language for differentiable computation</h2>

<p markdown="1">

Sheaf brings Clojure’s code-as-data to machine learning, with models as
inspectable, composable, and compiled data structures.

</p>

<div class="hero-actions" markdown="1">
[Get started](quickstart.md){ .md-button .md-button--primary }
[GitHub](https://github.com/sheaf-lang/sheaf){ .md-button }
</div>

</div>

<div class="flex-section flex-section-header" markdown="1">

<div markdown="1">

## For ML Researchers

- **No classes, no boilerplate** — Write math, not plumbing
- **Runtime Observability** — Catch NaN, trace shapes and profile performance
  without code changes
- **Single binary framework** — One executable, no dependencies. Train and run
  on GPU out of the box

</div>

<div markdown="1">

## For Agentic AI

- **Context Density** — 60-75% fewer tokens than equivalent Python for the same
  architecture
- **Uniform Syntax** — Single syntactic form for all operations reduces
  ambiguity and generation errors
- **Immediate Onboarding** — Built-in context generator for Claude Code, Cursor,
  and Copilot

</div>

</div>

## Neural Networks as Math

<div class="flex-section" markdown="1">
<div markdown="1">

In Sheaf, a neural network is a composition of mathematical functions over a
parameter tree.

Sheaf's functional core makes differentiation and compilation possible without
graph annotations. Supported tensor functions can be differentiated with
value-and-grad and compiled automatically for CPU or GPU.

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

## Models as Data

<div class="flex-section" markdown="1">
<div markdown="1">

Because models are data, Sheaf requires no module classes, registration, or
parameter groups. Even structural operations like pruning, freezing, or weight
sharing are expressed as regular data transformations.

Sheaf brings compile-time macros to the computation graph itself, generating
architecture variants from a single template.

</div>

<div markdown="1">

```clojure
;; Grow a model: add a layer at runtime
(defn append-layer [params new-layer]
  (assoc params :layers
    (append (get params :layers) new-layer)))

;; Swap the output head for a different task
(defn hot-swap-head [model task-id heads]
  (assoc model :head (get heads task-id)))
```

</div>
</div>

## Observability

<div class="flex-section" markdown="1">
<div markdown="1">

In Sheaf, every function call, tensor shape, and numerical statistic is
observable at runtime.

A tracer logs the full call hierarchy with tensor statistics. Guards halt
execution on numerical invariants like NaN or range violations. A profiler
attributes wall time to each function in the call tree.

</div>

<div markdown="1">

=== "Tracer"

    ```text
    ├─ [train-step] dict(keys:["l1", "l2"]), f32[4x2] [min:0.00e0 max:1.00e0] (32B), f32[4x1] [min:0.00e0 max:1.00e0] (16B), 0.700000
    │ ├─ [forward] f32[4x2] [min:0.00e0 max:1.00e0] (32B), dict(keys:["l1", "l2"])
    │ │ ├─ [relu] f32[4x8] [min:-1.37e0 max:2.33e0] (128B)
    │ │ └─ ← f32[4x8] [min:0.00e0 max:2.33e0] (128B) (0.8μs)
    │ │ ├─ [sigmoid] f32[4x1] [min:-5.48e-2 max:1.18e0] (16B)
    │ │ └─ ← f32[4x1] [min:4.86e-1 max:7.66e-1] (16B) (1.8μs)
    │ └─ ← f32[4x1] [min:4.86e-1 max:7.66e-1] (16B) (0.0μs)
    ...
    ```

=== "Guards"

    ```text
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

    ```text
    Profiler: 3.78s wall

    Function                          Calls      Total       Self   Avg/call
     ------------------------------------------------------------------------
     gpt-forward                         500      3.17s      3.17s     6.35ms
     softmax                             500   184.70ms   184.70ms    369.4μs
     slide-window                        500   142.79ms   142.79ms    285.6μs
     reshape                             501   109.23ms   109.23ms    218.0μs
     choice                              500    92.11ms    92.11ms    184.2μs
     generate-token                      500      3.59s    28.62ms     7.18ms
     io                                    5    26.76ms    26.76ms     5.35ms
     <lambda>                            502      3.75s    11.83ms     7.46ms
     ... 21 others                      7527                4.86ms

     Call tree:

     ├── generate (3.75s, 1 call)
     │   ├── reduce (3.75s, 1 call)
     │   │   └── <lambda> (3.75s, 501 calls)
     │   │       ├── generate-token (3.59s, 500 calls)
     │   │       │   ├── gpt-forward (3.17s, 500 calls)
     │   │       │   ├── softmax (184.70ms, 500 calls)
     │   │       │   ├── reshape (109.22ms, 500 calls)
     │   │       │   ├── choice (92.11ms, 500 calls)
     │   │       │   └── ... 7 others (1.69ms, 4000 calls)
     │   │       ├── slide-window (142.79ms, 500 calls)
     │   │       └── ... 4 others (2.04ms, 3502 calls)
     │   └── ... 2 others (1.5μs, 2 calls)
     └── ... 7 others (26.86ms, 21 calls)
    ```

</div>
</div>

## Compact by Construction

<div class="flex-section">
<div markdown="1">

The complete NanoGPT implementation in Sheaf is 2,542 tokens, while the
equivalent PyTorch is 7,486 — a 66% context reduction. Sheaf's uniform syntax
keeps the code concise and unambiguous.
<br/><br/>

Context usage counts GPT-4 tokens (tiktoken) across model, training, and
sampling code. Deploy size is the minimal runtime required to train and run a
model on a CUDA GPU.

</div>
<div class="bar-chart">
  <div class="bar-group">
    <div class="bar-title"><strong>Code size</strong> (GPT-4 tokens)</div>
    <div class="bar-row">
      <span class="bar-label">Sheaf</span>
      <div class="bar-track"><div class="bar bar--sheaf" style="width: 34%;"></div></div>
      <span class="bar-value">2,542</span>
    </div>
    <div class="bar-row">
      <span class="bar-label">PyTorch</span>
      <div class="bar-track"><div class="bar bar--pytorch" style="width: 100%;"></div></div>
      <span class="bar-value">7,486</span>
    </div>
  </div>
  <div class="bar-group">
    <div class="bar-title"><strong>Deploy size</strong></div>
    <div class="bar-row">
      <span class="bar-label">Sheaf</span>
      <div class="bar-track"><div class="bar bar--sheaf" style="width: 0.2%;"></div></div>
      <span class="bar-value">4 MB</span>
    </div>
    <div class="bar-row">
      <span class="bar-label">PyTorch</span>
      <div class="bar-track"><div class="bar bar--pytorch" style="width: 100%;"></div></div>
      <span class="bar-value">~2.4 GB</span>
    </div>
  </div>
  <div class="bar-note">GPT-2 124M · model + training loop + sampler · token count via tiktoken · Sheaf binary includes GPU runtime.</div>
</div>
</div>

## Native Runtime

<div class="flex-section" markdown="1">
<div markdown="1">

Sheaf is written in Rust. The complete runtime with GPU backends ships as a
single 4 MB executable.

The compiler toolchain is downloaded on first use and is not required to run a
compiled model.

</div>

<div markdown="1">

```
# Standalone, self-contained deployment
$ du -h *
128K	__sheaf__                 # compiled model
3.2M	data
4.0K	model.shf
164M	out-weights
3.8M	sheaf                     # runtime
4.0K	train.shf
```

</div>
</div>

---

<div class="hero-actions" markdown="1">
[Get started](quickstart.md){ .md-button .md-button--primary }
</div>
