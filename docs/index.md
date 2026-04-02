---
hide:
  - toc
---

<div style="text-align: center; margin-bottom: 3rem;" class="intro-text">

<div style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 2rem;">
  <img src="images/sheaf-dark.png" alt="Sheaf" style="height: 5.5rem;">
  <h1 style="margin: 0; font-size: 4.5rem; font-weight: 500; color: black;">Sheaf</h1>
</div>

<h2 style="max-width: 36rem; margin-left: auto; margin-right: auto; margin-top: 1rem;">A functional language for differentiable computation</h2>

<p  style="text-align: center; size: 1rem; max-width: 40rem; margin-left: auto; margin-right: auto; margin-top: 1rem;"  markdown="1">

Sheaf brings Clojure’s code-as-data to machine learning, with models as inspectable, composable, and compiled data structures.

</p>

</div>

<div class="flex-section flex-section-header" markdown="1">

<div markdown="1">

## For ML Researchers

- **No classes, no boilerplate** — Write math, not plumbing
- **Runtime Observability** — Catch NaN, trace shapes and profile performance without code changes
- **Single binary framework** — One executable, no dependencies. Train and run on GPU out of the box

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

In Sheaf, a neural network is a composition of mathematical functions over a parameter tree.

Sheaf is purely functional, so differentiation and compilation require no annotations. Any pure function can be differentiated with value-and-grad and is automatically compiled to GPU code.

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

Because models are data, Sheaf requires no module classes, registration, or parameter groups. Even structural operations like pruning, freezing, or weight sharing are expressed as regular data transformations.

Sheaf brings compile-time macros to the computation graph itself, generating architecture variants from a single template.

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

## **Observability**

<div class="flex-section" markdown="1">
<div markdown="1">

In Sheaf, every function call, tensor shape, and numerical statistic is observable at runtime.

A tracer logs the full call hierarchy with tensor statistics. Guards halt execution on numerical invariants like NaN or range violations. A profiler attributes wall time to each function in the call tree.

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
    ...
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
    Profiler: 3.63s wall

      Function                          Calls      Total       Self   Avg/call
      ------------------------------------------------------------------------
      gpt-forward                         100      1.72s      3.72s    37.23ms
      reshape                             301   900.57ms   900.57ms     2.99ms
      choice                              100   622.85ms   622.85ms     6.23ms
      softmax                             100   158.67ms   158.67ms     1.59ms
      generate-token                      100      5.56s   158.37ms    55.65ms
      io                                    4    45.11ms    45.11ms    11.28ms
      <lambda>                            102      5.58s    12.88ms    54.71ms
      ... 23 others                      1728                5.42ms

      Call tree:

      ├── generate (3.58s, 1 call)
      │   ├── reduce (3.58s, 1 call)
      │   │   └── <lambda> (3.58s, 101 calls)
      │   │       ├── generate-token (3.56s, 100 calls)
      │   │       │   ├── gpt-forward (1.72s, 100 calls)
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

A complete GPT-2 124M implementation in Sheaf is 1,908 tokens, while the equivalent PyTorch is 7,486. Sheaf's uniform syntax keeps the code concise and unambiguous.
<br/><br/>

Context usage counts GPT-4 tokens (tiktoken) across model, training, and sampling code. Deploy size is the minimal runtime required to train and run a model on a CUDA GPU.

</div>
<div>
<div>
<div style="margin-bottom: 1.5rem;">
  <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.4rem;"><span><strong>Code size</strong> (GPT-4 tokens)</span></div>
  <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.2rem;">
    <span style="font-size: 0.78rem; width: 80px; color: #666;">Sheaf</span>
    <div style="flex: 1; background: #e8e8e8; border-radius: 6px; height: 20px; overflow: hidden;"><div style="background: rgb(255, 110, 66); width: 24%; height: 100%; border-radius: 6px;"></div></div>
    <span style="font-size: 0.78rem; width: 80px; text-align: right;">1,908</span>
  </div>
  <div style="display: flex; align-items: center; gap: 0.8rem;">
    <span style="font-size: 0.78rem; width: 80px; color: #666;">PyTorch</span>
    <div style="flex: 1; background: #e8e8e8; border-radius: 6px; height: 20px; overflow: hidden;"><div style="background: #612156; width: 93%; height: 100%; border-radius: 6px;"></div></div>
    <span style="font-size: 0.78rem; width: 80px; text-align: right;">7,486</span>
  </div>
</div>
<div style="margin-bottom: 1.5rem;">
  <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.4rem;"><span><strong>Deploy size</strong></span></div>
  <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.2rem;">
    <span style="font-size: 0.78rem; width: 80px; color: #666;">Sheaf</span>
    <div style="flex: 1; background: #e8e8e8; border-radius: 6px; height: 20px; overflow: hidden;"><div style="background: rgb(255, 110, 66); width: 2%; height: 100%; border-radius: 6px; min-width: 6px;"></div></div>
    <span style="font-size: 0.78rem; width: 80px; text-align: right;">4 MB</span>
  </div>
  <div style="display: flex; align-items: center; gap: 0.8rem;">
    <span style="font-size: 0.78rem; width: 80px; color: #666;">PyTorch</span>
    <div style="flex: 1; background: #e8e8e8; border-radius: 6px; height: 20px; overflow: hidden;"><div style="background: #612156; width: 98%; height: 100%; border-radius: 6px;"></div></div>
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

The compiler toolchain is downloaded on first use and is not required to run a compiled model.

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

<div style="text-align: center; margin-bottom: 3rem;" class="intro-text"; markdown="1">
[Get started](quickstart.md){ .md-button }
</div>
