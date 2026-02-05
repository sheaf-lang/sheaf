---
hide:
  - toc
---

<div style="text-align: center; margin-bottom: 3rem;" class="intro-text">

<div style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 2rem;">
  <img src="images/sheaf-dark.png" alt="Sheaf" style="height: 5.5rem;">
  <h1 style="margin: 0; font-size: 4.5rem; font-weight: 500; color: black;">Sheaf</h1>
</div>

<h2>A Functional Language for Differentiable Programs</h2>

<p>Sheaf is a functional language for describing differentiable computation as topology rather than execution.<br>
It is inspired by Clojure, JIT-compiles through XLA, and integrates with the Python ecosystem.</p>

</div>

<div class="flex-section flex-section-header" markdown="1">

<div markdown="1">

## For ML Researchers

- **Low Boilerplate** — Functional parameter handling and lean syntax
- **Ecosystem Integration** — Seamless interoperability with Python and JAX
- **Interactive REPL** — Live evaluation and rapid architectural iteration
- **Runtime Observability** — Guards and traces expose tensor stability and failure modes
</div>

<div markdown="1">

## For Agentic AI

- **Context Density** — High logic-to-token ratio allows agents to fit complex architectures and reasoning chains within tight context limits
- **Uniform Syntax** — Single syntactic form for all operations reduces ambiguity and generation errors
- **Context Setup** — A dedicated context generator aligns agents with Sheaf semantics from the first prompt
</div>

</div>

## **Neural Networks as Math**

<div class="flex-section" markdown="1">
<div markdown="1">

Neural networks in Sheaf are written as mathematical transformations.

Layers, activations, and parameter bindings form explicit data flow, without imperative state management.

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
        (-> h   ;; 1. Self-Attention
            (layer-norm (get layer-p :ln1) 2)
            (multi-head-attention layer-p config)
            (first)  ;; Attention output
            (+ h))   ;; Residual 1

        (-> h   ;; 2. MLP
            (layer-norm (get layer-p :ln2) 2)
            (mlp (get layer-p :mlp))
            (+ h)))) ;; Residual 2
    ```

</div>

</div>

## **Models as Data**

<div class="flex-section" markdown="1">
<div markdown="1">

In PyTorch or JAX, changing a model's structure live requires careful orchestration of recompilation and parameter trees.

In Sheaf, model transformations like pruning, expansion or specialization are expressed as regular data operations. The model rebuilds itself through JIT-compiled functions that manipulate the parameter tree directly.

</div>

<div markdown="1">

```clojure

;; Model transformations are just data operations

(defn append-layer [params new-layer] :jit
  (-> params
    (get :layers)
    (append new-layer)
    (as-> new-layers (assoc params :layers new-layers))))


(defn hot-swap-head [model task-id heads] :jit
  (let [new-head (get heads task-id)]
    (assoc model :head new-head)))

```

</div>
</div>

## **Differentiable Logic**

<div class="flex-section" markdown="1">
<div markdown="1">

Sheaf expresses logical rules as pure functional transformations, so the logical pipeline remains JIT-compatible and fully differentiable.

</div>
<div markdown="1">

```clojure
;; Symbolic filtering: find elements matching P AND Q
;; Logic is expressed as a continuous mask

(defn neuro-unify [embeddings queries temp] :jit
  (let [scores (+ (@ embeddings (get queries :P))
                  (@ embeddings (get queries :Q)))
        mask (softmax (/ scores temp) :axis 0)]
    (-> embeddings
      (* (reshape mask '[-1 1]))
      (sum :axis 0))))
```

</div>
</div>

## **Macros as Architecture**

<div class="flex-section" markdown="1">
<div markdown="1">

Macros allow architectural patterns to be abstracted once and expanded at compile-time.
They operate on code as data, generating computation graphs before evaluation.

Meta-architectures can be defined without encoding complex structural logic through Python loops or classes.

</div>

<div markdown="1">

```clojure
;; Define a macro to templatize a model

(defmacro defmodel [name input-params & layers]
  `(defn ~name [~(first input-params) p]
     (as-> ~(first input-params) _
       ~@(map transform-layer layers))))


;; Derive a new model from it

(defmodel new-mlp [x]
  (layer :l1 (linear 128) :relu)
  (layer :l2 (linear 10)  :softmax))
```

</div></div>

## **Built-in Observability**

<div class="flex-section" markdown="1">
<div markdown="1">

Sheaf provides a tracer that exposes the computation graph as it executes.

In tracing mode, each function call records its inputs and outputs, including shapes, statistics, memory footprint, and execution time.

Tracing reflects the logical hierarchy of the program.
Data flow, intermediate transformations, and numerical behavior are directly observable.

</div>

<div markdown="1">

```css
│ ├─ [get-in]
│ │ ├─ → dict(keys:['emb', 'final_ln', 'head', 'layers'])
│ └─ ← f32[65x256] [μ:-2.44e-04 min:-2.08e-01 max:1.66e-01] (65.00KB) (1.0μs)
...
│ ├─ [reduce]
│ │ ├─ → f32[1x128x256] [μ:8.72e-04 min:-1.82e-01 max:2.05e-01] (128.00KB)
│ │ ├─ [transformer-block]
│ │ │ ├─ → f32[1x128x256] [μ:8.72e-04 min:-1.82e-01 max:2.05e-01] (128.00KB)
│ │ │ ├─ → dict(keys:['attn', 'ln1', 'ln2', 'mlp'])
│ │ │ ├─ → dict(keys:['d_model', 'n_layers', 'n_heads', 'batch_size', 'block_size', 'vocab_size', 'lr'])
│ │ │ ├─ [layer-norm]
│ │ │ │ ├─ → f32[1x128x256] [μ:8.72e-04 min:-1.82e-01 max:2.05e-01] (128.00KB)
│ │ │ │ ├─ → 2
│ │ │ │ ├─ [mean]
│ │ │ │ │ ├─ → f32[1x128x256] [μ:8.72e-04 min:-1.82e-01 max:2.05e-01] (128.00KB)
│ │ │ │ │ ├─ → axis: 2
│ │ │ │ │ ├─ → keepdims: True
│ │ │ │ └─ ← f32[1x128x1] [μ:8.72e-04 min:-2.32e-03 max:2.57e-03] (512B) (0.7μs)

```

</div>
</div>

## **Runtime Guards**

<div class="flex-section" markdown="1">
<div markdown="1">

Sheaf includes guard primitives to assert numerical and structural invariants during execution.

Guards halt execution the moment a tensor deviates from its expected bounds. They can detect NaNs, range violations or shape mismatches.

When a guard fails, Sheaf provides a traceback of the specific operations that induced the failure.

</div>
<div markdown="1">

```clojure
(defn train-step [p x y lr]
  (let ([loss grads] ((value-and-grad loss-fn) p))
    (guard :no-nan grads)
```

```css
/!\ Guard Breached: :no-nan
Value stats: f32[3] [μ:inf min:inf max:inf] [NaN DETECTED] (12B)

Backtrace (last 100 operations):

├─ [value-and-grad]
│ ├─ → <function LambdaForm.compile.<locals>.anonymous_func at 0x10d06bce0>
....
│ │ ├─ [loss-fn]
│ │ │ ├─ → 0.0
│ │ │ ├─ [/]
│ │ │ │ ├─ → 1.0
│ │ │ │ ├─ → f32[] [μ:0.00e+00 min:0.00e+00 max:0.00e+00] (4B)
│ │ │ └─ ← f32[] (4B) (2.3μs)
│ │ └─ ← tuple(len:2) (0.9μs)

```

</div>
</div>

## **Python Interoperability**

<div class="flex-section" markdown="1">
<div markdown="1">

Sheaf functions compile to pure JAX functions natively callable from Python.

Python provides data, execution context, and tooling, while Sheaf defines structure.

</div>

<div markdown="1">

```python
from sheaf import Sheaf

shf = Sheaf()

# Load Sheaf code from a string or a file object

shf.load("(defn add-five [x] (+ 5 x))")
result = shf.add_five(10)
```

</div>
</div>

---

<div style="text-align: center; margin-bottom: 3rem;" class="intro-text"; markdown="1">
[Get started](quickstart.md){ .md-button }
</div>
