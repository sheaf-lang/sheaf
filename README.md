<img width="487" height="146" alt="Screenshot 2026-01-24 at 17 33 42" src="https://github.com/user-attachments/assets/32d9a66b-2a78-44eb-8970-e17f19fc28fe" />

Sheaf is a functional language for differentiable computation.
It is inspired by Clojure and is designed to operate within the Python/JAX machine learning ecosystem.

Sheaf has three main goals:

- Clojure Paradigm: Homoiconicity (code is data), immutability and minimalist syntax for differentiable computation.
- Native XLA Performance: Sheaf code runs on GPUs/TPUs through JAX. Models can self-evolve without stopping the JIT engine.
- Seamless Interop with Python: Compiled to pure JAX functions natively callable from Python.

### Sample code:

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

### Python interoperability

```python
from sheaf import Sheaf
shf = Sheaf("mlp.shf")

for x, y in zip(X, shf.forward(X, params)):
    print(f"  {x} -> {y[0]:.3f}")
```

- Website

- Reference documentation

- Quick Start guide
