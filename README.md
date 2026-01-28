<img width="479" height="154" alt="Screenshot 2026-01-27 at 22 55 15" src="https://github.com/user-attachments/assets/b79b4f7f-dc77-459e-a3a6-987bda4e70f2" />

Sheaf is a functional language for differentiable computation.
It is inspired by Clojure and is designed to operate within the Python/JAX machine learning ecosystem.

Sheaf has three main goals:

- Clojure Paradigm: Homoiconicity (code is data), immutability and minimalist syntax.
- Native XLA Performance: Sheaf code runs on GPUs/TPUs through JAX.
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

- [Website](http://sheaf-lang.org/)
- [Reference documentation](http://sheaf-lang.org/starting/)
- [Quick Start guide](http://sheaf-lang.org/reference/)
