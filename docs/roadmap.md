---
hide:
  - toc
---

# Roadmap

These are the planned features for the next releases of Sheaf.

### Language

- **`def` for global constants**: Immutable top-level bindings, as in Clojure. While not required per-se, it eliminates the need to pass configuration dictionaries everywhere.

- **`loop` / `recur`**: Explicit tail-recursive loops. Sheaf uses `repeat`, but `loop` and `recur` are more natural for someone coming from Clojure.

- **`reverse` / `flip`**: Reverse a tensor along an axis. Currently requires manual index construction.

- **`stack`**: Combine multiple tensors into a new dimension.

- **`inc` / `dec`**: Small convenience to increment and decrement, as in Clojure. Currently `(+ var 1)` and `(- var 1)`.

- **`argmax` returns integers**: `argmax` and `argmin` currently return floats. They will return integer tensors for direct use as indices.

### Macros

- **Enriched `defmacro`**: `range` and `reduce` available at compile time, enabling macros that generate architecture variants from a single template.

### Operations

- **Convolution primitives**: `conv1d` and `conv2d` via `stablehlo.convolution`, exposed through the standard library.

- **`vmap` on dictionaries**: `vmap` currently only accepts tensor arguments. PyTree support (automatic flattening/unflattening of dicts) should be added to match the behavior of `value-and-grad`.

### Autodiff

- **Gradient checkpointing**: Recompute forward activations during the backward pass instead of storing them all. This will reduce memory usage for deep models (the GPT-2 124M training currently uses 13 GB).

- **Scalar parameters in `value-and-grad`**: Float scalars in parameter dictionaries (e.g., `{:w 5.0}`) will produce correct gradients. Currently requires wrapping in a 1-element tensor.

### Performance

- **KV cache helpers**: A `cached-attention` stdlib function to simplify implementing KV cache in transformer models.

- **Batch generation mode**: Compile the full autoregressive generation loop into a single dispatch, returning all tokens at once.

### Developer experience

- **Error call stack propagation**: When an error occurs inside a stdlib function, the error message will show the user's call site, not the stdlib internals.

- **Jupyter integration**: A Sheaf kernel for Jupyter, allowing interactive notebook workflows with inline tensor visualization and training loops.

- **`:trace` and `:blame` in the REPL**: These observability modes were tied to V1 semantics and temporarily removed. They will be re-introduced with behavior adapted to the V2 execution model.

### Distribution

- **NCCL all-reduce**: Multi-GPU training via NCCL collective operations, for data-parallel training across multiple devices.
