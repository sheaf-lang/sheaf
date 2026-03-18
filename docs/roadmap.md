---
hide:
  - toc
---

# Roadmap

## Version 2.1 (planned features)

### Performance

- **KV cache**: Sheaf 2.0 currently doesn't have a KV cache, which is essential for competitive inference at scale. Without it, each generated token recomputes full O(n^2) attention over the entire context. With it, each step costs O(n) after the first pass.

- **Batch generation mode**: Currently, autoregressive generation calls the compiled model once per token. Batch mode would compile the full generation loop into a single dispatch, returning all tokens at once.

### Distribution

- **NCCL all-reduce**: Multi-GPU training via NCCL collective operations, for data-parallel training across multiple devices.

### Developer experience

- **Jupyter integration**: A Sheaf kernel for Jupyter, allowing interactive notebook workflows with inline tensor visualization and training loops.

- **`:trace` and `:blame` in the REPL**: These observability modes were tied to V1 semantics and temporarily removed. They will be re-introduced with behavior adapted to the V2 execution model.

### Misc

- **`vmap` on dictionaries**: `vmap` currently only accepts tensor arguments. PyTree support (automatic flattening/unflattening of dicts) will be added to match the behavior of `value-and-grad`.
