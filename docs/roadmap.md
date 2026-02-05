---
hide:
  - toc
---

# Planned evolutions for Sheaf

Sheaf currently compiles through JAX to XLA, inheriting Python's runtime overhead. The next major version will target StableHLO directly, eliminating the Python dependency for compiled programs.

**Upcoming architecture**:

- Sheaf compiler rewritten in Rust
- Direct StableHLO emission (bypassing JAX)
- AOT compilation via [IREE](https://github.com/iree-org/iree)
- Autodiff via [Enzyme](https://enzyme.mit.edu/) or IREE's gradient primitives

**Benefits**:

- Runtime footprint: 1-2MB (vs 500-700MB with Python+JAX or PyTorch)
- Compile-time footprint: 30-50MB
- No Python runtime required for inference or training
- Standalone executables for deployment

This will align the technical stack with the language: elegance and concision from syntax to runtime. The transition will also preserve Sheaf's homoiconic nature and functional purity yet eliminate accidental complexity inherited from Python.

### Parallelism and distribution

Sheaf does not currently expose low-level sharding primitives like `pmap`. Instead, it relies on JAX's automatic partitioning, which works well for single-machine training but limits multi-node scaling.

In V2, distribution could be expressed through:

- StableHLO's native sharding annotations
- Macro-based rewrites during compilation
- IREE's multi-device runtime capabilities

This would give explicit control over parallelization while keeping mathematical definitions clean from hardware concerns.
