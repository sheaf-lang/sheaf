# Hydra

A self-evolving neural network implemented in Sheaf.

## How it works

The network starts with a single linear head (no hidden layers), which cannot
solve XOR. Every 20 epochs, the training loop checks for a loss plateau
(progress < 0.003). When one is detected, `grow-hydra` appends a new hidden
layer. JAX retraces `forward` at that point, but the cost is negligible for
networks of typical depth.

## Zero recompilation at grow

`grow-hydra` appends a new layer and reinitialises the head. This changes the
pytree structure, so JAX _retraces_ `forward`, but retracing is not the same
as recompilation. The XLA kernels produced by the trace are cached by shape.
All shapes that appear after a grow (`[4,32]`, `[32,1]`, etc.) were already
seen during the initial training phase, so they hit the cache.

`verify.sh` proves this empirically using `JAX_LOG_COMPILES=1`:

```
XLA Recompilation Report
 Total compilations : 71
 Before grow        : 71
 After grow         : 0
--> 'grow' required zero recompilation
```
