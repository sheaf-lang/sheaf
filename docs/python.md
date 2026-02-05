# Python Interoperability

Sheaf code compiles to pure JAX functions that integrate seamlessly with Python. Sheaf functions can be called from Python, the compilation registry can be accessed, and Sheaf and Python code can be mixed in the same project.

---

## Basic Usage: Inline Code

The simplest way to use Sheaf from Python is to load code directly as a string:

```python
from sheaf import Sheaf

shf = Sheaf()

# Load Sheaf code as a string
shf.load("""
  (defn add-five [x]
    (+ x 5))

  (defn square [x]
    (* x x))
""")

# Call Sheaf functions as Python methods
result = shf.add_five(10)      # => 15
result = shf.square(3)         # => 9

# Works with arrays too
import jax.numpy as jnp
arr = jnp.array([1, 2, 3])
result = shf.add_five(arr)     # => Array([6., 7., 8.])
```

### Python Naming Convention

Sheaf function names use kebab-case (hyphens), but Python methods use underscores. Sheaf automatically converts:

```sheaf
(defn train-step [params grad]
  ...)

(defn compute-loss [x y]
  ...)
```

```python
# Access with underscores instead of hyphens
shf.train_step(params, grad)
shf.compute_loss(x, y)
```

---

## File-Based Code

For larger projects, Sheaf code can be loaded from files:

### Loading a Single File

```python
from sheaf import Sheaf

# Load from a relative path (relative to calling script)
shf = Sheaf("model.shf")

# Or load after creation
shf = Sheaf()
shf.load_from_path("path/to/model.shf")
```

### Relative Path Resolution

Paths are resolved relative to the calling script's directory, not the current working directory:

```python
# In /project/train/runner.py
shf = Sheaf("../models/nn.shf")  # Resolves to /project/models/nn.shf

# Works correctly regardless of where the script is run from:
# $ cd /project && python train/runner.py
# $ cd /project/train && python runner.py
```

---

## Accessing the Registry

The **registry** contains all compiled functions. It can be introspected from Python:

### Getting Registry Metadata

```python
from sheaf import Sheaf

shf = Sheaf()
shf.load("""
  (defn forward [x params]
    (+ (@ x (:W params)) (:b params)))

  (defn loss [pred target]
    (mean (* (- pred target) (- pred target))))
""")

# Get metadata about all functions
registry = shf.get_registry()

print(registry)
# Output:
# {
#   'forward': {'params': ['x', 'params'], 'source': '(defn forward ...'},
#   'loss': {'params': ['pred', 'target'], 'source': '(defn loss ...'}
# }

# Check what functions are available
if 'forward' in shf.registry:
    forward_fn = shf.registry['forward']
    # forward_fn can now be called with appropriate arguments
```

### Common Registry Tasks

```python
# List all loaded functions
print(list(shf.registry.keys()))

# Get a function by name
fn = shf.registry['my-function']
result = fn(arg1, arg2)

# Get function parameters
meta = shf.get_registry()
params = meta['my-function']['params']
print(f"Parameters: {params}")

# Call with kwargs (Python standard)
fn = shf.registry['my-function']
result = fn(x=value1, y=value2)
```

---

## Accessing the Environment

The **environment (env)** contains all variables and intermediate values defined or computed during execution:

### Viewing Environment Contents

```python
from sheaf import Sheaf

# List what's available in the environment
print(list(shf.env.keys()))  # Shows built-in functions

shf = Sheaf()
shf.load("""
  (defn create-model [key]
    {:W (random-normal key '[10 5])
     :b (zeros '[5])})

  (defn get-config []
    {:learning-rate 0.001
     :batch-size 32})
""")

# Call functions
model = shf.create_model(shf.env['random-key'](0))
config = shf.get_config()

# Get all user-defined functions with metadata
registry = shf.get_registry()

print(registry.keys())
# Output: dict_keys(['create-model', 'get-config'])

# Access config values from returned dictionary
lr = config['learning-rate']
print(f"Learning rate: {lr}")
```

### Working with Tensors

```python
shf.load("""
  (defn create-weights [key]
    {:W (random-normal key '[3 4])
     :b (zeros '[4])})
""")

# Create weights
weights = shf.create_weights(shf.env['random-key'](0))

# Access tensor values
W = weights['W']
print(f"Shape: {W.shape}")      # (3, 4)
print(f"Dtype: {W.dtype}")      # float32

# Modify tensors using JAX
import jax.numpy as jnp
new_W = W * 0.5
print(new_W)  # JAX Array
```

---

## PyTree Conversion

Sheaf provides conversion utilities for working with JAX pytrees:

### Converting Sheaf Values to PyTrees

```python
from sheaf import Sheaf
import jax

shf = Sheaf()
shf.load("""
  (defn create-params [key]
    (let [[k1 k2] (random-split key 2)]
      {:layer1 {:W (random-normal k1 '[3 4]) :b (zeros '[4])}
       :layer2 {:W (random-normal k2 '[4 1]) :b (zeros '[1])}}))
""")

params = shf.create_params(shf.env['random-key'](42))

# Convert to pure JAX pytree
pytree = shf.to_pytree(params)

# Now use with JAX transformations
def loss_fn(pytree_params):
    # params is a pure pytree here
    return jax.numpy.sum(pytree_params['layer1']['W'])

loss = loss_fn(pytree)
```

### Converting PyTrees Back to Sheaf

```python
# Apply JAX transformation to pytree (e.g., scale all weights)
import jax.tree_util as tree_util
pytree_result = tree_util.tree_map(lambda x: x * 0.1, pytree)

# Convert back to Sheaf-compatible structure
sheaf_result = shf.from_pytree(pytree_result)

print(sheaf_result['layer1']['W'].shape)  # (3, 4)
```

---

## Introspecting Language Features

### Available Special Forms

```python
# See all special forms (defn, let, vmap, scan, reduce, etc.)
special_forms = shf.get_special_forms()
print(special_forms)
# ['and', 'assoc', 'def', 'defn', 'dissoc', 'let', 'reduce', 'scan', 'vmap', ...]
```

### Function Inspection

```python
# Load neural network library
shf.load("(use nn)")

# Get detailed info about a function
registry = shf.get_registry()

if 'layer-norm' in registry:
    info = registry['layer-norm']
    print(f"Parameters: {info['params']}")  # ['x', 'p', 'axis']
    print(f"Source:\n{info['source']}")
```

---

## Mixing Python and Sheaf Code

Data preprocessing can be done in Python, then passed to Sheaf for computation:

```python
from sheaf import Sheaf
import jax.numpy as jnp
import numpy as np

shf = Sheaf()
shf.load("""
  (defn forward [x params]
    (+ (@ x (get params :W)) (get params :b)))
""")

# Data preprocessing in Python
raw_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
normalized = (raw_data - raw_data.mean()) / raw_data.std()

# Computation in Sheaf
params = {'W': jnp.array([[1, 2], [3, 4]]), 'b': jnp.array([0, 0])}
output = shf.forward(normalized, params)

print(output)  # JAX array result
```

---

## Error Handling

Sheaf provides detailed error messages that include line numbers and context:

```python
from sheaf import Sheaf

shf = Sheaf()

try:
    shf.load("""
      (defn broken [x]
        (undefined-function x))
    """)
    result = shf.broken(5)
except Exception as e:
    print(f"Error: {e}")
    # Error: Symbol not found: 'undefined-function' (line 3)
```

---

## Type Safety and JAX Integration

Sheaf functions are pure JAX functions. They:

- **Compile to JAX bytecode** for performance
- **Support automatic differentiation** via `jax.grad` and `jax.value_and_grad`
- **Compose with JAX transformations** like `jax.jit`, `jax.vmap`, `jax.scan`

```python
from sheaf import Sheaf
import jax

shf = Sheaf()
shf.load("""
  (defn model [x params]
    (sigmoid (+ (@ x (:W params)) (:b params))))
""")

# Extract the JAX function
model_fn = shf.registry['model']

# Use JAX transformations
grad_fn = jax.grad(model_fn)
jitted_fn = jax.jit(model_fn)

# Compose them
fast_grad = jax.jit(jax.grad(model_fn))
```

---

## Summary

| Operation           | Code                       | Notes                       |
| ------------------- | -------------------------- | --------------------------- |
| Inline code         | `shf.load("(defn ...")`    | Simple, interactive         |
| File-based          | `shf = Sheaf("model.shf")` | Better for larger projects  |
| Call function       | `shf.function_name(args)`  | Python naming (underscores) |
| Get registry        | `shf.get_registry()`       | Metadata about functions    |
| Get environment     | `shf.get_env()`            | Metadata about variables    |
| Access function     | `shf.registry['name']`     | Direct JAX function         |
| Access variable     | `shf.env['name']`          | Direct value                |
| Convert to pytree   | `shf.to_pytree(value)`     | For JAX operations          |
| Convert from pytree | `shf.from_pytree(pytree)`  | Reconstruct Sheaf value     |
| Introspect forms    | `shf.get_special_forms()`  | Available language features |

For detailed function signatures and examples, see the [Function Reference](reference.md).
