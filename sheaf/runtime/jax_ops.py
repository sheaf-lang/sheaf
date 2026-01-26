# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Maps Sheaf primitive symbols to JAX/LAX numerical implementations.
Defines the foreign function interface (FFI) for tensor operations.
"""

import jax
import jax.numpy as jnp


def sheaf_tensor(data):
    """
    Converts a list, tuple, or nested structure to a JAX tensor.

    Useful for converting quoted lists '[1 2 3] or dynamically generated
    lists (via cons, append, etc.) into tensors for computation.

    Examples:
        (tensor '[1 2 3 4 5 6])           ; => f32[6]
        (reshape (tensor '[1 2 3 4 5 6]) '[2 3]) ; => f32[2x3]
        (+ (tensor (cons 1 '[2 3])) 10)  ; => f32[3]
    """
    return jnp.array(data)


def sheaf_append(lst, x):
    """
    Appends an element to a list, tuple, or JAX array.
    Used for accumulating generated tokens.
    """
    if isinstance(lst, list):
        return lst + [x]
    if isinstance(lst, tuple):
        return lst + (x,)
    # Fallback for JAX arrays (note: this creates a new array)
    return jnp.append(lst, x)


def sheaf_append_and_roll(window, new_id):
    """
    Efficiently updates a rolling context window for autoregressive inference.
    Shifts the window to the left and adds the new ID at the end.
    """
    # Ensure new_id is an array to allow concatenation
    new_id_arr = jnp.atleast_1d(jnp.array(new_id, dtype=jnp.int32))
    # Concatenate the window (minus the first element) with the new ID
    return jnp.concatenate([window[1:], new_id_arr])


def sheaf_randint(key, shape, minval, maxval):
    # JAX randint requires concrete shape and boundaries
    return jax.random.randint(key, tuple(shape), minval, maxval)


def sheaf_reshape(a, *shape_args):
    # Reshape tensor by flattening into a single dimension
    flat_shape = []
    for item in shape_args:
        if isinstance(item, (tuple, list)):
            flat_shape.extend(item)
        else:
            flat_shape.append(item)
    return jnp.reshape(a, tuple(flat_shape))


def sheaf_shape(tensor, axis=None):
    """
    Handles JAX arrays
    """
    try:
        s = tensor.shape
        if axis is not None:
            # axis can be negative (like -1), so we check against range
            return s[axis]
        return s
    except (AttributeError, IndexError, TypeError) as e:
        # If it's not a tensor or axis is wrong, we provide context
        if not hasattr(tensor, "shape"):
            raise TypeError(
                f"Object has no 'shape' attribute. Type: {type(tensor).__name__}"
            )
        raise IndexError(
            f"Dimension index {axis} is out of range for shape {tensor.shape}"
        )


def sheaf_dynamic_slice(x, start, length):
    """
    JAX dynamic_slice on first axis. For JIT-compatible slicing with variable indices.
    Use (slice x start end) for Python-style slicing on strings/lists/tensors.
    """
    return jax.lax.dynamic_slice_in_dim(x, start, length, axis=0)


def sheaf_transpose(tensor, axes=None):
    if axes is None:
        return jnp.transpose(tensor)
    # Convert list/tuple to tuple if needed
    if isinstance(axes, (list, tuple)):
        axes = tuple(axes)
    # If the user provides axes, we use them as a permutation
    return jnp.transpose(tensor, axes=axes)


def sheaf_tree_map(f, *trees):
    def safe_f(*args):
        # Check if any argument passed to the lambda is a module
        import types

        for i, arg in enumerate(args):
            if isinstance(arg, types.ModuleType):
                raise TypeError(
                    f"Leaf in tree-map at position {i} is a module! Type: {type(arg)}"
                )
        return f(*args)

    return jax.tree_util.tree_map(safe_f, *trees)


def sheaf_tree_map_zeros(tree):
    """
    Creates a new tree with the same structure as the input,
    but with all leaf values set to zero.
    Essential for initializing optimizer states (Adam).
    """

    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def sheaf_flatten(tree):
    """
    Flattens a PyTree into a single list of leaves.
    Returns (leaves, tree_def) where tree_def can be used to unflatten.

    Converts Sheaf-specific types (SheafList) to Python equivalents first.

    Examples:
        (flatten {:a 1 :b 2})           -> ([1, 2], ...)
        (flatten '(1 2 3 4))            -> ([1, 2, 3, 4], ...)
        (first (flatten {:a 1 :b 2}))   -> [1, 2]
    """
    # Import here to avoid circular imports
    from sheaf.core.parser import SheafList

    def convert_sheaf_types(obj):
        """Convert SheafList to Python list recursively."""
        if isinstance(obj, SheafList):
            return [convert_sheaf_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_sheaf_types(v) for k, v in obj.items()}
        else:
            return obj

    # Convert tree to native Python types
    tree = convert_sheaf_types(tree)

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    # Return as a tuple so users can get leaves with (first ...)
    return (leaves, treedef)


def sheaf_tree_reduce(f, tree, init=None):
    """
    Reduces all leaves of a PyTree using the provided function.

    Converts Sheaf-specific types (SheafList) to Python equivalents first.

    Examples:
        (tree-reduce + {:a 1 :b 2 :c 3} 0)          -> 6
        (tree-reduce + '(1 2 3 4) 0)                -> 10
        (tree-reduce * '(2 3) 1)                    -> 6
    """
    # Import here to avoid circular imports
    from sheaf.core.parser import SheafList

    def convert_sheaf_types(obj):
        """Convert SheafList to Python list recursively."""
        if isinstance(obj, SheafList):
            return [convert_sheaf_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_sheaf_types(v) for k, v in obj.items()}
        else:
            return obj

    # Convert tree to native Python types
    tree = convert_sheaf_types(tree)

    return jax.tree_util.tree_reduce(f, tree, init)


def get_jax_env():
    return {
        "append": sheaf_append,
        "append-and-roll": sheaf_append_and_roll,
        "arange": jnp.arange,
        "choice": jax.random.choice,
        "tensor": sheaf_tensor,
        "einsum": jnp.einsum,
        "flatten": sheaf_flatten,
        "maximum": jnp.maximum,
        "minimum": jnp.minimum,
        "ndim": lambda x: x.ndim,
        "normalize": lambda x: x / (jnp.sum(x, axis=-1, keepdims=True) + 1e-12),
        "one-hot": jax.nn.one_hot,
        "ones": jnp.ones,
        "product": jnp.prod,
        "random-key": jax.random.key,
        "random-normal": jax.random.normal,
        "random-randint": sheaf_randint,
        "random-split": jax.random.split,
        "random-uniform": jax.random.uniform,
        "range": lambda *args: jnp.arange(*args),
        "reshape": sheaf_reshape,
        "roll": jnp.roll,
        "shape": sheaf_shape,
        "dynamic-slice": sheaf_dynamic_slice,
        "swapaxes": jnp.swapaxes,
        "tanh": jnp.tanh,
        "tensor-split": jnp.split,
        "top_k": jax.lax.top_k,
        "transpose": sheaf_transpose,
        "tree-map": sheaf_tree_map,
        "tree-map-zeros": sheaf_tree_map_zeros,
        "tree-reduce": sheaf_tree_reduce,
        "tril": jnp.tril,
        "var": jnp.var,
        "where": jnp.where,
        "zeros": jnp.zeros,
        # "reshape": lambda a, *shape: jnp.reshape(a, shape),
        # "transpose": lambda a, *axes: jnp.transpose(a, axes if axes else None),
        # "tree-map": jax.tree_util.tree_map,
    }
