# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Provides fundamental data structure manipulation primitives.
"""

from functools import reduce


def _sheaf_get_in(obj, path):
    """Navigate through nested dicts/lists using a path."""
    # If path is not a list (single level), convert it
    if not isinstance(path, (list, tuple)):
        path = [path]

    res = obj
    for key in path:
        # Auto-clean Lisp keywords: ':token' -> 'token' (but not bare ":")
        k = (
            key[1:]
            if isinstance(key, str) and key.startswith(":") and len(key) > 1
            else key
        )
        res = res[k]
    return res


def create_dict(*args):
    # Process arguments in pairs: key-value
    # args[i] will be the key (e.g., ":Wq"), args[i+1] the value
    d = {}
    for i in range(0, len(args), 2):
        key = args[i]
        if isinstance(key, str) and key.startswith(":") and len(key) > 1:
            key = key[1:]
        d[key] = args[i + 1]
    return d


def generic_apply(func, *args):
    """
    Apply a function to arguments.
    The last argument is expected to be a sequence (list, tuple, or JAX array)
    that will be unpacked.
    """
    if not args:
        return func()

    # Separate fixed arguments from the last one (the list to unpack)
    fixed_args = args[:-1]
    last_arg = args[-1]

    # Check if last argument is a sequence (list, tuple, or JAX array)
    # JAX arrays have __iter__ and can be unpacked
    if isinstance(last_arg, (list, tuple)) or hasattr(last_arg, "__iter__"):
        try:
            # Standard Lisp apply: (func fixed_args... *last_arg)
            return func(*fixed_args, *last_arg)
        except TypeError:
            # If unpacking fails, fall back to standard call
            return func(*args)
    else:
        # Fallback to standard call if the last argument is not a sequence
        return func(*args)


def generic_slice(obj, start, end=None):
    """
    Generic slice operation for strings, lists, and tensors.

    Examples:
        (slice "hello" 1)     -> "ello"
        (slice "hello" 1 3)   -> "el"
        (slice [1 2 3 4] 1 3) -> [2 3]
        (slice tensor 0 10)   -> tensor[0:10]

    Args:
        obj: sequence to slice (string, list, tuple, or tensor)
        start: start index
        end: end index (optional, None means to end)

    Returns:
        sliced object
    """
    if end is None:
        return obj[start:]
    return obj[start:end]


def cons(head, tail):
    """
    Construct a new list by prepending head to tail.

    Examples:
        (cons 1 [2 3])    -> [1 2 3]
        (cons 'a [])      -> ['a]
        (cons 'x ['y 'z]) -> ['x 'y 'z]

    Args:
        head: element to prepend
        tail: list or tuple to prepend to

    Returns:
        new list/tuple with head prepended to tail
    """
    if isinstance(tail, list):
        return [head] + tail
    elif isinstance(tail, tuple):
        return (head,) + tail
    else:
        raise TypeError(
            f"cons: second argument must be a list or tuple, got {type(tail)}"
        )


def generic_concat(*args, **kwargs):
    """
    Concatenate sequences (lists, tuples, or JAX arrays).

    Supports :axis keyword for array concatenation.
    Examples:
        (concat '(1 2) '(3 4)) -> (1 2 3 4)
        (concat [1 2] [3 4] :axis 0) -> [1. 2. 3. 4.]
    """
    if not args:
        return []

    # Extract axis from kwargs
    axis = kwargs.get("axis", 0) if kwargs else 0
    has_axis_kwarg = "axis" in kwargs

    # Check what we're concatenating
    has_jax_arrays = any(hasattr(arg, "shape") for arg in args)
    has_lists = any(isinstance(arg, (list, tuple)) for arg in args)

    # If we have JAX arrays, use JAX concatenation (possibly with axis)
    if has_jax_arrays:
        import jax.numpy as jnp

        arrays = []
        for arg in args:
            arrays.append(jnp.asarray(arg))
        return jnp.concatenate(arrays, axis=axis)

    # If we have only lists/tuples
    if has_lists:
        # If :axis is specified, convert to JAX arrays and concatenate
        if has_axis_kwarg:
            import jax.numpy as jnp

            arrays = [jnp.asarray(arg) for arg in args]
            return jnp.concatenate(arrays, axis=axis)

        # Otherwise, concatenate as lists
        result = []
        for arg in args:
            result.extend(arg)
        return result

    # Try JAX concatenation as fallback
    try:
        import jax.numpy as jnp

        arrays = [jnp.asarray(arg) for arg in args]
        return jnp.concatenate(arrays, axis=axis)
    except (ImportError, TypeError) as e:
        if has_axis_kwarg:
            raise ValueError(
                ":axis requires JAX arrays, but concatenation failed"
            ) from e
        # Fall back to string concatenation if JAX fails
        return "".join(map(str, args))


def count(lst):
    """
    Return the number of elements in a list or first dimension of an array.

    Examples:
        (count [1 2 3])  -> 3
        (count [])       -> 0
        (count [[1 2] [3 4]]) -> 2

    Returns:
        number of elements
    """
    if isinstance(lst, (list, tuple, str)):
        return len(lst)
    # For JAX arrays, return the size of the first dimension
    elif hasattr(lst, "shape"):
        return lst.shape[0] if lst.shape else 0
    else:
        return 0


def empty_q(lst):
    """
    Check if a list or array is empty.

    Examples:
        (empty? [])     -> True
        (empty? [1 2])  -> False
        (empty? [])     -> True (empty JAX array)

    Args:
        lst: list or array to check

    Returns:
        True if empty, False otherwise
    """
    if isinstance(lst, (list, tuple)):
        return len(lst) == 0
    # For JAX arrays, check size
    elif hasattr(lst, "size"):
        return lst.size == 0
    else:
        return False


def rest(lst):
    """
    Return all elements of a list except the first.

    Examples:
        (rest [1 2 3])  -> [2 3]
        (rest ['a])     -> []
        (rest [])       -> []

    Returns:
        list without the first element
    """
    if not isinstance(lst, (list, tuple)):
        raise TypeError(f"rest: argument must be a list, got {type(lst)}")
    return list(lst[1:]) if len(lst) > 0 else []


def sheaf_sort(seq, **kwargs):
    """
    Sort a list or tensor. Polymorphic: behavior and valid options depend on type.

    Options:
        | Option     | List | Tensor |
        |------------|------|--------|
        | :reverse   | yes  | yes    |
        | :axis      | no   | yes    |
        | :key       | yes  | no     |

    Examples:
        (sort '("c" "a" "b"))                    -> ("a" "b" "c")
        (sort '("c" "a" "b") :reverse)           -> ("c" "b" "a")
        (sort '("cat" "ant" "bee") :key len)     -> ("ant" "bee" "cat")
        (sort [3.0 1.0 2.0])                     -> [1. 2. 3.]
        (sort [3.0 1.0 2.0] :reverse)            -> [3. 2. 1.]
        (sort [[3 1] [2 4]] :axis 1)             -> [[1 3] [2 4]]
    """
    reverse = kwargs.get("reverse", False)
    axis = kwargs.get("axis", None)
    key = kwargs.get("key", None)

    is_tensor = hasattr(seq, "shape")

    # --- Validate option/type combinations ---
    if is_tensor:
        if key is not None:
            raise TypeError("sort: :key is not supported on tensors (only on lists)")
    else:
        if axis is not None:
            raise TypeError("sort: :axis is not supported on lists (only on tensors)")

    # --- Tensor path ---
    if is_tensor:
        import jax.numpy as jnp

        ax = axis if axis is not None else -1
        result = jnp.sort(seq, axis=ax)
        if reverse:
            # Flip along the sort axis
            result = jnp.flip(result, axis=ax)
        return result

    # --- List path ---
    result = sorted(seq, key=key, reverse=reverse)
    return result


def symbol_q(obj):
    """
    Check if object is a symbol.

    In Sheaf, symbols are represented as strings.

    Examples:
        (symbol? 'foo)   -> True
        (symbol? "foo")  -> True
        (symbol? 42)     -> False

    Returns:
        True if object is a symbol/string, False otherwise
    """
    return isinstance(obj, str)


def gensym(prefix="G__"):
    """
    Generate a unique symbol.

    Useful for creating unique variable names in macros.

    Examples:
        (gensym)       -> "G__1"
        (gensym "tmp") -> "tmp2"

    Args:
        prefix: prefix for the generated symbol

    Returns:
        unique symbol string
    """
    import uuid

    return f"{prefix}{uuid.uuid4().hex[:8]}"


def sheaf_assoc(dict_obj, *key_val_pairs):
    """
    Associate (update) a dictionary with new key-value pairs.
    Returns a new dict with the updates applied (non-mutating).

    Examples:
        (assoc {:a 1} :b 2)              -> {:a 1, :b 2}
        (assoc {:a 1} :a 10 :b 2)        -> {:a 10, :b 2}
    """
    result = dict(dict_obj) if isinstance(dict_obj, dict) else {}
    for i in range(0, len(key_val_pairs), 2):
        key = key_val_pairs[i]
        val = key_val_pairs[i + 1]
        # Auto-clean Lisp keywords: ':token' -> 'token' (but not bare ":")
        clean_key = (
            key[1:]
            if isinstance(key, str) and key.startswith(":") and len(key) > 1
            else key
        )
        result[clean_key] = val
    return result


def sheaf_dissoc(dict_obj, keys_to_remove):
    """
    Dissociate (remove) keys from a dictionary.
    Returns a new dict with the specified keys removed (non-mutating).

    Args:
        dict_obj: dictionary to remove keys from
        keys_to_remove: list of keys to remove (e.g., [:a :b])

    Examples:
        (dissoc {:a 1 :b 2} [:b])          -> {:a 1}
        (dissoc {:a 1 :b 2 :c 3} [:a :c])  -> {:b 2}
    """
    result = dict(dict_obj) if isinstance(dict_obj, dict) else {}

    # Handle both list and tuple for keys_to_remove
    if isinstance(keys_to_remove, (list, tuple)):
        for key in keys_to_remove:
            # Auto-clean Lisp keywords: ':token' -> 'token' (but not bare ":")
            clean_key = (
                key[1:]
                if isinstance(key, str) and key.startswith(":") and len(key) > 1
                else key
            )
            result.pop(clean_key, None)  # Remove key if it exists
    else:
        # Single key case
        clean_key = (
            keys_to_remove[1:]
            if isinstance(keys_to_remove, str)
            and keys_to_remove.startswith(":")
            and len(keys_to_remove) > 1
            else keys_to_remove
        )
        result.pop(clean_key, None)

    return result


def sheaf_merge(*dicts):
    """
    Merge multiple dictionaries into one.
    Later dicts override earlier ones for conflicting keys.
    Returns a new dict (non-mutating).

    Examples:
        (merge {:a 1} {:b 2})            -> {:a 1, :b 2}
        (merge {:a 1} {:a 10} {:b 2})    -> {:a 10, :b 2}
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def sheaf_keys(dict_obj):
    """
    Get all keys from a dictionary as a list.

    Examples:
        (keys {:a 1 :b 2})               -> (:a :b) or ['a', 'b']
    """
    if isinstance(dict_obj, dict):
        return list(dict_obj.keys())
    return []


def sheaf_vals(dict_obj):
    """
    Get all values from a dictionary as a list.

    Examples:
        (vals {:a 1 :b 2})               -> (1 2)
    """
    if isinstance(dict_obj, dict):
        return list(dict_obj.values())
    return []


def get_core_env():
    return {
        "apply": generic_apply,
        "assoc": sheaf_assoc,
        "chars": lambda s: list(str(s)),
        "cons": cons,
        "count": count,
        # "dict": create_dict,
        "dissoc": sheaf_dissoc,
        "empty?": empty_q,
        "filter": lambda pred, lst: tuple(x for x in lst if pred(x)),
        "find": lambda pred, lst: next((x for x in lst if pred(x)), None),
        "first": lambda x: x[0] if x else None,
        "gensym": gensym,
        # "get" is now a special form in compiler.py to avoid keyword argument issues
        # "get": lambda obj, *keys: obj[...],
        "get-in": _sheaf_get_in,
        "index-of": lambda lst, val: next(
            (i for i, x in enumerate(lst) if x == val), -1
        ),
        "keys": sheaf_keys,
        "last": lambda x: x[-1] if x else None,
        # "list": lambda *args: list(args),
        "map": lambda f, lst: tuple(f(x) for x in lst),
        "merge": sheaf_merge,
        "nth": lambda x, i: x[i],
        "reduce": lambda f, acc, lst: reduce(f, lst, acc),
        "rest": rest,
        "slice": generic_slice,
        "sort": sheaf_sort,
        "symbol?": symbol_q,
        "vals": sheaf_vals,
    }
