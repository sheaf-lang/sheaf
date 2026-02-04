# 2026 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
I/O operations for Sheaf.

Single entry point: (io verb path [format])
Verbs: load, save, read, lines

Format is inferred from file extension.
Override with a keyword arg: (io "load" "weights.dat" :safetensors)

Supported formats:
  .safetensors  — lazy tensor loading via mmap (dtype preserved)
  .pkl          — pickle (legacy, discouraged)
  .txt          — plain text
  .json         — JSON object
  .jsonl        — newline-delimited JSON (streaming via "lines")
"""

import glob as glob_module
import json
import os
import pickle

# Format registry
EXTENSION_TO_FORMAT = {
    ".safetensors": "safetensors",
    ".pkl": "pkl",
    ".pickle": "pkl",
    ".txt": "txt",
    ".json": "json",
    ".jsonl": "jsonl",
}


def _infer_format(path, hint=None):
    """Infer format from extension, or use hint if provided."""
    if hint:
        return hint
    _, ext = os.path.splitext(path)
    fmt = EXTENSION_TO_FORMAT.get(ext.lower())
    if fmt is None:
        raise ValueError(
            f"Unknown format for '{path}'. "
            f"Supported extensions: {list(EXTENSION_TO_FORMAT.keys())}. "
            f'Or pass a format hint, e.g. (io "load" "{path}" :safetensors)'
        )
    return fmt


# SafeTensorsHandle — lazy handle, loads tensors on demand
class SafeTensorsHandle:
    """
    Lazy handle over one or more .safetensors files.
    Tensors are read from disk only when accessed via [] (i.e. Sheaf's `get`).
    dtype is preserved as stored in the file — never cast to f32.
    """

    def __init__(self, paths):
        from safetensors import safe_open

        # paths: single file or list of shards
        if isinstance(paths, str):
            paths = [paths]
        self._paths = paths

        # Build key → path index from all shards (header only, no tensor data read)
        self._index = {}
        for p in paths:
            with safe_open(p, framework="numpy") as f:
                for k in f.keys():
                    self._index[k] = p

    def __getitem__(self, key):
        """Load a single tensor by key. Called by GetForm."""
        import jax.numpy as jnp
        from safetensors import safe_open

        # Strip Sheaf keyword prefix if present
        if isinstance(key, str) and key.startswith(":"):
            key = key[1:]

        if key not in self._index:
            raise KeyError(
                f"Tensor '{key}' not found. "
                f"Available keys: {list(self._index.keys())[:20]}"
                + (" ..." if len(self._index) > 20 else "")
            )

        path = self._index[key]
        with safe_open(path, framework="numpy") as f:
            return jnp.asarray(f.get_tensor(key))  # numpy → JAX, dtype preserved

    def keys(self):
        return list(self._index.keys())

    def __contains__(self, key):
        if isinstance(key, str) and key.startswith(":"):
            key = key[1:]
        return key in self._index

    def __repr__(self):
        return f"SafeTensorsHandle(keys={len(self._index)}, shards={len(self._paths)})"


# Loaders / savers per format
def _load_safetensors(path):
    """
    Load safetensors — single file or glob pattern (sharding).
    Returns a SafeTensorsHandle (lazy).
    """
    # Glob expansion for sharded models: (io "load" "model-*.safetensors")
    paths = sorted(glob_module.glob(path))
    if not paths:
        # Not a glob — treat as single file
        if os.path.exists(path):
            paths = [path]
        else:
            raise FileNotFoundError(f"No file matching '{path}'")
    return SafeTensorsHandle(paths)


def _load_safetensors_index(path):
    """
    Load a HuggingFace shard index (model.safetensors.index.json).
    Resolves shard paths relative to the index file's directory,
    then returns a SafeTensorsHandle over all shards.
    """
    with open(path, "r") as f:
        index = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(path))
    # index["weight_map"] is {tensor_name: shard_filename}
    shard_files = sorted(set(index["weight_map"].values()))
    shard_paths = [os.path.join(base_dir, s) for s in shard_files]

    missing = [p for p in shard_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing shard files: {missing}")

    return SafeTensorsHandle(shard_paths)


def _save_safetensors(path, data):
    """Save a pytree (dict of tensors) to safetensors."""
    import numpy as np
    from safetensors.numpy import save_file

    # Flatten nested dicts with dot-separated keys (mirrors HF convention)
    flat = {}

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else k
                _flatten(v, new_key)
        else:
            flat[prefix] = np.asarray(obj)

    _flatten(data)
    save_file(flat, path)


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pkl(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_txt(path):
    with open(path, "r") as f:
        return f.read()


# Streaming: "lines" verb
class LazyLines:
    """
    Lazy line iterator over a text file.
    Supports iteration (for reduce/map in Sheaf) without loading the whole file.
    """

    def __init__(self, path):
        self._path = path

    def __iter__(self):
        with open(self._path, "r") as f:
            for line in f:
                yield line.rstrip("\n")

    def __repr__(self):
        return f"LazyLines({self._path!r})"


# Main dispatch: io
KNOWN_FORMATS = {"safetensors", "pkl", "pickle", "txt", "json", "jsonl"}


def io_call(verb, path=None, *args, **kwargs):
    """
    Single entry point for all I/O.

    Verbs:
      load  — deserialize file → pytree / string / dict
      save  — serialize data → file
      read  — read file as raw string
      lines — return lazy line iterator (streaming)

    Format hint (keyword flag after path):
      (io "load" "weights.dat" :safetensors)

    The compiler turns :safetensors into kwargs={"safetensors": True}.
    We scan kwargs keys against known formats to extract the hint.
    """
    # --- entropy: no path involved, just n_bytes ---
    # (io "entropy")     → 4 bytes (default, fits int32 / random-key)
    # (io "entropy" 16)  → 16 bytes (UUID-scale)
    if verb == "entropy":
        n_bytes = int(path) if path is not None else 4
        return int.from_bytes(os.urandom(n_bytes), "big")

    # Extract format hint from kwargs flags: (io "load" "x.dat" :safetensors)
    hint = None
    for k in kwargs:
        if k in KNOWN_FORMATS:
            hint = k
            break

    # Remaining positional args: data payload for "save"
    data = args[0] if args else None

    # --- Special case: .safetensors.index.json (HF shard index) ---
    if path.endswith(".safetensors.index.json"):
        if verb == "load":
            return _load_safetensors_index(path)
        raise ValueError(
            f"Verb '{verb}' not supported on shard index files (use 'load')"
        )

    fmt = _infer_format(path, hint)

    # ---- load ----
    if verb == "load":
        if fmt == "safetensors":
            return _load_safetensors(path)
        if fmt == "pkl":
            return _load_pkl(path)
        if fmt == "json":
            return _load_json(path)
        if fmt == "jsonl":
            return _load_json(path)  # eager load of jsonl as list
        if fmt == "txt":
            return _load_txt(path)
        raise ValueError(f"No loader for format '{fmt}'")

    # ---- save ----
    if verb == "save":
        if data is None:
            raise ValueError('(io "save" path data) — missing data argument')
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if fmt == "safetensors":
            _save_safetensors(path, data)
        elif fmt == "pkl":
            _save_pkl(path, data)
        elif fmt == "json":
            _save_json(path, data)
        elif fmt == "txt":
            with open(path, "w") as f:
                f.write(str(data))
        else:
            raise ValueError(f"No saver for format '{fmt}'")
        return None

    # ---- read ----
    if verb == "read":
        with open(path, "r") as f:
            return f.read()

    # ---- lines ----
    if verb == "lines":
        return LazyLines(path)

    # ---- exists ----
    if verb == "exists":
        return os.path.exists(path)

    raise ValueError(
        f"Unknown io verb '{verb}'. Supported: load, save, read, lines, exists, entropy"
    )


# Public env
def get_io_env():
    return {
        "io": io_call,
    }
