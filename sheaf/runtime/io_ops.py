# 2026 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
I/O operations for Sheaf.

Single entry point: (io verb path [format] [dtype])
Verbs: load, save, read, lines

Format is inferred from file extension.
Override with a keyword arg: (io "load" "weights.dat" :safetensors)

Supported formats:
  .safetensors  — lazy tensor loading via mmap (dtype preserved)
  .npy          — numpy arrays, mmap'd (dtype from header)
  .bin          — raw binary, mmap'd (dtype must be explicit: :i32, :i16, etc.)
  .pkl          — pickle (legacy, discouraged)
  .txt          — plain text
  .json         — JSON object
  .jsonl        — newline-delimited JSON (streaming via "lines")

Glob patterns are supported for .npy and .bin: returns a ShardedHandle
(virtual concatenated view, zero-copy via mmap).

  (io "load" "tokens/shard-*.bin" :i32)  → ShardedHandle over sorted shards
  (io "load" "train.npy")                → NpyHandle (single file)
"""

import glob as glob_module
import json
import mmap
import os
import pickle
import struct

# Dtype flags → numpy dtype strings.
# Matches the dtype keywords in types.md: :f32, :f16, :bf16, :i32, :i16, :u32, :bool
DTYPE_FLAGS = {"f32", "f16", "bf16", "i32", "i16", "u32", "bool"}

# numpy dtype string ↔ (struct format char, byte width)
_NPY_DTYPE_META = {
    "float32": ("f", 4),
    "float16": ("e", 2),
    "bfloat16": (None, 2),  # no struct char — handled via raw bytes
    "int32": ("i", 4),
    "int16": ("h", 2),
    "uint32": ("I", 4),
    "bool": ("?", 1),
}

# Sheaf flag → numpy dtype string
_FLAG_TO_NPY_DTYPE = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "i32": "int32",
    "i16": "int16",
    "u32": "uint32",
    "bool": "bool",
}


def _parse_npy_header(fd):
    """
    Parse a .npy file header. Returns (dtype_str, shape, data_offset).
    Supports NPY format v1.0 and v2.0.
    Spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
    """
    magic = fd.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("Not a valid .npy file (bad magic bytes)")
    major, minor = struct.unpack("BB", fd.read(2))
    if major == 1:
        header_len = struct.unpack("<H", fd.read(2))[0]
    elif major == 2:
        header_len = struct.unpack("<I", fd.read(4))[0]
    else:
        raise ValueError(f"Unsupported .npy format version: {major}.{minor}")
    header_str = fd.read(header_len).decode("latin1").strip()
    # Header is a Python literal dict: "{'descr': '<f4', 'fortran_order': False, 'shape': (1024,), }"
    header = eval(header_str)  # noqa: S307 — controlled input, .npy files only
    descr = header["descr"]
    shape = header["shape"]
    # descr like '<f4', '|i2', '<i4' — strip endian char, map to dtype name
    _descr_to_dtype = {
        "<f4": "float32",
        ">f4": "float32",
        "=f4": "float32",
        "<f2": "float16",
        ">f2": "float16",
        "=f2": "float16",
        "<i4": "int32",
        ">i4": "int32",
        "=i4": "int32",
        "<i2": "int16",
        ">i2": "int16",
        "=i2": "int16",
        "<u4": "uint32",
        ">u4": "uint32",
        "=u4": "uint32",
        "|b1": "bool",
        "|?1": "bool",
    }
    dtype_str = _descr_to_dtype.get(descr)
    if dtype_str is None:
        raise ValueError(f"Unsupported .npy dtype descriptor: {descr}")
    data_offset = fd.tell()
    return dtype_str, shape, data_offset


class NpyHandle:
    """
    Memory-mapped handle over a single .npy file.
    Dtype and shape are read from the header — no data is loaded into RAM.
    Supports slicing via __getitem__; returns JAX arrays on demand.

    Examples:
        (io "load" "train.npy")             → NpyHandle
        (dynamic-slice dataset 0 1024)      → f32[1024] (reads 4KB from disk)
    """

    def __init__(self, path):
        self._path = path
        with open(path, "rb") as f:
            self._dtype, self._shape, self._data_offset = _parse_npy_header(f)
        _, self._byte_width = _NPY_DTYPE_META[self._dtype]
        # Total number of elements along axis 0
        self._len = self._shape[0] if self._shape else 1
        # Element stride in bytes (all elements after axis 0)
        self._stride = self._byte_width
        for dim in self._shape[1:]:
            self._stride *= dim
        # Open mmap (read-only, shared)
        self._fd = open(path, "rb")
        self._mmap = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """Slice along axis 0. idx can be int, slice, or range."""
        import jax.numpy as jnp
        import numpy as np

        scalar_idx = isinstance(idx, int)
        if scalar_idx:
            if idx < 0:
                idx += self._len
            start, count = idx, 1
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self._len)
            if step != 1:
                raise ValueError("NpyHandle does not support step != 1")
            count = stop - start
        elif hasattr(idx, "__iter__"):
            # range or list of indices — convert to contiguous slice if possible
            indices = list(idx)
            if indices == list(range(indices[0], indices[-1] + 1)):
                start, count = indices[0], len(indices)
            else:
                # Non-contiguous: read each element individually
                return jnp.stack([self[i] for i in indices])
        else:
            raise TypeError(
                f"NpyHandle index must be int, slice, or range, got {type(idx)}"
            )

        byte_start = self._data_offset + start * self._stride
        byte_end = byte_start + count * self._stride
        raw = self._mmap[byte_start:byte_end]

        _np_dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "int32": np.int32,
            "int16": np.int16,
            "uint32": np.uint32,
            "bool": np.bool_,
        }
        arr = np.frombuffer(raw, dtype=_np_dtype_map[self._dtype])
        if len(self._shape) > 1:
            arr = arr.reshape((count,) + self._shape[1:])
        result = jnp.array(arr)
        if scalar_idx and result.ndim > 0:
            result = result[0]  # squeeze leading dim: (1, ...) → (...)
        return result

    def close(self):
        self._mmap.close()
        self._fd.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"NpyHandle(path={self._path!r}, shape={self._shape}, dtype={self._dtype})"
        )


class ShardedHandle:
    """
    Virtual concatenated view over multiple files (glob pattern).
    Each shard is mmap'd independently — no data copied to RAM.
    Slicing computes which shard(s) contain the requested range
    and reads only from those.

    Works with .npy (dtype from header) and .bin (dtype explicit).
    Shards are sorted lexicographically — naming convention matters:
        shard-001.bin, shard-002.bin, ...  ✓
        shard-1.bin, shard-10.bin, ...     ✗ (10 sorts before 2)

    Examples:
        (io "load" "tokens/shard-*.bin" :i32)  → ShardedHandle, 10 shards
        (dynamic-slice dataset 0 4096)         → reads from shard-001 only
    """

    def __init__(self, paths, dtype=None):
        self._shards = []  # list of (cumulative_offset, length, handle_or_path)
        cumulative = 0
        for p in sorted(paths):
            if p.endswith(".npy"):
                handle = NpyHandle(p)
                if dtype is None:
                    dtype = handle.dtype  # infer from first .npy header
                length = len(handle)
                self._shards.append((cumulative, length, handle))
            else:
                # Raw binary: dtype must be provided
                if dtype is None:
                    raise ValueError(
                        f"dtype is required for raw binary shard '{p}'. "
                        f"Supported flags: {sorted(DTYPE_FLAGS)}"
                    )
                file_size = os.path.getsize(p)
                _, byte_width = _NPY_DTYPE_META[dtype]
                length = file_size // byte_width
                self._shards.append((cumulative, length, p))  # path, opened lazily
            cumulative += length
        if dtype is None:
            raise ValueError(
                "ShardedHandle requires at least one shard or an explicit dtype"
            )
        self._dtype = dtype
        _, self._byte_width = _NPY_DTYPE_META[dtype]
        self._total_len = cumulative
        # Cache of open mmap handles for .bin files
        self._bin_mmaps = {}

    @property
    def shape(self):
        return (self._total_len,)

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self._total_len

    def _find_shard(self, global_idx):
        """Binary search: find which shard contains global_idx. Returns (shard_index, local_idx)."""
        lo, hi = 0, len(self._shards) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            offset, length, _ = self._shards[mid]
            if global_idx < offset:
                hi = mid - 1
            elif global_idx >= offset + length:
                lo = mid + 1
            else:
                return mid, global_idx - offset
        raise IndexError(
            f"Index {global_idx} out of range (total length: {self._total_len})"
        )

    def _read_bin(self, shard_idx, local_start, count):
        """Read count elements from a .bin shard at local_start."""
        import jax.numpy as jnp
        import numpy as np

        _, _, path = self._shards[shard_idx]
        if shard_idx not in self._bin_mmaps:
            fd = open(path, "rb")
            self._bin_mmaps[shard_idx] = (
                fd,
                mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ),
            )
        _, mm = self._bin_mmaps[shard_idx]

        byte_start = local_start * self._byte_width
        byte_end = byte_start + count * self._byte_width
        raw = mm[byte_start:byte_end]

        _np_dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "int32": np.int32,
            "int16": np.int16,
            "uint32": np.uint32,
            "bool": np.bool_,
        }
        return jnp.array(np.frombuffer(raw, dtype=_np_dtype_map[self._dtype]))

    def __getitem__(self, idx):
        """Slice along the virtual concatenated axis."""
        import jax.numpy as jnp

        if isinstance(idx, int):
            shard_idx, local_idx = self._find_shard(idx)
            _, _, handle_or_path = self._shards[shard_idx]
            if isinstance(handle_or_path, NpyHandle):
                return handle_or_path[local_idx]
            return self._read_bin(shard_idx, local_idx, 1)[0]

        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._total_len)
            if step != 1:
                raise ValueError("ShardedHandle does not support step != 1")
        elif hasattr(idx, "__iter__"):
            indices = list(idx)
            if indices == list(range(indices[0], indices[-1] + 1)):
                start, stop = indices[0], indices[-1] + 1
            else:
                return jnp.stack([self[i] for i in indices])
        else:
            raise TypeError(
                f"ShardedHandle index must be int, slice, or range, got {type(idx)}"
            )

        # Collect chunks across shard boundaries
        chunks = []
        pos = start
        while pos < stop:
            shard_idx, local_start = self._find_shard(pos)
            _, shard_len, handle_or_path = self._shards[shard_idx]
            # How many elements we can read from this shard
            available = shard_len - local_start
            needed = stop - pos
            count = min(available, needed)

            if isinstance(handle_or_path, NpyHandle):
                chunks.append(handle_or_path[local_start : local_start + count])
            else:
                chunks.append(self._read_bin(shard_idx, local_start, count))

            pos += count

        return jnp.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    def close(self):
        for _, _, handle_or_path in self._shards:
            if isinstance(handle_or_path, NpyHandle):
                handle_or_path.close()
        for fd, mm in self._bin_mmaps.values():
            mm.close()
            fd.close()
        self._bin_mmaps.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"ShardedHandle(shards={len(self._shards)}, total_len={self._total_len}, dtype={self._dtype})"


# Format registry
EXTENSION_TO_FORMAT = {
    ".safetensors": "safetensors",
    ".npy": "npy",
    ".bin": "raw",
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
KNOWN_FORMATS = {"safetensors", "npy", "raw", "pkl", "pickle", "txt", "json", "jsonl"}


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

    # Extract dtype flag from kwargs: (io "load" "tokens.bin" :i32)
    dtype = None
    for k in kwargs:
        if k in DTYPE_FLAGS:
            dtype = _FLAG_TO_NPY_DTYPE[k]
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
        if fmt == "npy":
            paths = sorted(glob_module.glob(path))
            if not paths:
                if os.path.exists(path):
                    paths = [path]
                else:
                    raise FileNotFoundError(f"No .npy files match '{path}'")
            if len(paths) == 1:
                return NpyHandle(paths[0])
            return ShardedHandle(paths)
        if fmt == "raw":
            if dtype is None:
                raise ValueError(
                    '(io "load" "file.bin" :i32) — dtype flag is required for raw binary. '
                    f"Supported: {sorted(DTYPE_FLAGS)}"
                )
            paths = sorted(glob_module.glob(path))
            if not paths:
                if os.path.exists(path):
                    paths = [path]
                else:
                    raise FileNotFoundError(f"No .bin files match '{path}'")
            return ShardedHandle(paths, dtype=dtype)
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
