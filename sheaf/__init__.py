import inspect
import os

import jax.numpy as jnp

from .core.compiler import Sheaf as CoreSheaf
from .core.error_handler import install_exception_handler

__version__ = "1.2.0"

# Install error handler automatically when Sheaf is imported
install_exception_handler()


class Sheaf(CoreSheaf):
    def __init__(self, path=None):
        super().__init__()
        if path:
            self.load_from_path(path)

    def _get_external_caller_dir(self):
        """
        Finds the directory of the first script calling Sheaf
        that is not part of the sheaf package.
        """
        stack = inspect.stack()
        # Get the directory where this specific file (__init__.py) is located
        current_package_dir = os.path.dirname(os.path.abspath(__file__))

        for frame_info in stack:
            file_path = os.path.abspath(frame_info.filename)
            # If the file is not inside the sheaf package directory, it's our user!
            if not file_path.startswith(current_package_dir):
                return os.path.dirname(file_path)

        return os.getcwd()

    def load_from_path(self, path):
        if not os.path.isabs(path):
            caller_dir = self._get_external_caller_dir()
            path = os.path.join(caller_dir, path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Sheaf file not found: {path}")

        with open(path, "r") as f:
            source = f.read()
            self._loaded_source = source
            self.load(source)

    def __getattr__(self, name):
        lisp_name = name.replace("_", "-")
        if lisp_name in self.registry:
            return self.registry[lisp_name]
        raise AttributeError(f"Sheaf: function '{lisp_name}' not found.")

    def to_pytree(self, value):
        """
        Convert a Sheaf value to a JAX-compatible pytree.

        Converts Sheaf internal states into pure pytrees containing only:
        - dict
        - list
        - JAX arrays
        - scalars (int, float, bool)

        Args:
            value: A Sheaf value (dict, list, tensor, or scalar)

        Returns:
            A JAX-compatible pytree

        Raises:
            TypeError: If value contains non-serializable types (functions, symbols, etc.)
        """
        if isinstance(value, dict):
            return {k: self.to_pytree(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self.to_pytree(item) for item in value]
        elif isinstance(value, jnp.ndarray):
            return value
        elif isinstance(value, (int, float, bool)) or value is None:
            return value
        else:
            raise TypeError(
                f"Cannot serialize type {type(value).__name__} to pytree. "
                f"Only dict, list, JAX arrays, and scalars are allowed."
            )

    def from_pytree(self, tree):
        """
        Convert a JAX pytree back to a Sheaf value.

        Reconstructs a Sheaf state from a pytree produced by to_pytree.
        This is the inverse operation of to_pytree.

        Args:
            tree: A JAX-compatible pytree

        Returns:
            A Sheaf value (dict, list, tensor, or scalar)
        """
        if isinstance(tree, dict):
            return {k: self.from_pytree(v) for k, v in tree.items()}
        elif isinstance(tree, (list, tuple)):
            return [self.from_pytree(item) for item in tree]
        elif isinstance(tree, jnp.ndarray):
            return tree
        elif isinstance(tree, (int, float, bool)) or tree is None:
            return tree
        else:
            # This should not happen if the pytree came from to_pytree
            raise TypeError(
                f"Unexpected type {type(tree).__name__} in pytree. "
                f"Expected dict, list, JAX array, or scalar."
            )

    def get_registry(self):
        """
        Get metadata about all user-defined functions.

        Returns:
            dict: Mapping from function name to metadata dict containing:
                - params: list of parameter names
                - source: source code string (if available)
        """
        result = {}
        for name, func in self.registry.items():
            meta = {"params": [], "source": None}
            if hasattr(func, "__sheaf_params__"):
                meta["params"] = list(func.__sheaf_params__)
            if hasattr(func, "__sheaf_source__"):
                meta["source"] = func.__sheaf_source__
            result[name] = meta
        return result

    def get_env(self):
        """
        Get metadata about all variables in the environment.

        Returns:
            dict: Mapping from variable name to metadata dict containing:
                - type: string describing the type
                - shape: tuple (for tensors only)
                - dtype: string (for tensors only)
                - value: the actual value (for scalars only)
        """
        result = {}
        for name, val in self.env.items():
            meta = {"type": type(val).__name__}
            if isinstance(val, jnp.ndarray):
                meta["shape"] = tuple(val.shape)
                meta["dtype"] = str(val.dtype)
            elif callable(val):
                meta["type"] = "function"
            elif isinstance(val, (int, float, bool, str)):
                meta["value"] = val
            result[name] = meta
        return result

    def get_special_forms(self):
        """
        Get list of all special forms available in the language.

        Returns:
            list: Sorted list of special form names (defn, let, vmap, scan, etc.)
        """
        return sorted(self.special_forms.keys())

    def __repr__(self):
        """Pretty representation showing loaded functions."""
        n_funcs = len(self.registry)
        if n_funcs == 0:
            return f"<Sheaf (empty)>"

        func_names = ", ".join(sorted(self.registry.keys())[:5])
        if n_funcs > 5:
            func_names += f", ... +{n_funcs - 5} more"

        return (
            f"<Sheaf: {n_funcs} function{'s' if n_funcs != 1 else ''}> [{func_names}]"
        )

    def show(self, name=None):
        """
        Display function source code or loaded code.

        Args:
            name: Optional function name to show. If None, shows all loaded code.
        """
        if name is None:
            # Show all loaded source
            if hasattr(self, "_loaded_source"):
                print(self._loaded_source)
            else:
                print("No source available. Functions loaded:")
                for fname in sorted(self.registry.keys()):
                    print(f"  - {fname}")
        else:
            # Show specific function
            if name in self.registry:
                func = self.registry[name]
                if hasattr(func, "__sheaf_source__"):
                    print(func.__sheaf_source__)
                else:
                    print(f"No source available for '{name}'")
            else:
                print(f"Function '{name}' not found")


__all__ = ["Sheaf"]
