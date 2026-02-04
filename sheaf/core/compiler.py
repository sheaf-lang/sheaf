# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Sheaf compiler - orchestrates parsing, macro expansion, and compilation.
"""

import builtins
import os
import types

from ..runtime import core_ops, io_ops, jax_ops, math_ops, nn_ops, string_ops
from .error_handler import format_error, set_source
from .forms import special_forms
from .macro_engine import create_macro_engine
from .parser import (
    SheafList,
    SheafRuntimeError,
    SheafSymbol,
    SheafSyntaxError,
    SheafVector,
    parse_full,
)
from .tracer import sheaf_probe, shf_tracer


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class Sheaf:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.lib_dir = os.path.join(self.base_dir, "lib")
        self.load_path = [self.lib_dir, "."]
        self.current_file = None  # Track current file being loaded
        self.loaded_modules = set()  # Track loaded module paths to prevent re-loading

        self.env = self._init_env()
        self.registry = {}
        self.special_forms = special_forms
        self.macro_engine = create_macro_engine()  # Initialize macro engine
        self.macro_engine.compiler = self  # Connect macro engine to compiler
        self.trace = False
        self.dtype = "float32"  # Default dtype (f32)

    def _init_env(self):
        """Initialize the global environment with runtime operations."""
        env = {}
        env.update(core_ops.get_core_env())
        env.update(io_ops.get_io_env())
        env.update(jax_ops.get_jax_env())
        env.update(math_ops.get_math_env())
        env.update(nn_ops.get_nn_env())
        env.update(string_ops.get_string_env())

        env.update(
            {
                "...": Ellipsis,
                # "False": False,
                # "True": True,
                "concat": core_ops.generic_concat,
                "false": False,
                "len": len,
                "nil": None,
                "second": lambda x: x[1],
                "str": str,
                "true": True,
            }
        )
        return env

    def compile(self, exp, local_vars=None):
        """
        Compile a Sheaf S-expression into executable code.

        Pipeline:
        1. Macro expansion
        2. Handle literals and symbols
        3. Dispatch to special forms if applicable
        4. Handle tensor literals
        5. Execute standard function calls
        """
        try:
            if local_vars is None:
                local_vars = {}

            # --- Macros ---
            # Expand macros before compilation
            # Only expand if it's a list (potential macro call)
            if isinstance(exp, list) and len(exp) > 0:
                op = exp[0]
                if isinstance(op, str) and op in self.macro_engine.macros:
                    exp = self.macro_engine.expand(exp, recursive=True)

            # --- Literals ---
            if isinstance(exp, (int, float, bool)):
                return exp

            # --- Symbol Resolution ---
            if isinstance(exp, str):
                return self._resolve_symbol(exp, local_vars)

            # --- Vector Literal (from [] syntax) ---
            # In expression context, evaluate as a list/tuple of values
            if isinstance(exp, SheafVector):
                return self._compile_vector_literal(exp, local_vars)

            # --- Not a list? Return as-is ---
            if not isinstance(exp, list):
                return exp

            # --- Empty list ---
            if len(exp) == 0:
                return []

            op = exp[0]
            args = exp[1:]

            # --- Keyword list, like [:key1 val1 :key2 val2] ---
            if isinstance(op, str) and op.startswith(":"):
                return [self.compile(x, local_vars) for x in exp]

            # --- Check if op is a literal (number, string, tensor) - these cannot be called ---
            # Must check BEFORE tensor literal detection to reject (1 2 5) as invalid
            if isinstance(op, (int, float, bool)):
                hint = "\nHint: Use square brackets [...] instead of parentheses for data literals."
                raise SheafRuntimeError(
                    f"Cannot call a number as a function: {op}{hint}", exp
                )
            if isinstance(op, str) and op.startswith('"'):
                raise SheafRuntimeError(
                    f"Cannot call a string as a function: {op}\nHint: Use square brackets [...] instead of parentheses for data literals.",
                    exp,
                )

            # --- Tensor Literal ---
            if self._is_tensor_literal(exp):
                return self._compile_tensor_literal(exp)

            # --- Internal: Quote (apostrophe syntax) ---
            # Handled here directly, not exposed as a language keyword
            if op == "quote":
                if len(args) != 1:
                    raise ValueError("quote requires exactly one argument")
                expr = args[0]

                # Reject quoted s-expressions '(...) when used as data
                # They should only be used for symbolic/macro purposes
                if (
                    isinstance(expr, SheafList)
                    and hasattr(expr, "_bracket_type")
                    and expr._bracket_type == "("
                ):
                    # Check if it looks like data (numbers/primitives) vs symbolic code (has symbols)
                    # Allow if it contains any SheafSymbol (like 'defn', 'foo', etc.) - it's symbolic code
                    has_symbols = any(isinstance(item, SheafSymbol) for item in expr)
                    has_nested_lists = any(
                        isinstance(item, (SheafList, list)) for item in expr
                    )

                    # Only reject if it's purely data (no symbols, no nested s-expressions)
                    if expr and not has_symbols and not has_nested_lists:
                        raise SheafSyntaxError(
                            f"Type Error: expected a data sequence (Tensor or Literal List), but received a Symbolic S-Expression '{self._format_expr(expr)}. Did you mean to use '[]?",
                            line_num=getattr(expr, "line", None),
                        )

                # For vectors, return as raw Python tuple (useful for shapes)
                if isinstance(expr, SheafVector):

                    def vec_to_tuple(v):
                        result = []
                        for item in v:
                            if isinstance(item, SheafVector):
                                result.append(vec_to_tuple(item))
                            elif (
                                isinstance(item, SheafSymbol)
                                and len(item) >= 2
                                and item[0] == '"'
                                and item[-1] == '"'
                            ):
                                # String literal inside a quoted vector: strip the quotes
                                # so that '["hi" "hello"] becomes ("hi", "hello")
                                result.append(item[1:-1])
                            else:
                                result.append(item)
                        return tuple(result)

                    return vec_to_tuple(expr)
                # For other expressions, return as-is
                return expr

            # --- Internal: Quasiquote (backtick syntax) ---
            # Handled here directly, not exposed as a language keyword
            if op == "quasiquote":
                return self._expand_quasiquote(args[0], local_vars)

            # --- Special Forms Dispatch ---
            if isinstance(op, str) and op in self.special_forms:
                return self.special_forms[op].compile(self, args, local_vars)

            # --- Standard Function Call ---
            return self._compile_function_call(exp, op, args, local_vars)

        except Exception as e:
            if getattr(e, "is_sheaf_error", False):
                raise e

            func_name = local_vars.get("__current_func__", "top-level")
            # Extract filename from expression if available
            filename = (
                getattr(exp, "filename", None) if hasattr(exp, "filename") else None
            )
            formatted_msg = format_error(e, exp, func_name, filename)
            error = SheafRuntimeError(formatted_msg, exp)
            error.original_error = e
            raise error from None

    def _resolve_symbol(self, symbol, local_vars):
        # Resolve a symbol to its value
        # String literal — process escape sequences char by char
        # (replace-chain would mis-handle \\n → backslash+n)
        if symbol.startswith('"') and symbol.endswith('"'):
            raw = symbol[1:-1]
            result = []
            i = 0
            while i < len(raw):
                if raw[i] == "\\" and i + 1 < len(raw):
                    c = raw[i + 1]
                    if c == "n":
                        result.append("\n")
                    elif c == "t":
                        result.append("\t")
                    elif c == '"':
                        result.append('"')
                    elif c == "\\":
                        result.append("\\")
                    else:
                        result.append("\\")
                        result.append(c)
                    i += 2
                else:
                    result.append(raw[i])
                    i += 1
            return "".join(result)

        # Local variable
        if symbol in local_vars:
            return local_vars[symbol]

        # Global environment
        if symbol in self.env:
            return self.env[symbol]

        # Keywords
        if symbol.startswith(":"):
            return symbol

        # Block Python builtins/globals leak
        if hasattr(builtins, symbol) or symbol in globals():
            pass

        line_info = f" (line {symbol.line})" if hasattr(symbol, "line") else ""
        raise NameError(f"Symbol not found{line_info}: '{symbol}'")

    def _is_tensor_literal(self, exp):
        # Check if expression is a tensor literal, like [1 2 3]
        # Only SheafVector (brackets) count as tensor literals, not SheafList (parens)
        if not isinstance(exp, SheafVector) or len(exp) == 0:
            return False

        op = exp[0]
        if isinstance(op, (int, float)):
            return True
        if (
            isinstance(op, SheafVector)
            and len(op) > 0
            and isinstance(op[0], (int, float))
        ):
            return True
        return False

    def _compile_vector_literal(self, exp, local_vars):
        """Compile a vector literal [...].

        In expression context:
        - If all elements are numeric literals → JAX array (e.g., [1 2 3])
        - If nested vectors with all numeric → JAX array (e.g., [[1 2] [3 4]])
        - If contains symbols/vars → tuple (e.g., [D H] for shapes)
        - If contains function calls → tuple (e.g., [(+ 1 2) x])
        """
        import jax.numpy as jnp

        # Check if all elements are pure numeric literals
        all_numeric = all(isinstance(x, (int, float)) for x in exp)

        # Check if nested vectors with all numeric (recursively)
        def is_all_numeric_nested(items):
            for item in items:
                if isinstance(item, (int, float)):
                    continue
                elif isinstance(item, SheafVector):
                    if not is_all_numeric_nested(item):
                        return False
                else:
                    return False
            return True

        if all_numeric or (
            all(isinstance(x, SheafVector) for x in exp) and is_all_numeric_nested(exp)
        ):
            # Pure numeric literal or nested numeric vectors - create JAX array
            def to_list(item):
                if isinstance(item, SheafVector):
                    return [to_list(x) for x in item]
                return item

            data = [to_list(x) for x in exp]

            # Check for explicit dtype metadata from parser: [1 2 3] :bf16
            dtype = self.dtype
            if hasattr(exp, "_dtype"):
                dtype_keyword = exp._dtype
                dtype_map = {
                    ":f16": "float16",
                    ":f32": "float32",
                    ":bf16": "bfloat16",
                    ":i32": "int32",
                    ":u32": "uint32",
                    ":bool": "bool",
                }
                dtype = dtype_map.get(dtype_keyword, self.dtype)

            return jnp.array(data, dtype=dtype)

        # Mixed or contains symbols/vars - return as tuple for shape arguments
        evaluated = tuple(self.compile(x, local_vars) for x in exp)
        return evaluated

    def _expand_quasiquote(self, expr, local_vars):
        """Expand quasiquote (backtick) with unquote (~) and unquote-splicing (~@).

        Internal method - quasiquote is not exposed as a language keyword.
        The backtick syntax is transformed by the parser to (quasiquote ...).
        """
        # Handle unquote: ~expr evaluates expr
        if isinstance(expr, list) and len(expr) > 0:
            first = expr[0]
            if first == "unquote":
                if len(expr) != 2:
                    raise ValueError("unquote (~) requires exactly one argument")
                return self.compile(expr[1], local_vars)

            if first == "unquote-splicing":
                if len(expr) != 2:
                    raise ValueError(
                        "unquote-splicing (~@) requires exactly one argument"
                    )
                return ("__splice__", self.compile(expr[1], local_vars))

        # Handle lists: recurse and process splicing
        if isinstance(expr, list):
            result = []
            for item in expr:
                expanded = self._expand_quasiquote(item, local_vars)
                if (
                    isinstance(expanded, tuple)
                    and len(expanded) == 2
                    and expanded[0] == "__splice__"
                ):
                    spliced = expanded[1]
                    if isinstance(spliced, (list, tuple)):
                        result.extend(spliced)
                    else:
                        result.append(spliced)
                else:
                    result.append(expanded)
            return result

        # Handle vectors: recurse, return as tuple (for shapes)
        if isinstance(expr, SheafVector):
            result = []
            for item in expr:
                expanded = self._expand_quasiquote(item, local_vars)
                if (
                    isinstance(expanded, tuple)
                    and len(expanded) == 2
                    and expanded[0] == "__splice__"
                ):
                    spliced = expanded[1]
                    if isinstance(spliced, (list, tuple)):
                        result.extend(spliced)
                    else:
                        result.append(spliced)
                else:
                    result.append(expanded)
            return tuple(result)

        # Atoms: return as-is
        return expr

    def _is_all_numeric_nested(self, exp):
        """Check if a nested structure contains only numeric literals or simple symbols."""
        for x in exp:
            if isinstance(x, (int, float, str)):
                # Numeric literal or simple symbol
                continue
            elif isinstance(x, (list, SheafVector)):
                if not self._is_all_numeric_nested(x):
                    return False
            else:
                return False
        return True

    def _compile_tensor_literal(self, exp):
        # Compile a tensor literal to JAX array with current dtype or explicit dtype
        import jax.numpy as jnp

        def finalize_literal(item):
            if isinstance(item, (list, SheafVector)):
                return [finalize_literal(x) for x in item]
            return item

        # Check for explicit dtype metadata from parser: [1 2 3] :f32
        dtype = self.dtype
        if hasattr(exp, "_dtype"):
            dtype_keyword = exp._dtype
            dtype_map = {
                ":f16": "float16",
                ":f32": "float32",
                ":bf16": "bfloat16",
                ":i32": "int32",
                ":u32": "uint32",
                ":bool": "bool",
            }
            dtype = dtype_map.get(dtype_keyword, self.dtype)

        return jnp.array(finalize_literal(exp), dtype=dtype)

    def _compile_function_call(self, exp, op, args, local_vars):
        # Compile and execute a standard function call
        try:
            # Check if op is a literal (number, string, tensor) - these cannot be called
            # Lists are allowed because they might be dynamically generated functions
            if isinstance(op, (int, float, bool)):
                hint = ""
                if isinstance(op, (int, float)):
                    hint = "\nHint: Did you mean (max [10 52 8]) or (max '[10 52 8]) ?"
                raise SheafRuntimeError(
                    f"Cannot call a number as a function: {op}{hint}", exp
                )
            if isinstance(op, str) and op.startswith('"'):
                raise SheafRuntimeError(
                    f"Cannot call a string as a function: {op}", exp
                )

            func = self.compile(op, local_vars)

            # Type checks
            if isinstance(func, types.ModuleType):
                raise TypeError(
                    f"Symbol '{op}' resolved to a module instead of a function."
                )

            if not callable(func):
                raise TypeError(f"Symbol '{op}' is not callable (Type: {type(func)}).")

            if isinstance(func, str):
                raise TypeError(f"Unknown function '{func}'.")

            # --- Trace Start: Log call BEFORE evaluating arguments ---
            is_jit_func = getattr(func, "_sheaf_is_jit", False)

            if (
                getattr(self, "trace", False)
                or shf_tracer.enabled
                or shf_tracer.monitoring
            ) and not is_jit_func:
                shf_tracer.log_call(op, [], {})
            elif is_jit_func and (
                getattr(self, "trace", False)
                or shf_tracer.enabled
                or shf_tracer.monitoring
            ):
                shf_tracer.log_jit_call(op)

            # Compile arguments
            real_args, kwargs = self._compile_arguments(
                args, local_vars, op, is_jit_func
            )

            # Execute function
            res = func(*real_args, **kwargs)

            # --- Trace End ---
            if (
                getattr(self, "trace", False)
                or shf_tracer.enabled
                or shf_tracer.monitoring
            ) and not is_jit_func:
                shf_tracer.log_return(op, res)

            return res

        except Exception as e:
            if isinstance(e, SheafRuntimeError):
                raise e

            func_name = local_vars.get("__current_func__", "top-level")
            # Extract filename from expression if available
            filename = (
                getattr(exp, "filename", None) if hasattr(exp, "filename") else None
            )
            formatted_msg = format_error(e, exp, func_name, filename)
            error = SheafRuntimeError(formatted_msg, exp)
            error.original_error = e
            raise error from None

    def _compile_arguments(self, args, local_vars, op, is_jit_func):
        # Compile function arguments, handling both positional and keyword args
        real_args = []
        kwargs = {}
        i = 0
        is_dict_op = op == "dict"

        while i < len(args):
            # Keyword argument detection
            if not is_dict_op and isinstance(args[i], str) and args[i].startswith(":"):
                # Check if this is a flag (keyword without value) or a keyword with value
                key_name = args[i][1:]  # Strip ':' prefix

                # Flag without value: :keepdims, :normalize, etc.
                # Check if next arg is also a keyword or if we're at end of args
                if (i + 1) >= len(args) or (
                    isinstance(args[i + 1], str) and args[i + 1].startswith(":")
                ):
                    # This is a flag - set to True
                    kwargs[key_name] = True
                    i += 1
                else:
                    # This is a keyword with value
                    arg_expr = args[i + 1]
                    is_nested_call = isinstance(arg_expr, list) and len(arg_expr) > 0

                    val = self.compile(arg_expr, local_vars)
                    kwargs[key_name] = val

                    # Log simple values (nested calls log themselves)
                    if (
                        (
                            getattr(self, "trace", False)
                            or shf_tracer.enabled
                            or shf_tracer.monitoring
                        )
                        and not is_nested_call
                        and not is_jit_func
                    ):
                        shf_tracer.log_arg(val, name=key_name)
                    i += 2
            else:
                # Positional argument
                arg_expr = args[i]
                is_nested_call = isinstance(arg_expr, list) and len(arg_expr) > 0

                val = self.compile(arg_expr, local_vars)
                real_args.append(val)

                # Log simple values
                if (
                    (
                        getattr(self, "trace", False)
                        or shf_tracer.enabled
                        or shf_tracer.monitoring
                    )
                    and not is_nested_call
                    and not is_jit_func
                ):
                    shf_tracer.log_arg(val)
                i += 1

        return real_args, kwargs

    def load(self, code, filename="<sheaf>"):
        # Load and compile Sheaf source code
        set_source(code, filename)

        # Save previous file and set current
        prev_file = self.current_file
        self.current_file = filename

        try:
            expressions = parse_full(code, filename)
            for ast in expressions:
                self.compile(ast, {})
        except SheafSyntaxError as e:
            # Create a fake expression with line info for the formatter
            class FakeSyntaxExp:
                def __init__(self, line, filename):
                    self.line = line
                    self.filename = filename

                def __repr__(self):
                    return "<syntax error>"

            exp = FakeSyntaxExp(e.line_num, filename) if e.line_num else None
            formatted_msg = format_error(e, exp, "parsing", filename)
            error = SheafRuntimeError(formatted_msg, exp)
            error.original_error = e
            raise error from None
        finally:
            # Restore previous file
            self.current_file = prev_file

        return self.registry

    def load_file(self, path):
        # Load a Sheaf source file
        import os

        # Get absolute path to track loaded modules
        abs_path = os.path.abspath(path)

        # Mark as loaded to prevent duplicate loads via (use ...)
        self.loaded_modules.add(abs_path)

        with open(path, "r") as f:
            code = f.read()
        return self.load(code, filename=path)

    def _format_expr(self, expr):
        """Format an expression for error messages."""
        if isinstance(expr, SheafList):
            items = " ".join(str(item) for item in expr)
            return f"({items})"
        elif isinstance(expr, SheafVector):
            items = " ".join(str(item) for item in expr)
            return f"[{items}]"
        else:
            return str(expr)

    def __getattr__(self, name):
        # Avoid infinite recursion on special attributes
        if name.startswith("_") or name in ("env", "registry", "special_forms"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Convert Python snake_case to Sheaf hyphen-case
        sheaf_name = name.replace("_", "-")

        if sheaf_name in self.registry:
            return self.registry[sheaf_name]

        raise AttributeError(
            f"Function '{name}' (or '{sheaf_name}') not found in Sheaf registry"
        )
