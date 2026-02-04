# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Binding special forms: defn, lambda, let, defmacro
"""

from ..parser import SheafRuntimeError
from ..tracer import shf_tracer
from .base import SpecialForm, _warn_parens_in_binding


class DefnForm(SpecialForm):
    """defn function definition: (defn name [params] body)"""

    def __init__(self):
        super().__init__("defn")

    def _expr_to_source(self, expr, indent=0):
        """Convert a parsed expression back to readable source code."""
        if isinstance(expr, list):
            if not expr:
                return "[]"

            # Special handling for 'let' bindings
            if len(expr) > 0 and expr[0] == "let":
                # (let [bindings...] body...)
                lines = ["(let ("]
                bindings = expr[1]
                # Group bindings in pairs
                for i in range(0, len(bindings), 2):
                    if i + 1 < len(bindings):
                        var_name = bindings[i]
                        var_val = self._expr_to_source(bindings[i + 1], indent + 4)
                        lines.append(" " * (indent + 6) + f"{var_name} {var_val}")
                lines.append(" " * (indent + 4) + ")")
                # Body expressions
                for body_expr in expr[2:]:
                    lines.append(
                        " " * (indent + 2) + self._expr_to_source(body_expr, indent + 2)
                    )
                lines.append(" " * indent + ")")
                return "\n".join(lines)

            # Format as multi-line if contains nested lists
            has_nested = any(isinstance(e, list) for e in expr)
            if has_nested and len(expr) > 2:
                lines = ["("]
                for i, e in enumerate(expr):
                    if i == 0:
                        lines[0] += self._expr_to_source(e, indent)
                    else:
                        lines.append(
                            " " * (indent + 2) + self._expr_to_source(e, indent + 2)
                        )
                lines.append(" " * indent + ")")
                return "\n".join(lines)
            else:
                # Simple one-liner
                items = " ".join(self._expr_to_source(e, indent) for e in expr)
                return f"({items})"
        elif isinstance(expr, str):
            # Keep strings as-is
            if expr.startswith(":") or expr.startswith('"'):
                return expr
            return expr
        else:
            # Numbers, etc.
            return str(expr)

    def compile(self, compiler, args, local_vars):
        # Check if :jit flag is at the end of body
        # Syntax: (defn name [params] body... [:jit])
        is_jit = len(args) > 2 and args[-1] == ":jit"

        name = args[0]
        params = args[1]
        body = args[2:-1] if is_jit else args[2:]

        # Check if trying to shadow a special form as function name
        if hasattr(compiler, "special_forms") and name in compiler.special_forms:
            raise SheafRuntimeError(
                f"Error: Cannot shadow special form '{name}'. It is a reserved keyword.\n"
                f"Special forms like 'fn', 'get', 'let', etc. cannot be used as function names.",
                args,
            )

        # Check if any parameter name shadows a special form
        if hasattr(compiler, "special_forms"):
            for param in params:
                if param in compiler.special_forms:
                    raise SheafRuntimeError(
                        f"Error: Cannot use special form '{param}' as a parameter name. It is a reserved keyword.\n"
                        f"Special forms like 'fn', 'let', 'if', etc. cannot be used as parameter names.",
                        args,
                    )

        # Warn if using () instead of [] for parameters
        _warn_parens_in_binding("function parameters", params)

        def generated_func(*input_args, **kwargs):
            # 1. Check if 'trace' was passed in this specific call
            trace_call = kwargs.pop("trace", False)
            log_call = kwargs.pop("log", "console")
            scope_call = kwargs.pop("scope", None)

            # 2. Setup tracing if needed
            original_trace_state = getattr(compiler, "trace", False)
            if trace_call:
                compiler.trace = True
                shf_tracer.enabled = True
                shf_tracer.monitoring = True
                shf_tracer.level = 0

                shf_tracer.mode = (
                    trace_call if isinstance(trace_call, str) else "normal"
                )
                shf_tracer.log_format = log_call
                shf_tracer.scope_filter = scope_call

                if scope_call:
                    print(
                        f"--- Selective Tracing: {scope_call} (Mode: {shf_tracer.mode}) ---"
                    )
                else:
                    print(
                        f"--- Tracing Sheaf Function: {name} [Mode: {shf_tracer.mode}] ---"
                    )

            try:
                # 3. Standard execution logic
                # NOTE: We start with an empty context, not local_vars from definition time.
                # The only variables in scope are the function parameters.
                # Handle both positional args and kwargs
                context = dict(zip(params, input_args))
                # Add remaining kwargs (convert snake_case to match Sheaf params)
                for key, value in kwargs.items():
                    # Convert Python snake_case to Sheaf hyphen-case if needed
                    sheaf_key = key.replace("_", "-")
                    if sheaf_key in params:
                        context[sheaf_key] = value
                    elif key in params:
                        context[key] = value
                context["__current_func__"] = name

                res = None
                for expression in body:
                    res = compiler.compile(expression, context)

                return res

            finally:
                # 4. Clean up trace state
                if trace_call:
                    shf_tracer.enabled = False
                    compiler.trace = original_trace_state

        # Apply JAX JIT if requested
        if is_jit:
            import jax

            # HashableDict is defined in compiler.py but we need to avoid circular import
            # So we import it locally here
            class HashableDict(dict):
                def __hash__(self):
                    return hash(tuple(sorted(self.items())))

            static_argnums = []
            if "config" in params:
                static_argnums.append(params.index("config"))

            # Create a wrapper to ensure dictionaries are hashable for JAX
            base_func = generated_func
            # Create the jit object once — reused across all calls
            jitted_fn = jax.jit(base_func, static_argnums=tuple(static_argnums))

            def jitted_wrapper(*args, **kwargs):
                # Extract trace/log/scope kwargs BEFORE passing to jax.jit
                trace_kwarg = kwargs.pop("trace", False)
                kwargs.pop("log", None)  # Remove but ignore
                kwargs.pop("scope", None)  # Remove but ignore

                # Warn user if they try to trace a JIT function
                if trace_kwarg:
                    print(
                        f"Warning: Cannot trace JIT-compiled function '{name}'. Tracing disabled."
                    )

                new_args = list(args)
                for idx in static_argnums:
                    if isinstance(new_args[idx], dict):
                        new_args[idx] = HashableDict(new_args[idx])

                return jitted_fn(*new_args, **kwargs)

            generated_func = jitted_wrapper
            generated_func._sheaf_is_jit = True

        # Check for redefinition
        if name in compiler.registry or name in compiler.env:
            # Determine where it's defined
            location = "user code" if name in compiler.registry else "standard library"

            raise SheafRuntimeError(
                f"Error:\nFunction '{name}' is already defined in {location}. "
                f"Redefinition is not allowed to prevent shadowing bugs.",
                args,
            )

        # Store source code for inspection in REPL
        params_str = "[" + " ".join(str(p) for p in params) + "]"
        source_lines = [f"(defn {name} {params_str}"]
        for expr in body:
            source_lines.append("  " + self._expr_to_source(expr, 2))
        if is_jit:
            source_lines.append("  :jit)")
        else:
            source_lines.append(")")
        generated_func.__sheaf_source__ = "\n".join(source_lines)
        generated_func.__sheaf_name__ = name
        generated_func.__sheaf_params__ = params

        # Register the function
        compiler.registry[name] = generated_func
        compiler.env[name] = generated_func
        return generated_func


class LambdaForm(SpecialForm):
    """lambda anonymous function: (lambda [params] body)"""

    def __init__(self):
        super().__init__("lambda")

    def compile(self, compiler, args, local_vars):
        # Format: (lambda [params] body)
        l_params, *l_body = args

        # Warn if using () instead of [] for parameters
        _warn_parens_in_binding("lambda parameters", l_params)

        # Capture the current local_vars at definition time
        def anonymous_func(*l_args, closure_env=dict(local_vars)):
            # Merge closure_env with current lambda arguments
            l_context = dict(closure_env)
            l_context.update(dict(zip(l_params, l_args)))

            res = None
            for expr in l_body:
                res = compiler.compile(expr, l_context)
            return res

        return anonymous_func


class LetForm(SpecialForm):
    """let local bindings: (let [var1 val1 var2 val2] body)"""

    def __init__(self):
        super().__init__("let")

    def compile(self, compiler, args, local_vars):
        # args is [bindings_list, body_exp1, body_exp2, ...]
        bindings, *body = args

        # Warn if using () instead of [] for bindings
        _warn_parens_in_binding("let bindings", bindings)

        # Copy context to avoid polluting parent scope
        current_context = dict(local_vars)

        # Process pairs: (var1 val1 var2 val2 ...)
        for i in range(0, len(bindings), 2):
            target = bindings[i]
            # Compile the value using the context updated by previous pairs
            val = compiler.compile(bindings[i + 1], current_context)

            if isinstance(target, list):  # Support for [a b] (split key)
                for name, v in zip(target, val):
                    current_context[name] = v
            else:
                current_context[target] = val

        # Execute the body with the final context
        res = None
        for expression in body:
            res = compiler.compile(expression, current_context)
        return res


class DefmacroForm(SpecialForm):
    """defmacro macro definition: (defmacro name [params] body)"""

    def __init__(self):
        super().__init__("defmacro")

    def compile(self, compiler, args, local_vars):
        """
        Define a macro at compile-time.

        Syntax: (defmacro name [params] body-template)

        Example:
            (defmacro when [cond body]
              `(if ~cond ~body nil))
        """
        if len(args) < 3:
            raise ValueError("defmacro requires name, params, and body")

        name = args[0]
        params = args[1]
        body_template = args[2]  # Usually a quasiquoted expression

        # Create an expander function
        def expander(macro_args):
            # Bind macro arguments to parameters
            bindings = compiler.macro_engine._bind_params(params, macro_args)
            # Substitute in the template
            return compiler.macro_engine._substitute(body_template, bindings)

        # Register the macro in the macro engine
        compiler.macro_engine.defmacro_native(name, expander)

        # defmacro doesn't return a runtime value
        return None
