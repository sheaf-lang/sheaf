# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Control flow special forms: if, case, guard, repeat
"""

from ..parser import SheafRuntimeError
from ..tracer import shf_tracer
from .base import SpecialForm, _warn_parens_in_binding


class IfForm(SpecialForm):
    """if conditional: (if cond then else)"""

    def __init__(self):
        super().__init__("if")

    def compile(self, compiler, args, local_vars):
        cond = compiler.compile(args[0], local_vars)
        return (
            compiler.compile(args[1], local_vars)
            if cond
            else compiler.compile(args[2], local_vars)
        )


class CaseForm(SpecialForm):
    """case pattern matching: (case target val1 result1 val2 result2 ... default)"""

    def __init__(self):
        super().__init__("case")

    def compile(self, compiler, args, local_vars):
        # (case target val1 result1 val2 result2 ... default)
        target_val = compiler.compile(args[0], local_vars)

        # Iterate through pairs
        for i in range(1, len(args) - 1, 2):
            case_val = compiler.compile(args[i], local_vars)
            if target_val == case_val:
                return compiler.compile(args[i + 1], local_vars)

        # If odd number of arguments, the last one is the default
        if len(args) % 2 == 0:
            return compiler.compile(args[-1], local_vars)
        return None


class GuardForm(SpecialForm):
    """guard runtime assertions: (guard :no-nan x) or (guard :shape [64 256] x)"""

    def __init__(self):
        super().__init__("guard")

    def compile(self, compiler, args, local_vars):
        # Format: (guard :type x) or (guard :type expected x)
        shf_tracer.monitoring = True

        guard_type = args[0]

        if guard_type == ":no-nan":
            # (guard :no-nan x)
            val_expr = args[1]
            val = compiler.compile(val_expr, local_vars)
            return shf_tracer.trigger_guard(":no-nan", val)

        elif guard_type in (":shape", ":range"):
            # (guard :shape expected x) or (guard :range expected x)
            # expected must be a literal list, not compiled (we need Python values, not JAX tracers)
            expected_expr = args[1]
            if not isinstance(expected_expr, list):
                raise SheafRuntimeError(
                    f"guard {guard_type} expects a literal list, got {expected_expr}",
                    args,
                )
            # Convert to Python list of concrete values
            expected = [
                float(x) if isinstance(x, (int, float)) else x for x in expected_expr
            ]
            val_expr = args[2]
            val = compiler.compile(val_expr, local_vars)
            return shf_tracer.trigger_guard(guard_type, val, expected)

        raise SheafRuntimeError(f"Unknown guard type: {guard_type}", args)


class DoForm(SpecialForm):
    """sequential evaluation: (do expr1 expr2 ... exprN) → retourne exprN"""

    def __init__(self):
        super().__init__("do")

    def compile(self, compiler, args, local_vars):
        res = None
        for expr in args:
            res = compiler.compile(expr, local_vars)
        return res


class WhileForm(SpecialForm):
    """while loop: (while cond [acc init] body)"""

    def __init__(self):
        super().__init__("while")

    def compile(self, compiler, args, local_vars):
        # Syntax: (while cond [acc init] body)
        cond_expr = args[0]
        binding_acc = args[1]  # [acc_name, init_expr]
        body = args[2]

        _warn_parens_in_binding("while accumulator binding", binding_acc)

        acc_name, init_expr = binding_acc[0], binding_acc[1]
        current_val = compiler.compile(init_expr, local_vars)

        while True:
            loop_ctx = dict(local_vars)
            loop_ctx[acc_name] = current_val

            if not compiler.compile(cond_expr, loop_ctx):
                break

            current_val = compiler.compile(body, loop_ctx)

        return current_val


class RepeatForm(SpecialForm):
    """repeat loop: (repeat [i n] [acc init] body)"""

    def __init__(self):
        super().__init__("repeat")

    def compile(self, compiler, args, local_vars):
        # Syntax: (repeat [i 6] [acc_name init_val] body)
        binding_iter = args[0]  # [i, 6]
        binding_acc = args[1]  # [h, x]

        # Warn if using () instead of [] for bindings
        _warn_parens_in_binding("repeat iterator binding", binding_iter)
        _warn_parens_in_binding("repeat accumulator binding", binding_acc)

        idx_name, count_expr = binding_iter[0], binding_iter[1]
        count = compiler.compile(count_expr, local_vars)

        acc_name, init_expr = binding_acc[0], binding_acc[1]
        current_val = compiler.compile(init_expr, local_vars)

        body = args[2]

        for i in range(int(count)):
            # Context for this iteration
            loop_ctx = dict(local_vars)
            loop_ctx[idx_name] = i
            loop_ctx[acc_name] = current_val  # Inject previous value

            # Evaluate body
            current_val = compiler.compile(body, loop_ctx)

        return current_val
