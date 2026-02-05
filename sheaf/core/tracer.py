# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Implements real-time telemetry, numerical monitoring, and declarative guards
with backtrace capabilities for JAX computation graphs.
"""

import json
import time

import jax
import jax.numpy as jnp


class Tracer:
    def __init__(self):
        self.enabled = False
        self.monitoring = False
        self.emergency_mode = False
        self.already_traced = False
        self.level = 0
        self.mode = "normal"  # fast, normal, verbose
        self.log_format = "console"  # console or json
        self.json_path = "sheaf_trace.jsonl"
        self.scope_filter = None

        # Ring buffer for backtracking the last 100 operations
        self.ring_buffer = []
        self.ring_buffer_max = 100
        self.silent_monitoring = False  # True when monitoring but not displaying

        # Colors
        self.RED = "\033[91m"
        self.YELLOW = "\033[93m"
        self.BLUE = "\033[94m"
        self.GREEN = "\033[92m"
        self.RESET = "\033[0m"

    def _should_trace(self, op_name):
        # Should this operation be traced?
        if not self.enabled:
            return False

        # If scope_filter is set, only trace functions in the filter
        if self.scope_filter:
            # scope_filter is now a set of function names
            return op_name in self.scope_filter

        return True

    def _format_value(self, val, mode=None):
        # Format based on the current trace mode
        if mode is None:
            mode = self.mode

        # Handle non-tensor types
        if isinstance(val, dict):
            return f"dict(keys:{list(val.keys())})"
        elif isinstance(val, (list, tuple)):
            type_name = type(val).__name__
            return f"{type_name}(len:{len(val)})"
        elif isinstance(val, str):
            return f'"{val}"' if len(val) < 50 else f'"{val[:47]}..."'
        elif isinstance(val, (int, float, bool)):
            return str(val)

        # Handle JAX arrays/tensors
        if not hasattr(val, "shape"):
            return str(val)

        dtype = str(val.dtype).replace("float", "f").replace("int", "i")
        shape_str = "x".join(str(d) for d in val.shape)
        result = f"{dtype}[{shape_str}]"

        # Memory calculation
        bytes_size = val.nbytes
        if bytes_size < 1024:
            mem = f"({bytes_size}B)"
        elif bytes_size < 1024**2:
            mem = f"({bytes_size / 1024:.2f}KB)"
        else:
            mem = f"({bytes_size / 1024**2:.2f}MB)"

        if mode == "fast":
            return f"{result} {mem}"

        # Normal and Verbose: add stats before memory
        try:
            v_min = float(val.min())
            v_max = float(val.max())
            is_finite = bool(jnp.all(jnp.isfinite(val)))

            if mode == "verbose":
                v_mean = float(val.mean())
                result += f" [μ:{v_mean:.2e} min:{v_min:.2e} max:{v_max:.2e}]"
            else:  # normal
                result += f" [min:{v_min:.2e} max:{v_max:.2e}]"

            if not is_finite:
                result += f" {self.RED}[NaN DETECTED]{self.RESET}"

            # Add memory after stats
            result += f" {mem}"
        except:
            # If stats fail, still add memory
            result += f" {mem}"

        return result

    def _log_to_json(self, event_type, op_name, data, level):
        # Log trace event to JSON file
        event = {
            "type": event_type,
            "op": op_name,
            "level": level,
            "timestamp": time.time(),
            "data": data,
        }
        try:
            with open(self.json_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except:
            pass  # Silent fail for JSON logging

    def _add_to_ring_buffer(self, entry):
        # Add entry to ring buffer (FIFO with max size)
        self.ring_buffer.append(entry)
        if len(self.ring_buffer) > self.ring_buffer_max:
            self.ring_buffer.pop(0)

    def reset_ring_buffer(self):
        self.ring_buffer = []
        self.level = 0

    def log_call(self, op_name, args, kwargs):
        # Function call header before evaluating arguments
        # When monitoring is active but trace is not, we do silent monitoring
        should_monitor = self.monitoring and not self.enabled

        if not self._should_trace(op_name) and not should_monitor:
            return

        # Capture level befpre any modification (that's critical with JAX)
        call_level = self.level

        # Increment global level for nested calls and arguments
        self.level += 1

        # Format the output
        indent = "│ " * call_level
        color = self.RED if self.emergency_mode else self.BLUE
        output = f"{indent}{color}├─ [{op_name}]{self.RESET}"

        if should_monitor:
            # Store in ring buffer instead of printing (silent monitoring)
            self._add_to_ring_buffer(output)
        elif self.log_format == "console":
            print(output)
        elif self.log_format == "json":
            self._log_to_json("call", op_name, {}, call_level)

    def log_jit_call(self, op_name):
        # Log JIT-compiled function call with special message
        if not self._should_trace(op_name):
            return

        call_level = self.level
        indent = "│ " * call_level
        color = self.YELLOW

        if self.log_format == "console":
            print(f"{indent}{color}├─ [{op_name}] (JIT-cached){self.RESET}")
            print(
                f"{indent}{color}│ └─ [Sub-calls hidden by JAX JIT cache]{self.RESET}"
            )
        elif self.log_format == "json":
            self._log_to_json("jit_call", op_name, {"cached": True}, call_level)

    def log_arg(self, value, name=None):
        # Log a single argument after it's been evaluated
        should_monitor = self.monitoring and not self.enabled

        if not self.enabled and not should_monitor:
            return

        # Arguments are at current level
        arg_level = self.level

        def _log_argument(val):
            indent = "│ " * arg_level
            # Use verbose mode for monitoring to get full stats
            formatted = self._format_value(
                val, mode="verbose" if should_monitor else self.mode
            )

            if should_monitor:
                # Store in ring buffer
                if name:
                    self._add_to_ring_buffer(f"{indent}├─ → {name}: {formatted}")
                else:
                    self._add_to_ring_buffer(f"{indent}├─ → {formatted}")
            elif self.log_format == "console":
                if name:
                    print(f"{indent}├─ → {name}: {formatted}")
                else:
                    print(f"{indent}├─ → {formatted}")
            elif self.log_format == "json":
                self._log_to_json(
                    "arg", name or "positional", {"value": formatted}, arg_level
                )

        if isinstance(value, (jax.Array, jnp.ndarray)):
            jax.debug.callback(_log_argument, value)
        else:
            _log_argument(value)

    def log_return(self, op_name, result):
        """Log function return value"""
        should_monitor = self.monitoring and not self.enabled

        if not self._should_trace(op_name) and not should_monitor:
            return

        # Capture level before decrementing
        # The return should be at the same level as the call
        return_level = self.level - 1
        start_time = time.perf_counter()

        # Decrement level immediately (before callback executes)
        self.level = max(0, self.level - 1)

        # Capture all variables for the closure
        def _print_return(res):
            elapsed = (time.perf_counter() - start_time) * 1000
            indent = "│ " * return_level

            if self.log_format == "json":
                self._log_to_json(
                    "return",
                    op_name,
                    {
                        "result": self._format_value(
                            res, mode="verbose" if should_monitor else self.mode
                        ),
                        "elapsed_ms": elapsed,
                    },
                    return_level,
                )
            else:
                # Use verbose mode for monitoring to get full stats
                formatted = self._format_value(
                    res, mode="verbose" if should_monitor else self.mode
                )
                # Use microseconds for very fast operations
                if elapsed < 0.01:
                    time_str = f"({elapsed * 1000:.1f}μs)"
                elif elapsed < 1.0:
                    time_str = f"({elapsed:.2f}ms)"
                else:
                    time_str = f"({elapsed:.0f}ms)"

                output = f"{indent}└─ ← {formatted} {time_str}"

                if should_monitor:
                    self._add_to_ring_buffer(output)
                else:
                    print(output)

        # Always print return synchronously to ensure every ├─ has a matching └─
        # Don't use callbacks to avoid timing issues with JIT
        _print_return(result)

    def trigger_guard(self, guard_type, val, expected=None):
        # Check guard conditions and raise exception if violated
        if not self.monitoring:
            return val

        def _check(v):
            if self.already_traced:
                return

            # Auto-convert Python scalars to JAX arrays
            if not isinstance(v, jnp.ndarray):
                v = jnp.asarray(v)

            failed = False
            error_msg = ""

            if guard_type == ":range":
                v_min = float(v.min())
                v_max = float(v.max())
                if v_min < expected[0] or v_max > expected[1]:
                    failed = True
                    error_msg = f"Value range [{v_min:.2e}, {v_max:.2e}] outside allowed range [{expected[0]}, {expected[1]}]"

            elif guard_type == ":no-nan":
                if not jnp.all(jnp.isfinite(v)):
                    failed = True
                    error_msg = "Tensor contains NaN or Inf values"

            elif guard_type == ":shape":
                actual_shape = list(v.shape)
                expected_shape = [int(d) for d in expected]
                if actual_shape != expected_shape:
                    failed = True
                    error_msg = (
                        f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
                    )

            if failed:
                self.emergency_mode = True

                print(f"{self.RED}/!\\ Guard Breached: {guard_type}{self.RESET}")
                # print(f"Error: {error_msg}")
                print(f"Value stats: {self._format_value(v, mode='verbose')}")

                # Display ring buffer backtrace
                if self.ring_buffer:
                    print(
                        f"\nBacktrace (last {len(self.ring_buffer)} operations):{self.RESET}\n"
                    )
                    for entry in self.ring_buffer:
                        print(entry)
                    print("\n--- End of Backtrace ---\n")
                else:
                    print(
                        f"\n{self.YELLOW}(No backtrace available - monitoring was not active){self.RESET}\n"
                    )

                # Exit directly to avoid Python/JAX traceback
                # Use os._exit to bypass exception handling
                import os

                os._exit(1)

        if isinstance(val, (jax.Array, jnp.ndarray)):
            jax.debug.callback(_check, val)
        else:
            _check(val)

        return val


shf_tracer = Tracer()


def sheaf_probe(label, x):
    # Debug probe that prints value info at runtime"""

    def _print_probe(val):
        # Type check: If it's not a JAX array, don't use jnp.isfinite
        is_jax = hasattr(val, "dtype") and hasattr(val, "shape")

        if is_jax:
            is_finite = jnp.all(jnp.isfinite(val))
            color = "\033[91m" if not is_finite else "\033[94m"
            stats = f"{val.dtype}{list(val.shape)} [min:{val.min():.2e} max:{val.max():.2e}]"
        else:
            # Handle tuples (like shapes), lists, or scalars
            is_finite = True
            color = "\033[94m"
            stats = f"{type(val).__name__}: {val}"

        reset = "\033[0m"
        print(f"{color}PROBE [{label}]{reset} {stats}")

    # For JIT compatibility, use debug.callback
    jax.debug.callback(_print_probe, x)
    return x
