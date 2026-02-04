#!/usr/bin/env python3
# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Sheaf Console - Interactive REPL for the Sheaf language.

Usage:
    python -m sheaf.repl
    or
    sheaf console
"""

import os
import sys

import jax.numpy as jnp

from sheaf import __version__
from sheaf.core.compiler import Sheaf
from sheaf.core.error_handler import install_exception_handler
from sheaf.core.tracer import shf_tracer
from sheaf.repl.help import get_help

# Try to import readline for line editing and history
try:
    import readline

    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False


class SheafCompleter:
    """Auto-completion for Sheaf REPL."""

    def __init__(self, compiler):
        self.compiler = compiler
        self.commands = [
            ":help",
            ":h",
            ":quit",
            ":q",
            ":exit",
            ":trace",
            ":scope",
            ":env",
            ":registry",
            ":reg",
            ":show",
            ":clear",
        ]

    def complete(self, text, state):
        """Return the state-th completion for text."""
        if state == 0:
            # First call: compute all matches
            if text.startswith(":"):
                # Complete commands
                self.matches = [cmd for cmd in self.commands if cmd.startswith(text)]
            else:
                # Complete function/variable names
                all_names = []

                # Add functions from registry
                if self.compiler.registry:
                    all_names.extend(self.compiler.registry.keys())

                # Add variables from environment
                if self.compiler.env:
                    all_names.extend(self.compiler.env.keys())

                # Add special forms (vmap, scan, let, defn, etc.)
                if self.compiler.special_forms:
                    all_names.extend(self.compiler.special_forms.keys())

                # Filter matches
                self.matches = [name for name in all_names if name.startswith(text)]
                self.matches = sorted(set(self.matches))  # Unique and sorted

        # Return the state-th match, or None if exhausted
        try:
            return self.matches[state]
        except IndexError:
            return None


def is_balanced(text):
    """Check if parentheses and brackets are balanced in the text."""
    stack = []
    pairs = {"(": ")", "[": "]", "{": "}"}

    in_string = False
    in_comment = False

    i = 0
    while i < len(text):
        char = text[i]

        # Handle comments (from ; to end of line)
        if char == ";":
            in_comment = True
        elif char == "\n":
            in_comment = False

        # Skip if we're in a comment
        if in_comment:
            i += 1
            continue

        # Handle strings
        if char == '"':
            in_string = not in_string

        # Skip if we're in a string
        if in_string:
            i += 1
            continue

        # Check brackets/parens
        if char in pairs:
            stack.append(char)
        elif char in pairs.values():
            if not stack:
                return False  # Closing without opening
            opener = stack.pop()
            if pairs[opener] != char:
                return False  # Mismatched

        i += 1

    # All brackets should be closed
    return len(stack) == 0


def format_result(value):
    if isinstance(value, jnp.ndarray):
        # Format JAX arrays with shape and dtype info
        shape_str = "x".join(str(d) for d in value.shape)
        dtype_str = (
            str(value.dtype)
            .replace("bfloat16", "bf16")
            .replace("float32", "f32")
            .replace("int32", "i32")
        )

        # Show actual values for small arrays
        if value.size <= 10:
            return f"Tensor {dtype_str}[{shape_str}] = {value}"
        else:
            # For larger arrays, show shape and stats
            import numpy as np

            mean_val = float(np.mean(value))
            min_val = float(np.min(value))
            max_val = float(np.max(value))

            stats = f"μ={mean_val:.3f} min={min_val:.3f} max={max_val:.3f}"
            return f"Tensor {dtype_str}[{shape_str}] ({stats})"

    elif value is None:
        return "nil"

    elif value is True:
        return "true"

    elif value is False:
        return "false"

    elif isinstance(value, str):
        # Handle keyword symbols (like :yes, :no)
        if value.startswith(":"):
            return value
        else:
            # Regular string - show with quotes
            return repr(value)

    elif isinstance(value, tuple):
        # Format tuples using Sheaf quoted list syntax '[...]
        if not value:
            return "'[]"
        items = ", ".join(format_result(item) for item in value)
        return f"'[{items}]"

    elif isinstance(value, list):
        # Format Python lists using Sheaf quoted list syntax '[...]
        if not value:
            return "'[]"
        items = ", ".join(format_result(item) for item in value)
        return f"'[{items}]"

    elif isinstance(value, dict):
        # Format dictionaries with keys and values (like params trees)
        if not value:
            return "{}"

        # Build dict representation
        items = []
        keys_list = list(value.keys())

        # Show up to 10 items
        for key in keys_list[:10]:
            val = value[key]

            # Format value
            if isinstance(val, jnp.ndarray):
                # JAX tensor: show value if small, else dtype and shape
                shape_str = "x".join(str(d) for d in val.shape)
                dtype_str = (
                    str(val.dtype)
                    .replace("bfloat16", "bf16")
                    .replace("float32", "f32")
                    .replace("int32", "i32")
                )
                if val.size <= 10:
                    val_str = f"{dtype_str}[{shape_str}] = {val}"
                else:
                    val_str = f"{dtype_str}[{shape_str}]"
            elif isinstance(val, dict):
                # Nested dict: format recursively with actual values
                if not val:
                    val_str = "{}"
                else:
                    nested_items = []
                    for k in list(val.keys())[:3]:  # Show first 3 keys of nested dict
                        v = val[k]
                        if isinstance(v, jnp.ndarray):
                            # For arrays, show actual values if small, else show shape
                            if v.size <= 5:
                                # Small array: show values as list
                                arr_str = str(v.tolist())
                            else:
                                # Large array: show dtype and shape
                                shape_str = "x".join(str(d) for d in v.shape)
                                dtype_str = (
                                    str(v.dtype)
                                    .replace("bfloat16", "bf16")
                                    .replace("float32", "f32")
                                    .replace("int32", "i32")
                                )
                                arr_str = f"{dtype_str}[{shape_str}]"
                            nested_items.append(f":{k} {arr_str}")
                        elif isinstance(v, dict):
                            # Deeply nested dict: show as {...}
                            nested_items.append(f":{k} {{...}}")
                        else:
                            # Scalars and other types
                            nested_items.append(f":{k} {repr(v)}")
                    if len(val) > 3:
                        nested_items.append("...")
                    val_str = "{" + " ".join(nested_items) + "}"
            elif isinstance(val, (list, tuple)):
                # List/tuple: show as [...]
                val_str = "[...]"
            elif isinstance(val, str):
                # String: show with quotes
                val_str = repr(val)
            else:
                # Other: show as-is
                val_str = repr(val)

            items.append(f":{key} {val_str}")

        # If more than 10 items, add ellipsis
        if len(keys_list) > 10:
            items.append("...")

        return "{" + ", ".join(items) + "}"

    else:
        # Default Python repr
        return repr(value)


def print_help():
    print("""
Sheaf Console - Interactive REPL

Commands:
  :help, :h [name]    Show help (optionally for a function/special-form)
  :quit, :q, :exit    Exit the REPL
  :trace <mode>       Set trace mode: off, fast, normal, verbose
  :scope <name>       Filter traces to functions matching <name>
  :scope off          Disable scope filtering
  :env                Show current environment (defined variables)
  :registry, :reg     List user-defined functions
  :show <name>        Show value of variable or function source
  :clear              Clear the screen

Expression evaluation:
  Type any Sheaf expression and press Enter to evaluate it.

Examples:
  (+ 1 2)
  (let (x [1 2 3]) (shape x))
  [1 2 3 4] :bf16

Tab: Completes the current command (or lists all commands if empty).

Press Ctrl+C or Ctrl+D to exit.
""")


def run_repl():
    """Run the main REPL loop."""
    # NOTE: We don't install the exception handler here because we handle
    # exceptions manually in the REPL loop for better control

    # Create compiler instance
    compiler = Sheaf()

    # Setup readline history and completion if available
    history_file = os.path.expanduser("~/.sheaf_history")
    if HAS_READLINE:
        try:
            readline.read_history_file(history_file)
            readline.set_history_length(1000)
        except (FileNotFoundError, PermissionError):
            pass  # No history file yet or no permission

        # Setup auto-completion
        completer = SheafCompleter(compiler)
        readline.set_completer(completer.complete)

        # Set delimiters to make Lisp symbols work better
        # Don't break on '-' so we can complete 'multi-head-attention'
        readline.set_completer_delims(" \t\n()[]{}")

        # Configure tab completion (compatible with both GNU readline and libedit)
        if "libedit" in readline.__doc__:
            # macOS uses libedit
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            # Linux/BSD use GNU readline
            readline.parse_and_bind("tab: complete")

    print(f"Welcome to Sheaf Console v{__version__}")
    print("Type :help or :h for help, :quit or :q to exit")
    print()

    while True:
        try:
            # Read input (potentially multi-line)
            try:
                line = input("sheaf> ")

                # Check for unclosed parentheses/brackets - continue reading
                while not is_balanced(line):
                    continuation = input("...    ")
                    line += "\n" + continuation

                # Save complete multi-line command to history (not just last line)
                if HAS_READLINE and line.strip() and "\n" in line:
                    # Remove individual continuation lines from history
                    # and add the complete multi-line command as one entry
                    readline.remove_history_item(
                        readline.get_current_history_length() - 1
                    )
                    readline.add_history(line)

            except EOFError:
                print("\nBye!")
                break

            # Skip empty lines
            if not line.strip():
                continue

            # Handle commands
            if line.startswith(":"):
                cmd_parts = line[1:].strip().split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else None

                if cmd in ("quit", "q", "exit"):
                    print("Bye!")
                    break

                elif cmd in ("help", "h"):
                    # Help - with optional argument for specific symbol
                    if cmd_arg:
                        help_text = get_help(cmd_arg)
                        print(help_text)
                    else:
                        print_help()
                    continue

                elif cmd == "trace":
                    if not cmd_arg:
                        # Show current trace mode
                        status = "enabled" if shf_tracer.enabled else "disabled"
                        mode = shf_tracer.mode if shf_tracer.enabled else "off"
                        print(f"Trace: {status} (mode: {mode})")
                    elif cmd_arg == "off":
                        shf_tracer.enabled = False
                        shf_tracer.monitoring = False
                        print("Trace disabled")
                    elif cmd_arg in ("fast", "normal", "verbose"):
                        shf_tracer.enabled = True
                        shf_tracer.monitoring = True
                        shf_tracer.mode = cmd_arg
                        print(f"Trace enabled: {cmd_arg} mode")
                    else:
                        print(f"Unknown trace mode: {cmd_arg}")
                        print("Valid modes: off, fast, normal, verbose")
                    continue

                elif cmd == "scope":
                    if not cmd_arg:
                        # Show current scope filter
                        if shf_tracer.scope_filter:
                            print(f"Scope filter: {shf_tracer.scope_filter}")
                        else:
                            print("Scope filter: none (tracing all functions)")
                    elif cmd_arg == "off":
                        shf_tracer.scope_filter = None
                        print("Scope filter disabled")
                    else:
                        shf_tracer.scope_filter = cmd_arg
                        print(f"Scope filter set to: {cmd_arg}")
                    continue

                elif cmd == "env":
                    # Show environment
                    registry = compiler.registry
                    env = compiler.env

                    print("Registry (functions):")
                    if registry:
                        for name in sorted(registry.keys()):
                            print(f"  {name}")
                    else:
                        print("  (empty)")

                    print("\nEnvironment (variables):")
                    if env:
                        for name in sorted(env.keys()):
                            val = env[name]
                            if callable(val):
                                print(f"  {name}: <function>")
                            elif hasattr(val, "shape"):
                                shape_str = "x".join(str(d) for d in val.shape)
                                dtype_str = (
                                    str(val.dtype)
                                    .replace("float32", "f32")
                                    .replace("int32", "i32")
                                )
                                print(f"  {name}: Tensor {dtype_str}[{shape_str}]")
                            else:
                                print(f"  {name}: {type(val).__name__}")
                    else:
                        print("  (empty)")
                    continue

                elif cmd in ("registry", "reg"):
                    # List user-defined functions
                    registry = compiler.registry
                    if registry:
                        print("User-defined functions:")
                        for name in sorted(registry.keys()):
                            print(f"  {name}")
                    else:
                        print("No user-defined functions yet.")
                        print("Try: (defn square [x] (* x x))")
                    continue

                elif cmd == "show":
                    # Show variable value or function source
                    if not cmd_arg:
                        print("Usage: :show <name>")
                        continue

                    # Check if it's in registry (user function)
                    if cmd_arg in compiler.registry:
                        func = compiler.registry[cmd_arg]
                        if hasattr(func, "__sheaf_source__"):
                            print(f"⇒ {func.__sheaf_source__}")
                        else:
                            print(f"⇒ <function {cmd_arg}>")
                    # Check if it's in environment (variable or builtin)
                    elif cmd_arg in compiler.env:
                        val = compiler.env[cmd_arg]
                        formatted = format_result(val)
                        if formatted:
                            print(f"⇒ {formatted}")
                        else:
                            print(f"⇒ {val}")
                    else:
                        print(f"Error: '{cmd_arg}' not found")
                        print("Try :env to see all available names")
                    continue

                elif cmd == "clear":
                    # Clear screen
                    os.system("clear" if os.name != "nt" else "cls")
                    continue

                else:
                    print(f"Unknown command: :{cmd}")
                    print("Type :help for available commands")
                    continue

            # Evaluate Sheaf expression
            try:
                from sheaf.core.error_handler import set_source
                from sheaf.core.parser import (
                    SheafRuntimeError,
                    SheafSyntaxError,
                    parse_full,
                )

                # Parse the input
                set_source(line, "<repl>")
                expressions = parse_full(line, "<repl>")

                # Evaluate each expression and keep the last result
                result = None
                for expr in expressions:
                    result = compiler.compile(expr, {})

                # Format and print result
                formatted = format_result(result)
                if formatted:
                    print(f"⇒ {formatted}")

            except KeyboardInterrupt:
                print("\nInterrupted")
                continue
            except (SheafRuntimeError, SheafSyntaxError) as e:
                # Sheaf errors - they come pre-formatted as the message
                # Just print the exception message
                print(f"\n{e}")
                continue
            except Exception as e:
                # Other exceptions - format nicely
                print(f"\nerror: {type(e).__name__}: {e}")
                continue

        except KeyboardInterrupt:
            print("\nUse :quit to exit")
            continue

    # Save history on exit
    if HAS_READLINE:
        try:
            readline.write_history_file(history_file)
        except Exception:
            pass  # Ignore errors saving history


def main():
    """Entry point for the REPL."""
    try:
        run_repl()
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
