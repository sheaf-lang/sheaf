# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Sheaf's error formatting logic
"""

import sys
from typing import Optional


class SheafErrorFormatter:
    def __init__(self):
        self.sources = {}  # filename -> source_code
        self.current_filename = "<sheaf>"

    def set_source(self, code: str, filename: str = "<sheaf>"):
        # Store the original Sheaf source code for error context
        # Keep a map of all loaded files
        self.sources[filename] = code
        self.current_filename = filename

    def get_code_context(self, line_num: int, context_lines: int = 2) -> str:
        # Get lines of code around the error with line numbers
        if not self.source_lines:
            return ""

        lines = []
        start = max(1, line_num - context_lines)
        end = min(len(self.source_lines), line_num + context_lines)

        for i in range(start, end + 1):
            line_text = self.source_lines[i - 1] if i <= len(self.source_lines) else ""
            prefix = "→ " if i == line_num else "  "
            lines.append(f"{prefix}{i:4d} | {line_text}")

        return "\n".join(lines)

    def format_error(
        self,
        error: Exception,
        expression=None,
        func_name: str = "top-level",
        filename: str = None,
    ) -> str:
        # Extract line number if available
        line_num = None
        if hasattr(expression, "line"):
            line_num = expression.line

        # Determine which source file to use
        if filename is None:
            filename = self.current_filename

        # Get source lines for this file
        source_lines = []
        if filename in self.sources:
            source_lines = self.sources[filename].splitlines()

        # Build the error message
        parts = []

        # Error type and message
        error_type = type(error).__name__
        error_msg = str(error)

        # Clean up common Python error messages
        if error_type == "TypeError":
            if "positional argument" in error_msg:
                error_msg = "wrong number of arguments"
            elif "got an unexpected keyword argument" in error_msg:
                error_msg = error_msg.replace(
                    "() got an unexpected keyword argument",
                    " received unexpected parameter",
                )
        elif error_type == "KeyError":
            error_msg = f"key not found: {error_msg}"
        elif error_type == "IndexError":
            error_msg = f"index out of range: {error_msg}"

        # Header with location
        location = f"{filename}"
        if line_num:
            location += f":{line_num}"
        if func_name != "top-level":
            location += f" in `{func_name}`"

        parts.append(f"error: {error_msg}")
        parts.append(f" --> {location}")

        # Show code context if we have line number
        if line_num and source_lines:
            parts.append("    |")
            # Get context lines
            context_lines = 2
            start = max(1, line_num - context_lines)
            end = min(len(source_lines), line_num + context_lines)

            for i in range(start, end + 1):
                line_text = source_lines[i - 1] if i <= len(source_lines) else ""
                if i == line_num:
                    parts.append(f"{i:3} | {line_text}")
                    # Add caret line pointing to error
                    if expression and str(expression) != "<syntax error>":
                        # Try to find the expression in the line
                        expr_str = str(expression)
                        # Strip comments before searching
                        code_part = (
                            line_text.split(";")[0] if ";" in line_text else line_text
                        )
                        if expr_str in code_part:
                            # Use rfind to get the last occurrence in code (usually the actual error)
                            col = code_part.rfind(expr_str)
                            parts.append(f"    | {' ' * col}{'^' * len(expr_str)}")
                        else:
                            # Expression not found literally - try to find first token (e.g., 'defmodel')
                            # or point to the whole line if it's a list expression
                            if isinstance(expression, list) and len(expression) > 0:
                                first_token = str(expression[0])
                                if first_token in line_text:
                                    col = line_text.index(first_token)
                                    parts.append(
                                        f"    | {' ' * col}{'^' * len(first_token)}"
                                    )
                                else:
                                    # Can't find anything - point to the whole non-whitespace part
                                    stripped = line_text.lstrip()
                                    indent = len(line_text) - len(stripped)
                                    parts.append(
                                        f"    | {' ' * indent}{'^' * len(stripped.rstrip())}"
                                    )
                            else:
                                # Single symbol or can't parse - point to whole line
                                stripped = line_text.lstrip()
                                indent = len(line_text) - len(stripped)
                                parts.append(
                                    f"    | {' ' * indent}{'^' * len(stripped.rstrip())}"
                                )
                    else:
                        parts.append(f"    | ^")
                else:
                    parts.append(f"{i:3} | {line_text}")
        else:
            # No line number available - this is a runtime error
            parts.append("    |")
            parts.append(
                "  = note: This error occurred at runtime (not during compilation)."
            )
            parts.append("  = note: The exact line number could not be determined.")
            if func_name != "top-level":
                parts.append(f"  = note: The error occurred in function `{func_name}`.")

        parts.append("    |")

        # Suggestions if we have some...
        suggestion = self.get_suggestion(error, expression)
        if suggestion:
            parts.append(f"  = note: {suggestion}")

        parts.append("")

        return "\n".join(parts)

    # Common misspellings / hallucinations mapped to the correct Sheaf form.
    # Keys are the wrong symbol; values are (correct_symbol, hint_text).
    _KNOWN_MISTAKES = {
        "def": (
            "defn",
            "Sheaf has no 'def'. Use 'defn' to define named functions: (defn name [args] body)",
        ),
        "define": (
            "defn",
            "Sheaf has no 'define'. Use 'defn' to define named functions: (defn name [args] body)",
        ),
        "lambda": (
            "fn",
            "Sheaf has no 'lambda'. Use anonymous functions with 'fn': (fn [args] body)",
        ),
        "set!": (
            None,
            "Sheaf is purely functional — there is no mutation. Use 'let' for new bindings or 'assoc' to update dicts.",
        ),
        "print!": (
            "print",
            "The print function has no '!' suffix in Sheaf. Use: (print ...)",
        ),
        "println": (
            "print",
            "Sheaf uses 'print', not 'println'. Newline is appended automatically.",
        ),
        "var": (
            "let",
            "Sheaf has no 'var'. Use 'let' for local bindings: (let [x 1] ...)",
        ),
        "const": (
            "defn",
            "Sheaf has no 'const'. Export a constant as a zero-arg function: (defn my-const [] value)",
        ),
        "return": (
            None,
            "Sheaf is expression-based — the last expression in a body is its return value. No explicit 'return' needed.",
        ),
        "import": ("use", "Sheaf uses 'use' to load modules: (use nn), (use optim)"),
        "require": ("use", "Sheaf uses 'use' to load modules: (use nn), (use optim)"),
    }

    def get_suggestion(self, error: Exception, expression) -> Optional[str]:
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # JAX TracerBoolConversionError - very common in JIT functions
        if (
            error_type == "TracerBoolConversionError"
            or "boolean conversion of traced array" in error_msg
        ):
            return (
                "Cannot use control flow (if/and/or) with traced values in JIT functions.\n"
                "  = hint: Use 'where' instead of 'if' for differentiable branching:\n"
                "         Replace: (if condition then-expr else-expr)\n"
                "         With:    (where condition then-expr else-expr)"
            )

        # --- Symbol not found: check known mistakes first, then generic advice ---
        if error_type == "NameError" or "symbol not found" in error_msg:
            # Extract the symbol name from "Symbol not found: 'xxx'" or "Symbol not found (line N): 'xxx'"
            symbol = None
            raw = str(error)
            if "'" in raw:
                symbol = raw.split("'")[-2]  # last pair of single quotes

            if symbol and symbol in self._KNOWN_MISTAKES:
                correct, hint = self._KNOWN_MISTAKES[symbol]
                return hint

            return "Check for typos in function or variable names."

        # --- JAX type errors on non-numeric data (strings passed where tensors expected) ---
        if "not a valid jax array type" in error_msg:
            return (
                "A non-numeric value (e.g. a string) was passed to a JAX operation.\n"
                '  = hint: String lists should be created with a quoted vector: \'["a" "b" "c"]\n'
                "         or returned from a function, not passed to tensor operations."
            )

        # --- Integer indexer required (common when using a scalar tensor as an index) ---
        if (
            "indexer must have integer" in error_msg
            or "integer or boolean type" in error_msg
        ):
            return (
                "Tensor indices must be plain integers, not scalar tensors.\n"
                "  = hint: Wrap with (int ...): e.g. (index-update t (int idx) value)"
            )

        # --- Message-based checks (independent of error type) ---
        if "shapes must be" in error_msg:
            return (
                "Shape arguments must be static tuples, not tensors.\n"
                "  = hint: Quote your shape: (zeros '[3 4]) not (zeros [3 4])"
            )

        if "broadcasting" in error_msg:
            return "Arrays have incompatible shapes for broadcasting."

        # --- Type-based checks ---
        if error_type == "TypeError":
            if "not callable" in error_msg:
                return "Make sure you're calling a function, not a value."
            if "argument" in error_msg:
                return "Check the number of arguments you're passing to the function."

        elif error_type == "KeyError":
            return "Verify that the key exists in the dictionary."

        return None


# Global formatter instance
_formatter = SheafErrorFormatter()


def set_source(code: str, filename: str = "<sheaf>"):
    _formatter.set_source(code, filename)


def format_error(
    error: Exception,
    expression=None,
    func_name: str = "top-level",
    filename: str = None,
) -> str:
    return _formatter.format_error(error, expression, func_name, filename)


def install_exception_handler():
    """
    Install a custom exception handler that catches Sheaf errors
    and displays them formatted + without Python traces.
    """
    original_excepthook = sys.excepthook

    def sheaf_excepthook(exc_type, exc_value, exc_traceback):
        # Check if this is a Sheaf error
        if hasattr(exc_value, "is_sheaf_error") and exc_value.is_sheaf_error:
            # This is already a formatted Sheaf error, just print it
            print(str(exc_value), file=sys.stderr)
        elif exc_traceback and "sheaf/core/compiler.py" in str(
            exc_traceback.tb_frame.f_code.co_filename
        ):
            # This is an error that originated in Sheaf but wasn't caught
            # Format it nicely
            formatted = format_error(exc_value)
            print(formatted, file=sys.stderr)
        else:
            # Not a Sheaf error, use default handler
            original_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = sheaf_excepthook
