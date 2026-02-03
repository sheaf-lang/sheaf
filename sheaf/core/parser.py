# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Transform S-expressions into executable abstract syntax trees (AST) for the Sheaf compiler.
"""

import re


class SheafRuntimeError(Exception):
    """Custom exception to carry Lisp context."""

    def __init__(self, message, expression=None):
        super().__init__(message)
        self.expression = expression
        self.is_sheaf_error = True


class SheafSyntaxError(SheafRuntimeError):
    """Syntax error in Sheaf code"""

    def __init__(self, message, line_num=None):
        if line_num is not None:
            message = f"line {line_num}: {message}"
        super().__init__(message)
        self.line_num = line_num
        self.is_sheaf_error = True


class SheafList(list):
    def __init__(self, *args, line=None, filename="<sheaf>"):
        super().__init__(*args)
        self.line = line
        self.filename = filename

    def __format__(self, format_spec):
        if "f" in format_spec:
            raise TypeError(
                f"Attempted to format a SheafList (line {self.line}) as a number. "
                f"Check the return values in your Sheaf function. "
                f"Content snippet: {str(self)[:50]}..."
            )
        return super().__format__(format_spec)


class SheafSymbol(str):
    def __new__(cls, content, line=None, filename="<sheaf>"):
        obj = str.__new__(cls, content)
        obj.line = line
        obj.filename = filename
        return obj


class SheafVector(list):
    """A vector/list literal created with [...] syntax.

    In expression context, this is evaluated as a data literal (like Python list).
    In binding context (first arg to defn, let, fn), it's used for destructuring.
    """

    def __init__(self, *args, line=None, filename="<sheaf>"):
        super().__init__(*args)
        self.line = line
        self.filename = filename
        self._is_vector = True  # Mark as vector literal


def tokenize(chars):
    # Remove comments: both ;; and single ; until end of line
    chars = re.sub(r";.*", "", chars)

    # Treat commas as whitespace (like Clojure), but preserve commas inside strings
    # We'll use a more sophisticated approach to only replace commas outside of strings
    result = []
    i = 0
    while i < len(chars):
        if chars[i] == '"':
            # Found start of string, find the end (handles \" escapes)
            result.append(chars[i])  # Add opening quote
            i += 1
            while i < len(chars) and chars[i] != '"':
                if chars[i] == "\\" and i + 1 < len(chars):
                    result.append(chars[i])  # keep the backslash
                    result.append(chars[i + 1])  # keep the escaped char
                    i += 2
                else:
                    result.append(chars[i])
                    i += 1
            if i < len(chars):
                result.append(chars[i])  # Add closing quote
                i += 1
        elif chars[i] == ",":
            # Replace comma with space (outside of strings)
            result.append(" ")
            i += 1
        else:
            # Keep other characters as-is
            result.append(chars[i])
            i += 1

    chars = "".join(result)

    # Updated pattern to capture backtick (`), tilde (~), and quote (') as separate tokens
    # ~@ must be captured as a single token
    # {} added for dict literals
    token_pattern = r'"([^"\\]|\\.)*"|~@|[()\[\]{}`~\']|[^\s()\[\]{}`~\']+'
    lines = chars.splitlines()
    tokens_with_meta = []
    for line_num, line in enumerate(lines, 1):
        for match in re.finditer(token_pattern, line):
            token = match.group()
            tokens_with_meta.append((token, line_num))
    return tokens_with_meta


def atom(token, line_num, filename="<sheaf>"):
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return SheafSymbol(token, line=line_num, filename=filename)


def parse(tokens, last_func=None, filename="<sheaf>"):
    if not tokens:
        raise SheafSyntaxError(
            "Unexpected end of file - missing closing parenthesis or bracket"
        )

    token_text, line_num = tokens.pop(0)

    # Contextual help: find the function name if we see 'defn'
    if token_text == "defn" and tokens:
        last_func = tokens[0][0]

    # Reader macros: ' ` ~ ~@
    if token_text == "'":
        # Quote: prevent evaluation
        # 'expr => (quote expr)
        return SheafList(
            ["quote", parse(tokens, last_func, filename)],
            line=line_num,
            filename=filename,
        )

    if token_text == "`":
        # Backtick: quasiquote
        # `expr => (quasiquote expr)
        return SheafList(
            ["quasiquote", parse(tokens, last_func, filename)],
            line=line_num,
            filename=filename,
        )

    if token_text == "~":
        # Tilde: unquote
        # ~expr => (unquote expr)
        return SheafList(
            ["unquote", parse(tokens, last_func, filename)],
            line=line_num,
            filename=filename,
        )

    if token_text == "~@":
        # Tilde-at: unquote-splicing
        # ~@expr => (unquote-splicing expr)
        return SheafList(
            ["unquote-splicing", parse(tokens, last_func, filename)],
            line=line_num,
            filename=filename,
        )

    if token_text == "{":
        # Dict literal: {:key1 val1 :key2 val2}
        L = SheafList(["dict"], line=line_num, filename=filename)
        while tokens and tokens[0][0] != "}":
            L.append(parse(tokens, last_func=last_func, filename=filename))

        if not tokens:
            ctx = f" in function `{last_func}`" if last_func else ""
            raise SheafSyntaxError(f"Unclosed brace in dict literal{ctx}", line_num)
        tokens.pop(0)  # consume }

        return L

    if token_text == "[":
        # Vector/list literal: [1 2 3] or [D D] for shapes
        V = SheafVector(line=line_num, filename=filename)
        while tokens and tokens[0][0] != "]":
            V.append(parse(tokens, last_func=last_func, filename=filename))

        if not tokens:
            ctx = f" in function `{last_func}`" if last_func else ""
            raise SheafSyntaxError(f"Unclosed bracket{ctx}", line_num)
        tokens.pop(0)  # consume ]

        # Check for dtype keyword after vector closing bracket: [1 2 3] :f32
        if tokens and tokens[0][0].startswith(":"):
            dtype_token = tokens[0][0]
            valid_dtypes = {
                ":f32",
                ":f16",
                ":bf16",
                ":i32",
                ":u32",
                ":bool",
            }
            if dtype_token in valid_dtypes:
                tokens.pop(0)  # consume dtype keyword
                V._dtype = dtype_token
            # else: it's a keyword argument, not a dtype - don't consume

        return V

    if token_text == "(":
        # S-expression: (op arg1 arg2 ...)
        L = SheafList(line=line_num, filename=filename)
        L._bracket_type = "("
        while tokens and tokens[0][0] != ")":
            L.append(parse(tokens, last_func=last_func, filename=filename))

        if not tokens:
            ctx = f" in function `{last_func}`" if last_func else ""
            raise SheafSyntaxError(f"Unclosed parenthesis{ctx}", line_num)
        tokens.pop(0)  # consume )

        return L
    elif token_text in (")", "]", "}"):
        ctx = f" in function `{last_func}`" if last_func else ""
        raise SheafSyntaxError(
            f"Unexpected closing character '{token_text}'{ctx}", line_num
        )
    else:
        return atom(token_text, line_num, filename)


def parse_full(code, filename="<sheaf>"):
    """Takes raw code and returns a list of expressions (AST)."""
    tokens = tokenize(code)
    expressions = []
    while tokens:
        expressions.append(parse(tokens, filename=filename))
    return expressions
