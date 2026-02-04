# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
String operations for Sheaf.

Provides a generic dispatch mechanism for string methods.
Only imported when needed to avoid runtime bloat.
"""

import re


def str_call(method_name, s, *args):
    """
    We provide only one generic operation instead of reimplementing Python in Sheaf...

    Examples:
        (str-call "upper" "hello")           -> "HELLO"
        (str-call "startswith" "hello" "he") -> True
        (str-call "replace" "foo" "o" "a")   -> "faa"
        (str-call "split" "a,b,c" ",")       -> ["a", "b", "c"]

    Args:
        method_name: name of the string method to call
        s: the string to operate on
        *args: additional arguments to pass to the method

    Returns:
        result of calling the method

    Raises:
        AttributeError: if the method doesn't exist
    """
    s_str = str(s)
    method = getattr(s_str, method_name, None)

    if method is None:
        raise AttributeError(f"String has no method '{method_name}'")

    if not callable(method):
        raise AttributeError(f"String attribute '{method_name}' is not callable")

    return method(*args)


_FORMAT_PLACEHOLDER = re.compile(r"\{[^}]*\}")


def sheaf_print(*args, **kwargs):
    """
    print with auto-format: if the first arg is a string containing format
    placeholders ({}, {:>3}, {:.4f}, etc.) and there are extra positional
    args, format it before printing.

        (print "Loss: {:.4f}" loss)        -> print(f"Loss: {loss:.4f}")
        (print "hello")                    -> print("hello")
        (print "x" :end "")               -> print("x", end="")
    """
    if (
        len(args) >= 2
        and isinstance(args[0], str)
        and _FORMAT_PLACEHOLDER.search(args[0])
    ):
        formatted = args[0].format(*args[1:])
        print(formatted, **kwargs)
    else:
        print(*args, **kwargs)


def get_string_env():
    return {
        "print": sheaf_print,
        "str-call": str_call,
    }
