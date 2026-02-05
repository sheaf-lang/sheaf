#!/usr/bin/env python3
# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.

"""
Sheaf Console - Command-line interface for the Sheaf language.
"""

import argparse
import os
import sys


def main():
    # Custom argument parser that allows positional file argument
    parser = argparse.ArgumentParser(
        description="Sheaf - A Functional Language for Differentiable Computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="sheaf [file.shf] [--trace [MODE]]",
        add_help=False,
    )
    parser.add_argument("file", nargs="?", help="Sheaf file to execute")
    parser.add_argument(
        "--trace",
        nargs="?",
        const=True,
        metavar="FUNCTIONS",
        help="Enable tracing (optionally scope to functions: forward,train-step)",
    )
    parser.add_argument(
        "--trace-out",
        choices=["console", "json"],
        default="console",
        help="Trace output format (default: console)",
    )
    parser.add_argument(
        "--trace-level",
        choices=["fast", "normal", "verbose"],
        default="normal",
        help="Trace detail level (default: normal)",
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message"
    )

    args = parser.parse_args()

    # Help
    if args.help:
        print("""
Sheaf - A Functional Language for Differentiable Computation

Usage:
    sheaf                              Launch interactive console (REPL)
    sheaf <file.shf>                   Execute a Sheaf file
    sheaf init-ai                      Initialize AI context file in current directory
    sheaf --help                       Show this help message

Trace options:
    --trace [FUNCTIONS]                Enable tracing (optionally scope to functions)
    --trace-out {console,json}         Output format (default: console)
    --trace-level {fast,normal,verbose} Detail level (default: normal)
""")
        return

    # No file argument -> check for special commands or launch REPL
    if not args.file:
        from sheaf.repl.__main__ import main as repl_main

        repl_main()
        return

    # Initialize AI context
    if args.file == "init-ai":
        import shutil

        dest = "sheaf-context.md"

        # Check if file already exists
        if os.path.exists(dest):
            print(f"sheaf-context.md already exists in {os.getcwd()}. Skipping.")
            return

        # Find the source files in the package using importlib.resources
        try:
            try:
                from importlib.resources import files
            except ImportError:
                # Fallback for Python < 3.9
                from importlib_resources import files

            # Get package directory
            try:
                assets_dir = files("sheaf").joinpath("assets")
                src_context = str(assets_dir.joinpath("sheaf-context.md"))
                src_reference = str(assets_dir.joinpath("reference.md"))
                if not os.path.exists(src_context):
                    # If that doesn't work, try the direct approach
                    import sheaf as sheaf_module

                    sheaf_dir = os.path.dirname(sheaf_module.__file__)
                    src_context = os.path.join(sheaf_dir, "assets", "sheaf-context.md")
                    src_reference = os.path.join(sheaf_dir, "assets", "reference.md")
            except:
                # Fallback: direct file path
                import sheaf as sheaf_module

                sheaf_dir = os.path.dirname(sheaf_module.__file__)
                src_context = os.path.join(sheaf_dir, "assets", "sheaf-context.md")
                src_reference = os.path.join(sheaf_dir, "assets", "reference.md")

            if not os.path.exists(src_context):
                print(f"Error: sheaf-context.md not found in package", file=sys.stderr)
                sys.exit(1)

            if not os.path.exists(src_reference):
                print(f"Error: reference.md not found in package", file=sys.stderr)
                sys.exit(1)

            # Copy sheaf-context.md and append reference.md
            with open(src_context, "r") as f:
                context_content = f.read()

            with open(src_reference, "r") as f:
                reference_content = f.read()

            # Combine: context + reference header + reference
            combined_content = (
                context_content + "\n\n---\n\n## REFERENCE\n\n" + reference_content
            )

            # Write the combined file
            with open(dest, "w") as f:
                f.write(combined_content)

            print(
                f"AI context with integrated reference copied to {os.path.abspath(dest)}."
            )
            print("\nYou can now ask your agent to read it.")
        except Exception as e:
            print(f"Error: Failed to create sheaf-context.md: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Execute file
    filename = args.file

    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}", file=sys.stderr)
        sys.exit(1)

    # Load and execute Sheaf file
    if not filename.endswith(".shf"):
        print(f"Warning: {filename} doesn't have .shf extension", file=sys.stderr)

    from sheaf import Sheaf
    from sheaf.core.tracer import shf_tracer

    # Enable tracing if requested
    if args.trace:
        shf_tracer.enabled = True
        shf_tracer.monitoring = True
        shf_tracer.mode = args.trace_level

        # Configure scope filter (comma-separated function names)
        # args.trace is either True (trace all) or a string (scope to functions)
        if isinstance(args.trace, str):
            shf_tracer.scope_filter = set(args.trace.split(","))
        else:
            shf_tracer.scope_filter = None

        # Configure output format
        shf_tracer.log_format = args.trace_out

    compiler = Sheaf()

    try:
        compiler.load_file(filename)
        # If there's a main function, call it
        if "main" in compiler.registry:
            result = compiler.registry["main"]()
            if result is not None:
                print(result)
    except Exception as e:
        # Error occurred during load or execution
        if getattr(e, "is_sheaf_error", False):
            # Sheaf error - message is already formatted
            print(str(e), file=sys.stderr)
        else:
            # Unexpected error - print with context
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
