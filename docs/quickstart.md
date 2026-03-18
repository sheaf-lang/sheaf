# Quick Start

Sheaf ships as a single native binary with no runtime dependencies.

## Installation

Download the binary for your platform from the [releases page](https://github.com/sheaf-lang/sheaf/releases):

| Platform              | Binary                |
| --------------------- | --------------------- |
| macOS (Apple Silicon) | `sheaf-macos-aarch64` |
| Linux x86_64          | `sheaf-linux-x86_64`  |
| Linux aarch64         | `sheaf-linux-aarch64` |

Make the binary executable and move it to your PATH:

```bash
chmod +x sheaf-macos-aarch64
mv sheaf-macos-aarch64 /usr/local/bin/sheaf
```

## REPL

Run `sheaf` with no arguments to start the interactive console:

```
Sheaf 2.0.0 (:h for help)
sheaf> (+ 1 2)
=> 3
sheaf> (map (fn [x] (* x x)) [1 2 3 4])
=> [1.0, 4.0, 9.0, 16.0]
```

Tab completion is available. Inline help: `:help <name>` (e.g., `:help let`).

## Running a file

To run a program, pass the source file to `sheaf`:

```bash
sheaf run.shf
```

## Examples

Download the examples archive from the [releases page](https://github.com/sheaf-lang/sheaf/releases). Each example is self-contained:

```bash
cd examples/mlp
sheaf run.shf
```

Available examples:

- **CLEVR**: A neuro-symbolic model for visual question answering
- **Hydra**: A self-modifying model that grows during training
- **MLP**: "Hello World" example of a small multi-layer network to solve XOR
- **NanoGPT**: A port of Karpathy's nanoGPT in Sheaf

## AI assistant bootstrap

`sheaf init-ai` generates a context file in the current directory. Provide this file to Claude Code, Cursor, Copilot, or any other assistant to get code generation, code completion, and debugging help.

```bash
sheaf init-ai
# Generates sheaf-context.md
```

## Next steps

Follow the [tutorial](starting.md) to learn the language fundamentals.
