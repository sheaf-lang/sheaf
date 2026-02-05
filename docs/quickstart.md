# Quick Start Guide

Sheaf is designed to integrate closely with the Python ecosystem, and is distributed as a Python package.
To get started quickly, install it via `pip`:

```bash
pip install sheaf-lang
```

### Getting the Examples

After installation, clone the repository to access example projects:

```bash
git clone https://github.com/sheaf-lang/sheaf
cd sheaf/examples
```

The repository includes four reference implementations in the `examples/` directory:

- **BareGPT** - A generative Transformer trained on Shakespeare
- **Clevr** - A neuro-symbolic model for visual question answering
- **Hydra** - A self-evolving model that grows during training
- **XOR MLP** - The "Hello World" of Sheaf, solving XOR with a tiny neural network

The examples have additional dependencies. Make sure they are installed:

```bash
pip install matplotlib numpy streamlit
```

---

### Sheaf Console

Type `sheaf` to start the Sheaf console and experiment with code:

```bash
Welcome to Sheaf Console v1.1.0
Type :help or :h for help, :quit or :q to exit

sheaf>

```

The Sheaf console comes with autocompletion (`[Tab]` key), and inline help is available with `:help <function-name>` (e.g., `:help let`).

From there, follow the [tutorial](starting.md) to get familiar with the functional approach.

### AI bootstrap

If you are using an AI assistant like Claude Code, Devstral or other LLM-based tools, Sheaf includes a dedicated bootstrap feature.

Running `sheaf init-ai` generates a context file in the current directory. Provide this file to your assistant to help with learning Sheaf, debugging code, or building models.
