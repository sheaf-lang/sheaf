Sheaf emerged in 2025 from a simple frustration with modern machine learning frameworks, and a desire for a modern Lisp for differentiable programming.

While Python works, layers of frameworks accumulated over time to express what the language itself cannot: computation graphs, differentiation, vectorization, and compilation. The graph is not the program itself but a side-effect of imperative control flow and a lot of auxiliary plumbing.

This contrasts with Lisp, which served as an environment for both symbolic and connectionist artificial intelligence research, from early systems on DEC PDP-10 machines to neural network frameworks such as LeCun’s Lush (which ironically later inspired Torch).

Lisp represents computation explicitly: programs are data structures that a system can inspect, transform, and compose, a concept known as homoiconicity. This allows building systems that operate on their own representations.

Sheaf brings homoiconicity to modern machine learning, and with that, the ability for a model to remain mere data that can observe and modify itself using the same tools it uses to manipulate tensors.

The technical inspiration to get there came from Clojure. Clojure demonstrated that a modern Lisp can coexist with a dominant ecosystem without replacing it, but by building on top of its runtime instead. Sheaf tested this idea with an interpreter: a Clojure-inspired syntax for tensor operations, and JAX/XLA as the numerical backend. It then evolved into its own standalone framework based on the same principles, but free from the boundaries of its former host.

Sheaf does not aim to replace existing AI tooling. It provides a space where a model's high-level representation and its machine-manipulable data structures remain the same.
