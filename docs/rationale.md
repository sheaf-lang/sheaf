Sheaf started in 2025 from a frustration with modern machine learning frameworks, and the desire for a modern Lisp for differentiable programming.

While Python works, layers of frameworks accumulated over time to express what the language itself cannot: computation graphs, differentiation, vectorization, and compilation. This also comes at a cost: much of the mathematical structure of models is now expressed indirectly, through imperative control flow and auxiliary plumbing.

This contrasts with Lisp, which served as an environment for both symbolic and connectionist artificial intelligence research, from early systems on DEC PDP-10 machines to neural network frameworks such as LeCun’s Lush.

Lisp represents computation explicitly. Programs are data structures that a system can inspect, transform, and compose, a concept known as homoiconicity. This allows building systems that operate on their own representations.

Sheaf aims to bring homoiconicity to modern machine learning, and with that, the ability for a model to remain mere data that can observe and modify itself using the same tools it uses to manipulate tensors.

The technical inspiration to get there came from Clojure. Clojure demonstrated that a modern Lisp can coexist with a dominant ecosystem without replacing it, but by building on top of its runtime instead. Sheaf follows the same approach: a functional layer for model description, delegating the numerical execution to XLA. Its syntax is also inspired by Clojure, adapted for tensor operations.

Sheaf does not aim to replace existing AI tooling. It provides a space where a model's high-level representation and its machine-manipulable data structures remain the same.
