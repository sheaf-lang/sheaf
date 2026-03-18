# Starting with Sheaf

### The Functional Mindset

Sheaf is a _functional_ language rather than an _imperative_ one. An imperative language like Python instructs the computer to change its internal state step-by-step. In Sheaf, transformations are defined by how data flows through nested functions to produce a result.

Functional programming avoids side effects, which makes code deterministic. It maps naturally to computational graphs and machine learning workloads.

### A first example

A functional language can appear confusing at first, because even a classic `val = 2 ; print val` example does not translate directly. Since "everything is a function", we would write a function `(defn val [] 2)` and then call it: `(val)`. Or, alternatively, evaluate both immediately: `((defn val [] 2))`

In Sheaf, as in all Lisp dialects, the fundamental unit of computation is the _expression_. There is no structural difference between a mathematical operator, a control flow structure, or a complex neural layer: they all follow the same notation.

For this reason, a global "variable" is effectively a nullary function (a function with no arguments) that returns a constant.

To define a function, we use the `defn` form, followed by the function name and its arguments enclosed in square brackets `[]` rather than parentheses `()`. This is borrowed from Clojure to differentiate the parameter list from the function calls.

In the REPL:

```
sheaf> (defn val [] 2)
=> <fn:val>
sheaf> (val)
=> 2
```

Adding `2` to it (Sheaf uses prefix notation, covered below):

```
sheaf> (+ 2 val)     ;; This won't work: val is a function, not a value
error: Expected numeric values, got int and function
 --> <repl>:1

sheaf> (+ 2 (val))
=> 4

sheaf> (defn val [] 4)
error: Redefinition of 'val' is not allowed.
```

The inability to change the value of `val` may come as surprise. In Sheaf, every value is immutable: once a function is defined, it cannot be changed. This is required both for predictability and for the JIT compiler.

### Expressions and Lambdas

When an expression like `(+ 1 2)` is evaluated, the interpreter evaluates a _form_ where the symbol `+` refers to a function, and `1` `2` are its arguments.

Conceptually, a Sheaf program is not a sequence of instructions, but a single, nested tree of expressions.

While `(+ 1 2)` will execute immediately, it can also be explicitly defined as a lambda with `(fn [] (+ 1 2))`.
When defined with `fn`, the logic won't be executed immediately but will return an anonymous function that can be called from within the parent function.
This is used when a small transformation is needed just once, without giving the function a permanent name.

```
sheaf> (* 5 10)
=> 50
sheaf> (* x 10)
error: Undefined symbol: x
 --> <repl>:1

sheaf> ((fn [x] (* x 10)) 5)
=> 50
```

Note the double `(( ))` to call the lambda immediately. Without them, the expression defines the function but does not apply it:

```
sheaf> (fn [x] (* x 10))
=> <function>
```

### Local bindings

Defining a function to return a scalar value isn't very helpful. Usually, Sheaf functions apply transformations to tensors and pass them to other functions.
Sheaf doesn't _assign variables_; it _binds local symbols_ with the form `let`. Variables exist within the function's scope and disappear after the evaluation is complete.

```
sheaf> (let [x [1 2 3]]
     >   x)
=> [1. 2. 3.]

sheaf> (let [x [1 2 3]]
     >   (+ x 2))
=> [3. 4. 5.]
```

Because of immutability, `x` is also not modified in the second example. Instead, a new tensor is created and returned. Once the command ends, it is no longer accessible.

### Prefix Notation (Polish Notation)

Another confusing aspect in our example may be the operation order: `(+ x 2)` instead of `(x + 2)`.

Sheaf, like Clojure, Scheme and other Lisps, use prefix notation, where the operator always precedes its operands.

This approach serves two critical purposes:

- Variadic Operations: Functions can take any number of arguments without repeating the operator. Instead of `1 + 2 + 3 + 4`, one simply writes `(+ 1 2 3 4)`.

- Consistency: Since everything between parens is a function, mathematical operators, built-in functions, and user-defined functions all follow the exact same structure: `(function argument1 argument2 ...)`. This uniformity is what allows the language to treat code as data.

The first symbol after an opening parenthesis is always a function.

### Conditionals: `if` and `where`

Most programming languages have an `if` statement for branching logic. Sheaf has two distinct forms for conditional evaluation, and understanding when to use each is important.

The `if` form works like a traditional conditional: it evaluates a condition, then returns either the "then" branch or the "else" branch:

```
sheaf> (if (> 5 3) :yes :no)
=> :yes
sheaf> (if false "not returned" "returned")
=> "returned"
```

This works for decisions based on configuration values or scalars. However, `if` cannot be used with tensor values inside JIT-compiled functions. The compiler needs to know the control flow at compile time, and tensor values are only known at runtime.

```clojure
(defn broken [x]
  (if (> x 0) x 0))  ;; This will fail in JIT context
```

For element-wise conditionals on tensors, `where` is used instead. It evaluates both branches and selects values based on a boolean mask:

```
sheaf> (where (> [1 5 3 8] 4) [1 5 3 8] 0)
=> [0. 5. 0. 8.]
```

A common pattern is implementing ReLU (Rectified Linear Unit), which returns `x` if positive, else `0`:

```clojure
(defn my-relu [x]
  (where (> x 0) x 0))
```

In summary: use `if` for static branching (configs, flags), and `where` for tensor operations.

### Recursion

Sheaf does not have imperative loops. Iteration is handled through higher-order functions.

A typical loop in Python:

```python
# Imperative loop
>>> total = 0
>>> for i in range(10):
>>>     total += i
>>> print(total)
45

# Vectorized imperative loop
>>> import numpy as np
>>> np.sum(range(10))
np.int64(45)
```

In Sheaf, use `reduce`:

```
sheaf> (reduce + 0 (range 10))
=> 45.0
```

`reduce` takes a function (here, `+`), a starting number (`0`), and a sequence on which to apply the function. For instance, `(reduce + 0 [1 2 3 4])` will result in adding `0 + 1 + 2 + 3 + 4`.

Other popular higher order functions are `map`, `vmap` and `scan`.

Unlike `reduce` which returns a single combined value, `map` applies a transformation to every element of an array and returns another array.

Squaring all values from 1 to 10:

```
sheaf> (map (fn [i] (* i i)) (range 1 11))
=> [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]
```

`vmap` is similar to `map`, but vectorizes the operation over the entire array:

```
sheaf> ((vmap (fn [i] (* i i)) 0) (range 1 11))
=> tensor i32[10] = [  1   4   9  16  25  36  49  64  81 100]
```

`scan` carries a state across a sequence (like `reduce`) but also returns all intermediate results. In differentiable programming, `scan` is the preferred way to implement recurrent structures or deep loops, as it allows the compiler to optimize backpropagation through time.

### Working with Parameters

In neural networks, parameters (weights and biases) are typically stored in nested dictionaries. Sheaf provides several tools to work with these structures elegantly.

The `get` function retrieves a value from a dictionary by key:

```
sheaf> (get {:learning-rate 0.001 :epochs 10} :learning-rate)
=> 0.001
```

For deeply nested structures, `get-in` navigates through multiple levels using a path vector:

```
sheaf> (get-in {:model {:layer1 {:W [[1 2] [3 4]] :b [0.1 0.2]}}} '[:model :layer1 :b])
=> [0.1 0.2]
```

A default value can be provided if the path does not exist:

```
sheaf> (get-in {:a 1} '[:missing :path] 42)
=> 42
```

When writing neural network layers, repeatedly calling `get` becomes tedious. The `with-params` form unpacks dictionary keys directly into the local scope. This is called _parameters destructuring_:

```
sheaf> (let [params {:W [[1 2] [3 4]] :b [0.5 0.5]}
     >       x [1.0 1.0]]
     >   (with-params [params]
     >     (+ (@ x W) b)))
=> [4.5 6.5]
```

Without `with-params`, the equivalent code would require explicit `get` calls:

```clojure
;; More verbose alternative
(let [params {:W [[1 2] [3 4]] :b [0.5 0.5]}
      x [1.0 1.0]]
  (+ (@ x (get params :W)) (get params :b)))
```

For nested parameter structures, a key can be specified to destructure a sub-dictionary:

```
sheaf> (let [model {:layer1 {:W [[1 0] [0 1]] :b [0 0]}
     >             :layer2 {:W [[2 2] [2 2]] :b [1 1]}}
     >       x [1.0 2.0]]
     >   (with-params [model :layer1]
     >     (+ (@ x W) b)))
=> [1. 2.]
```

Since all values are immutable, dictionaries are updated by creating new ones. The `assoc` form adds or updates keys:

```
sheaf> (assoc {:a 1} :b 2 :c 3)
=> {:a 1, :b 2, :c 3}
```

The `merge` function combines multiple dictionaries (later values override earlier ones):

```
sheaf> (merge {:lr 0.001 :epochs 10} {:lr 0.0001})
=> {:lr 0.0001, :epochs 10}
```

And `dissoc` removes keys:

```
sheaf> (dissoc {:a 1 :b 2 :c 3} [:a :c])
=> {:b 2}
```

These operations are essential for managing optimizer state, where parameters, momentum, and velocity are all stored in nested dictionaries that must be updated at each training step.

### The threading operators

In older Lisps such as Scheme, Common Lisp, or Emacs Lisp, nesting functions can sometimes become a "parentheses nightmare." One of the great innovations of Clojure is the threading operators `->` and `as->`.

These operators "thread" a value through a sequence of functions, making the code read like a pipeline from top to bottom.

`->` (_Thread-first_) automatically injects the result of the previous expression as the first argument of the next function call. It is ideal for linear transformations.

`as->` (_Thread-as_) binds the result to a specific symbol (a variable, in imperative talk), allowing us to place it anywhere in the next expressions. This is more typically used for operations where the input must be reused.

```
;; Traditional nesting
((fn [x] (* x x)) (* (+ 10 5) 2))
;; => 900

;; Threaded version
(as-> 10 x
  (+ x 5)
  (* x 2)
  (* x x))
;; => 900
```

In the NanoGPT example, both threading operators describe the flow of a tensor through layers:

```clojure
(defn transformer-block [x layer-p config]
  ;; as-> binds the initial 'x' to the name 'h'
  ;; This allows us to reuse 'h' multiple times within the block,
  ;; specifically for the residual connection at the end
  (as-> x h

    ;; We now switch to '->' because each function takes the output of the
    ;; previous one as its first argument, somewhat like with a shell pipe
    (-> h
        (layer-norm (get layer-p :ln1))
        (multi-head-attention layer-p config)
        (first) ;; Get the attention output, ignore weights
        (+ h))  ;; Residual 1: we explicitly reference 'h' named above

    ;; We re-bind the result of the previous block to 'h'
    (as-> h
        (-> h
            (layer-norm (get layer-p :ln2))
            (mlp (get layer-p :mlp))
            (+ h))))) ;; Residual 2
```

### Reading and writing files

In a functional language, file I/O cannot be treated as an ordinary function call. Reading from or writing to disk depends on external state, and therefore breaks determinism.

In Sheaf, all interactions with the file system are explicitly routed through a single form: `io`. This makes side effects visible in the code and prevents them from leaking into otherwise pure transformations.

**The io Form**

The `io` form takes a verb as its first argument, followed by verb-specific parameters. Each verb describes a concrete operation on external data, such as loading a file, saving a value, or streaming content.

This approach keeps the language core small while avoiding a proliferation of specialized I/O primitives.

Here is a simple example:

```clojure
;; Loading a configuration file into a dictionary
(let [config (io "load" "model_config.json")]
  (print (get config :layers)))

;; Reading a dataset as a raw string
(let [corpus (io "read" "data/shakespeare.txt")]
  (print "Length:" (len corpus)))

```

Depending on the verb and the file format, `io` may return an in-memory value or a lazy view over the underlying data.

**Large Files and Handles**

For large datasets, eagerly loading files into memory is often impractical. Instead, Sheaf uses _Handles_: lazy, memory-mapped views over file-backed data.

A Handle behaves like a regular collection, but data is only read from disk when it is accessed. This allows programs to work with datasets that are much larger than available RAM.

| Action | Form                  | Result                                |
| ------ | --------------------- | ------------------------------------- |
| Import | (io "load" path)      | A value, dictionary, or a lazy Handle |
| Export | (io "save" path data) | Serializes data to disk               |
| Stream | (io "lines" path)     | A lazy iterator over lines            |

**Working with Sharded Data**

Datasets are often split across multiple files for storage or distribution. In Sheaf, a _ShardedHandle_ presents multiple files as a single logical sequence.

Whether data is stored in one file or thousands, the access pattern remains the same.

```clojure
;; Treating multiple binary shards as a single sequence of tokens
(let [dataset (io "load" "tokens/shard-*.bin" :i32)]
  (print "Total tokens:" (len dataset))
  (get dataset (range 0 1024))) ; Reads only the required slices
```

Sharding, file boundaries, and memory mapping are handled by the runtime. From the program’s point of view, the dataset is just another collection.

**Summary**

The `io` form isolates side effects while still allowing efficient access to large, persistent datasets. This keeps most Sheaf code focused on data transformation, while I/O remains explicit and localized.

### Some quick exercises

**Exercise 1**:

- What is the result of `(1 + 2) * 3` ?
- Create an array of integers ranging from 1 to 5
- Get its mean value
- Get its shape

_Hint: The Sheaf REPL has autocompletion. Press [Tab] twice to see all the commands._

**Exercise 2**:

- Write a `square` function using the form `defn`, and call it with various numbers
- Write a lambda that returns its argument cubed (^3), using `fn`
- How could we make this work on an array instead of a scalar?

**Exercise 3**:

- We need to perform the following operations on input `x`: add 5, multiply by 2, square
- Implement it with a function that uses the threading operator `as->` to lower the amount of nested parentheses
- Write a function that sums all elements of an array, without using `sum`.

_Hint: To see the documentation and signature of a form, use `:help <form>`._

### Bracket reference

| Syntax   | Name            | Usage                                                                                      |
| -------- | --------------- | ------------------------------------------------------------------------------------------ |
| `(...)`  | Parentheses     | Function calls: `(+ 1 2)`, `(defn f [x] ...)`                                              |
| `[...]`  | Brackets        | Tensors: `[1 2 3]`, parameter lists: `(defn f [x y] ...)`, let bindings: `(let [x 1] ...)` |
| `'[...]` | Quoted brackets | Literal lists (shapes, paths): `'[3 4]`, `'[:a :b]`                                        |
| `{...}`  | Braces          | Dictionaries: `{:x 1 :y 2}`                                                                |

### Glossary

Sheaf, like Clojure and other Lisp dialects, uses its own terminology.

- **Application**: A function call (applying arguments to a function).
- **Binding**: The association of a value to a symbol within a specific scope (_assigning a variable_ in imperative talk).
- **Expression**: Any piece of code that, when evaluated, produces a value.
- **Form**: Any syntactically valid piece of code (a symbol, a literal, or a list).
- **Lambda** (`fn`): An anonymous function.
- **S-Expression**: The core of Lisp: a symbolic expression inside parentheses, and also the tree structure that represents both code and data.
- **Symbol**: A name (like `x` or `my-func`) that refers to a value or a function.
