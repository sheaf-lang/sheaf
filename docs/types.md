# Sheaf Data Types

Sheaf is **tensor-first**: all numeric values are tensors by default, with `f32` (32-bit float) as the default dtype.

## Summary Table

| Type           | Notation       | Example         | Description                      |
| -------------- | -------------- | --------------- | -------------------------------- |
| Tensor         | `[...]`        | `[1 2 3]`       | N-dimensional array, default f32 |
| Tensor (typed) | `[...] :dtype` | `[1 2] :bf16`   | Explicit dtype                   |
| Literal List   | `'[...]`       | `'[3 4]`        | List of static data              |
| Dictionary     | `{...}`        | `{:x 1}`        | Key-value pairs (pytree node)    |
| Keyword        | `:name`        | `:weight`       | Dictionary keys, options         |
| Boolean        | Literal        | `true`, `false` | Boolean values                   |
| String         | `"..."`        | `"hello"`       | Text strings                     |

---

## Tensors

Tensors are the primary data type in Sheaf. All numeric literals and vectors become tensors.

!!! note "**Note: Syntactic Context Matters**"

    Brackets `[]` become tensors in _expression_ context. In _syntactic_ positions, they serve other purposes:

    - `(defn f [x y] ...)` — parameter list, not a tensor

    - `(let [[a b] val] ...)` — destructuring pattern, not a tensor

    - `'[3 4]` — quoted shape literal, not a tensor

### Basic Tensors

```clojure
42                    ; Scalar value (f32)
3.14                  ; Scalar value (f32)
[1 2 3]               ; 1D tensor, shape (3,)
[[1 2] [3 4]]         ; 2D tensor, shape (2, 2)
```

### Dtype Specification

A dtype keyword may follow a vector literal to specify its precision:

```clojure
[1 2 3]               ; Default f32
[1 2 3] :f32          ; Explicit f32
[1 2 3] :bf16         ; BFloat16
[1 2 3] :i32          ; 32-bit integer
[1 2 3] :bool         ; Boolean
```

### Literal Lists

The quote `'` prevents evaluation, treating the following form as literal data rather than code or a numerical tensor. It is typically used to pass structural metadata, such as shapes for tensor creation functions:

```clojure
;; Without the quote, [2 3] is evaluated as a tensor
(nth [2 3] 0)  ; => tensor f32[] = 2.0

;; With the quote, '[2 3] is passed as a literal list
(nth '[2 3] 0) ; => 2

;; Passing shapes is a typical use for quoted vectors
(random-normal (random-key 0) '[3 4])   ; 3x4 random matrix from N(0,1)
(zeros '[2 3])                          ; 2x3 tensor of zeros
(ones '[10])                            ; 1D tensor of ones, length 10
(reshape (arange 6) '[2 3])             ; Reshape to 2x3
```

---

### List to Tensor Conversion

The `tensor` function converts a quoted list or dynamically-generated list into a tensor:

```clojure
;; From quoted list
(tensor '[1 2 3 4 5 6])                    ; => tensor i32[6] = [1 2 3 4 5 6]

;; From dynamically-generated list
(let [lst (cons 1 '[2 3])]                  ; cons returns a list
  (tensor lst))                             ; Convert to tensor
; => tensor i32[3] = [1 2 3]

;; Chaining with tensor operations
(reshape (tensor '[1 2 3 4 5 6]) '[2 3])   ; => tensor i32[2x3] = [[1 2 3] [4 5 6]]
```

### Type Conversion

Sheaf does not perform implicit type conversions. All dtype changes must be explicit.

`cast` converts a tensor to a different dtype. Shape is preserved.

```clojure
;; Direct annotation on vector literals
[1 2 3] :bf16                ; bf16 tensor

;; cast converts an existing tensor
(let [x [1 2 3] :bf16]
  (cast x :f32))             ; bf16 -> f32
```

`int` and `float` convert a scalar to a 32-bit integer or float respectively.

---

## Dictionaries

Key-value structures using curly braces. Keys are typically keywords.

```clojure
{}                    ; Empty dictionary
{:x 1 :y 2}           ; Simple dictionary
{:layer1 {:W [[1 2] [3 4]] :b [0.1 0.2]}}  ; Nested (pytree)
```

### Dictionary Operations

```clojure
(get {:x 1 :y 2} :x)              ; => 1
(assoc {:x 1} :y 2)               ; => {:x 1 :y 2}
(dissoc {:x 1 :y 2} [:x])         ; => {:y 2} (pass keys as a list)
(dissoc {:x 1 :y 2} [:x :y])      ; => {} (remove multiple keys)
(keys {:x 1 :y 2})                ; => '[:x :y] (literal list)
(vals {:x 1 :y 2})                ; => '[1 2] (literal list)
(merge {:x 1} {:y 2})             ; => {:x 1 :y 2}
```

Dictionaries are pytree nodes: Sheaf can differentiate through them and apply transformations like `vmap` to their contents.

---

## Keywords

Identifiers prefixed with `:`. Self-evaluating and used as dictionary keys.

```clojure
:x                    ; Keyword
:learning-rate        ; Keyword with hyphen
(get params :layer1)  ; Access dictionary value
```

---

## Booleans

```clojure
true                  ; Boolean true
false                 ; Boolean false
```

Comparison operations return boolean tensors:

```clojure
(> [1 2 3] 2)         ; => [false false true]
(== [1 2 1] 1)        ; => [true false true]  (element-wise)
(= [1 2 1] 1)         ; => false              (structural equality)
```

---

## Strings

```clojure
"hello"               ; String literal
```

String operations are available through builtin functions. Strings are not tensors and cannot be JIT-compiled.

---

## PyTrees

PyTrees are nested structures of dictionaries and tensors. They are mostly used for neural network parameters:

```clojure
{:layer1 {:W (random-normal (random-key 0) '[4 8]) :b (zeros '[8])}
 :layer2 {:W (random-normal (random-key 1) '[8 1]) :b (zeros '[1])}}
```

PyTrees enable:

- Gradient computation through nested structures via `value-and-grad`
- Batch operations via `vmap` (planned feature)
- Efficient iteration via `scan`

```clojure
(flatten params)           ; Get all tensor leaves
(tree-reduce + params 0.0) ; Sum all values in pytree
```

---

For function signatures and detailed examples, see the [Function Reference](reference.md).
