# Sheaf Language Reference

---

## Arithmetic & Math

### +

**Type:** function  
**Signature:** `(+ a b [c ...])`

Performs element-wise addition supporting standard broadcasting rules. The function is variadic and accepts two or more arguments.

```sheaf
(+ 1 2)              ; => 3
(+ [1 2] [3 4])      ; => [4. 6.]
(+ 1 [2 3] 4)        ; => [7. 8.]
```

---

### -

**Type:** function  
**Signature:** `(- a [b ...])`

Performs unary negation if a single argument is provided. If multiple arguments are provided, performs element-wise subtraction of all subsequent arguments from the first.

```sheaf
(- 5)                ; => -5
(- 10 3)             ; => 7
(- [5 10] [2 3])     ; => [3. 7.]
```

---

### \*

**Type:** function  
**Signature:** `(* a b [c ...])`

Computes element-wise multiplication between inputs with broadcasting support. Variadic for 2+ arguments. For matrix multiplication (dot product), see @.

```sheaf
(* 2 3)              ; => 6
(* [1 2] 3)          ; => [3. 6.]
(* [2 3] [4 5])      ; => [ 8. 15.]
```

---

### /

**Type:** function  
**Signature:** `(/ a b)`

Computes element-wise division. Inputs are automatically promoted to floating-point types to maintain precision.

```sheaf
(/ 10.0 2.0)         ; => 5.0
(/ [10 20] 2)        ; => [ 5. 10.]
```

---

### //

**Type:** function  
**Signature:** `(// a b)`

Computes element-wise integer division, rounding the result toward negative infinity (floor division). Supporting broadcasting, it returns the largest integer less than or equal to the algebraic quotient.

```sheaf
(// 7 2)             ; => 3
(// [7 9] 2)         ; => [3. 4.]
```

---

### mod

**Type:** function  
**Signature:** `(mod a b)` or `(% a b)`

Computes the element-wise modulo operation (remainder after division). The result has the same sign as the divisor. The `%` operator can also be used as an alias.

```sheaf
(mod 7 3)            ; => 1
(% 7 3)              ; => 1 (same, using % alias)
(mod [7 9] 2)        ; => [1. 1.]
(mod -7 3)           ; => 2 (same sign as divisor)
```

---

### \*\*

**Type:** function  
**Signature:** `(** base exponent)`

Computes element-wise exponentiation, raising the first input to the power of the second. Supports broadcasting between base and exponent tensors.

```sheaf
(** 2.0 3.0)         ; => 8.0
(** [2 3] 2)         ; => [4. 9.]
```

---

### abs

**Type:** function  
**Signature:** `(abs x)`

Computes the element-wise absolute value of the input. For complex inputs, this returns the magnitude.

```sheaf
(abs -5.0)           ; => 5.0
(abs [-3 2 -1])      ; => [3. 2. 1.]
```

---

### ash

**Type:** function
**Signature:** `(ash value shift)`

Arithmetic shift. Positive shift means left shift (multiply by 2^n), negative shift means right shift (divide by 2^n, floor toward negative infinity). Matches Common Lisp `ash` semantics. Works element-wise on tensors.

```sheaf
(ash 1 8)            ; => 256
(ash 256 -3)         ; => 32
(ash [512 1024] -8)  ; => [2 4]
```

---

### exp

**Type:** function  
**Signature:** `(exp x)`

Computes the element-wise exponential of the input (e^x). It is a fundamental building block for activation functions like Softmax and Sigmoid.

```sheaf
(exp 0.0)            ; => 1.0
(exp 1.0)            ; => 2.7182817
```

---

### log

**Type:** function  
**Signature:** `(log x)`

Computes the element-wise natural logarithm (base e). The function is defined for positive real inputs; values outside this domain may result in `NaN`.

```sheaf
(log 1.0)            ; => 0.0
(log 2.718281828)    ; => 1.0
(log [1.0 2.0])      ; => [0.        0.6931472]
```

---

### sqrt

**Type:** function  
**Signature:** `(sqrt x)`

Computes the element-wise non-negative square root of the input. For real-valued tensors, results are undefined for negative inputs.

```sheaf
(sqrt 4.0)           ; => 2.0
(sqrt [1.0 4.0 9.0]) ; => [1. 2. 3.]
```

---

### @

**Type:** function  
**Signature:** `(@ a b)`

Performs matrix multiplication, dot product, or tensor contraction depending on input dimensionality. It follows standard linear algebra rules for inner products and matrix-matrix or matrix-vector operations.

```sheaf
;; Vector dot product
(@ [1 2 3] [4 5 6])  ; => 32.0

;; Matrix multiply: 2x3 @ 3x2 => 2x2
(@ [[1 2 3] [4 5 6]] [[1 2] [3 4] [5 6]]) ; => [[22. 28.]  [49. 64.]]
```

---

## Comparison

### =

**Type:** function  
**Signature:** `(= a b)`

Performs structural equality comparison. Returns a single scalar boolean indicating whether the two inputs have the same shape and identical values across all elements.

```sheaf
(= 1 1)              ; => True
(= [1 2] [1 2])      ; => True
(= [1 2] [1 3])      ; => False
```

---

### ==

**Type:** function  
**Signature:** `(== a b)`

Performs element-wise equality comparison with broadcasting support. Returns a boolean tensor of the same shape as the broadcasted inputs where each element represents the result of the local equality check.

```sheaf
(== [1 2] [1 3])     ; => [ true false]
(== 1 [1 1 2])       ; => [ true  true false]
```

---

### !=

**Type:** function  
**Signature:** `(!= a b)`

Performs element-wise inequality comparison with broadcasting support. Returns a boolean tensor where each entry is true if the corresponding elements are not equal.

```sheaf
(!= [1 2] [1 3])     ; => [False  True]
(!= 1 2)             ; => true
```

---

### <, >, <=, >=

**Type:** function  
**Signature:** `(< a b)`, `(> a b)`, etc.

Perform element-wise relational comparisons using standard broadcasting rules. These functions return boolean tensors suitable for masking operations such as `where` or `select`.

```sheaf
(< [1 5] 3)          ; => [ true false]
(> 5 2)              ; => true
(<= [1 2 3] 2)       ; => [ true  true false]
```

---

## Logic

### and

**Type:** function  
**Signature:** `(and expr1 expr2 [...])`

Evaluates expressions from left to right. It short-circuits and returns `false` (or `nil`) at the first falsy value encountered. If all expressions evaluate to truthy, it returns the value of the last expression.

```sheaf
(and true false)     ; => false
(and true (> 2 1))   ; => true
(and 1 2 3)          ; => 3
```

---

### or

**Type:** function  
**Signature:** `(or expr1 expr2 [...])`

Evaluates expressions from left to right. It short-circuits and returns the first truthy value encountered. If all expressions are falsy, it returns the value of the last expression.

```sheaf
(or false true)      ; => true
(or false false nil) ; => nil
(or nil 42)          ; => 42
```

---

### not

**Type:** function  
**Signature:** `(not x)`

Computes the logical negation of the input. Returns `true` if the argument is `false` or `nil`, and `false` for any other truthy value.

```sheaf
(not true)           ; => False
(not false)          ; => True
```

---

### true, false, nil

**Type:** literals

Represent the fundamental constants for logical operations. In Sheaf, only `false` and `nil` are considered falsy; all other values (including `0`, empty strings, or empty tensors) are treated as truthy in logical contexts.

```sheaf
true                 ; Boolean true
false                ; Boolean false
nil                  ; Null/None value
```

---

## Activation Functions

### relu

**Type:** function  
**Signature:** `(relu x)`

Computes the Rectified Linear Unit activation function. It returns the element-wise maximum of `0` and the input `x`. This function is computationally efficient and helps mitigate the vanishing gradient problem.

```sheaf
(relu -1.0)          ; => 0.0
(relu [1.0 -2.0 3.0]) ; => [1. 0. 3.]
```

---

### leaky-relu

**Type:** function
**Signature:** `(leaky-relu x slope)`

Computes the Leaky Rectified Linear Unit activation function. Unlike standard ReLU, it allows a small non-zero gradient when the input is negative, defined as `slope * x`. This prevents "dead" neurons during training.

```sheaf
(leaky-relu -1.0 0.01)          ; => -0.01
(leaky-relu [-1.0 2.0] 0.1)     ; => [-0.1  2. ]
```

---

### sigmoid

**Type:** function  
**Signature:** `(sigmoid x)`

Applies the element-wise logistic sigmoid function. It maps any real-valued input into a range between `0` and `1`. Often used for binary classification or as a gating mechanism in recurrent architectures.

```sheaf
(sigmoid 0.0)        ; => 0.5
(sigmoid -100.0)     ; => 0.0
```

---

### tanh

**Type:** function  
**Signature:** `(tanh x)`

Computes the element-wise hyperbolic tangent. It maps inputs to the range `(-1, 1), providing a zero-centered output which often leads to faster convergence than the sigmoid function in hidden layers.

```sheaf
(tanh 0.0)           ; => 0.0
(tanh 1.0)           ; => 0.7615942
```

---

### gelu

**Type:** function  
**Signature:** `(gelu x)`

Computes the Gaussian Error Linear Unit activation. It weights inputs by their percentile according to a normal distribution. Standard in modern Transformer architectures (like BERT and GPT) for its smooth non-linearity.

```sheaf
(gelu 0.0)           ; => 0.0
(gelu 1.0)           ; => 0.841192
```

---

### selu

**Type:** function  
**Signature:** `(selu x)`

Applies the Scaled Exponential Linear Unit. When used with correct initialization, SELU enables self-normalizing neural networks, where the mean and variance of activations are naturally preserved across layers.

```sheaf
(selu [-1.0 1.0])    ; => [-1.1113307  1.050701 ]
```

---

### celu

**Type:** function  
**Signature:** `(celu x [:alpha a])`

Computes the Continuously Differentiable Exponential Linear Unit. Similar to ELU but ensures the derivative is continuous at `x=0`, facilitating smoother optimization.

```sheaf
(celu [-1.0 1.0])    ; => [-0.63212055  1.        ]
```

---

### silu

**Type:** function  
**Signature:** `(silu x)`

Computes the Sigmoid Linear Unit (also known as Swish). Defined as `x * sigmoid(x)`, it is a smooth, non-monotonic function that frequently outperforms ReLU in deep convolutional and transformer models.

```sheaf
(silu 0.0)           ; => 0.0
(silu 1.0)           ; => 0.7310586
```

---

### softmax

**Type:** function  
**Signature:** `(softmax x [:axis axis])`

Applies the Softmax function along the specified axis. It rescales elements such that they fall in the range `[0, 1]` and sum to `1`, effectively representing a categorical probability distribution.

```sheaf
(softmax [1.0 1.0])  ; => [0.5 0.5]
(softmax [1.0 2.0 3.0] :axis 0) ; => [0.09003057 0.24472848 0.66524094]
```

---

### log-softmax

**Type:** function  
**Signature:** `(log-softmax x [:axis axis])`

Computes the natural logarithm of the Softmax function. This implementation is numerically stable, preventing overflow/underflow issues that occur when computing `log` and `softmax` separately. Essential for training with cross-entropy loss.

```sheaf
(log-softmax [1.0 2.0]) ; => [-1.3132616  -0.31326163]
```

---

## Reductions

### sum

**Type:** function  
**Signature:** `(sum x [:axis axis :keepdims])`

Computes the sum of all elements in the input tensor. If an axis is specified, the reduction is performed along that dimension, collapsing it in the output.

```sheaf
(sum [1 2 3])        ; => 6.0
(sum [[1 2] [3 4]] :axis 0) ; => [4. 6.]
```

---

### mean

**Type:** function  
**Signature:** `(mean x [:axis axis :keepdims])`

Computes the arithmetic mean of tensor elements. When an axis is provided, it calculates the average along that specific dimension.

```sheaf
(mean [1.0 2.0 3.0]) ; => 2.0
(mean [[1 2] [3 4]] :axis 1) ; => [1.5 3.5]
```

---

### product

**Type:** function  
**Signature:** `(product x [:axis axis :keepdims])`

Computes the product of all elements in the input. Like other reduction functions, it supports axis-specific operations to multiply elements along a given dimension.

```sheaf
(product [1 2 3])    ; => 6.0
(product [2.0 3.0 4.0] :axis 0) ; => 24.0
```

---

### min, max

**Type:** function  
**Signature:** `(min x [:axis axis :keepdims])`, `(max x [:axis axis :keepdims])`

Return the single smallest or largest scalar value present in the entire input tensor. These functions perform a total reduction across all dimensions.

```sheaf
(min [3 1 4])        ; => 1.0
(max [1.0 2.0 10.0]) ; => 10.0
```

---

### minimum, maximum

**Type:** function  
**Signature:** `(minimum a b)`, `(maximum a b)`

Perform element-wise comparison between two tensors and return a new tensor containing the minimum or maximum value at each position. These functions support broadcasting if the input shapes differ.

```sheaf
(minimum [1 10] [5 2]) ; => [1. 2.]
(maximum [1 10] [5 2]) ; => [ 5. 10.]
```

---

### argmax, argmin

**Type:** function  
**Signature:** `(argmax x [:axis axis])`, `(argmin x [:axis axis])`

Return the index of the largest or smallest element in a tensor. Without `:axis`, the result is a single scalar index into the flattened array. With `:axis`, the reduction is performed along that axis, returning a tensor of indices one rank lower than the input.

```sheaf
(argmax [3 1 4 1 5])                ; => 4
(argmin [3 1 4 1 5])                ; => 1

;; Along a specific axis
(argmax [[1 5] [3 2]] :axis 1)      ; => [1 0]
(argmin [[1 5] [3 2]] :axis 0)      ; => [0 1]
```

---

## Tensor Shape & Structure

### shape

**Type:** function  
**Signature:** `(shape x [axis])`

Returns tensor dimensions as a tuple. With axis, returns size of that axis.

```sheaf
;; 3D tensor: 2 matrices of 3x2
(shape [[[1 2]
         [3 4]
         [5 6]]
        [[7 8]
         [9 10]
         [11 12]]])      ; => (2 3 2)

;; Size of axis 1 (middle dimension)
(shape [[[1 2]
         [3 4]
         [5 6]]
        [[7 8]
         [9 10]
         [11 12]]] 1)    ; => 3
```

---

### ndim

**Type:** function  
**Signature:** `(ndim x)`

Number of tensor dimensions.

```sheaf
;; 3D tensor (shape 2x3x2)
(ndim [[[1 2]
        [3 4]
        [5 6]]
       [[7 8]
        [9 10]
        [11 12]]])       ; => 3

;; 1D tensor (shape 3)
(ndim [1 2 3])           ; => 1
```

---

### len, count

**Type:** function  
**Signature:** `(len x)`, `(count x)`

Return the length of the first dimension for arrays/tensors, or number of elements in a sequence.

```sheaf
(len [1 2 3])              ; => 3
(count '[1 2 3 4])         ; => 4
(count [[1 2] [3 4]])      ; => 2
```

---

### reshape

**Type:** function  
**Signature:** `(reshape x new-shape)`

Gives a new shape to a tensor without changing its underlying data. The total number of elements must remain constant. A dimension value of `-1` may be used to let the compiler infer the correct size based on the remaining dimensions.

```sheaf
;; Explicit reshape
(reshape (arange 6) '[2 3]) ; => [[0 1 2]
 [3 4 5]]

;; Infer dimension with -1
(reshape (arange 9) '[3 -1]) ; => [[0 1 2] [3 4 5] [6 7 8]]
```

---

### transpose

**Type:** function
**Signature:** `(transpose x [perm])`
**Alias:** `tr`

Permutes the dimensions of a tensor according to the sequence of axes provided. This is a zero-copy operation in XLA that is critical for reordering dimensions (e.g., swapping batch and sequence axes or head dimensions).

```sheaf
;; Default transpose (reverse axes): (2 3 2) -> (2 3 2)
(shape (transpose [[[1 2]
                    [3 4]
                    [5 6]]
                   [[7 8]
                    [9 10]
                    [11 12]]])) ; => (2 3 2)

;; Custom permutation '[2 0 1]: (2 3 2) -> (2 2 3)
(shape (tr [[[1 2]
             [3 4]
             [5 6]]
             [[7 8]
             [9 10]
             [11 12]]] '[2 0 1])) ; => (2 2 3)
```

---

### swapaxes

**Type:** function  
**Signature:** `(swapaxes x axis1 axis2)`

Interchanges two specified axes of a tensor. This is a specialized version of transpose used to reorder dimensions, commonly used to switch between "batch-first" and "sequence-first" formats.

```sheaf
(shape (swapaxes [[1 2 3] [4 5 6]] 0 1)) ; => (3, 2)
```

---

### slice

**Type:** function  
**Signature:** `(slice x start end)`

Extracts a continuous sub-section of a tensor along its first axis, starting from the `start` index (inclusive) and stopping at the `end` index (exclusive).

```sheaf
(slice [0 1 2 3 4] 1 4)    ; => [1. 2. 3.]
```

---

### roll

**Type:** function  
**Signature:** `(roll x shift [:axis axis])`

Rolls tensor elements along the specified axis. Elements that roll off one end are reintroduced at the other. If no axis is provided, the tensor is flattened, rolled, and restored to its original shape.

```sheaf
(roll [1 2 3] 1)           ; => [3. 1. 2.]
(roll [1 2 3] -1)          ; => [2. 3. 1.]
```

---

### concat

**Type:** function  
**Signature:** `(concat x1 x2 [...] [:axis axis])`

Concatenate sequences (lists or arrays). Returns a list for list inputs, tensor for array inputs. Supports `:axis` for array concatenation (default 0).

```sheaf
;; Concatenate lists
(concat '[1 2] '[3 4])          ; => [1, 2, 3, 4]

;; Concatenate tensor (axis=0, default)
(concat [1 2] [3 4])          ;  => Tensor f32[4] [1. 2. 3. 4.]

;; Concatenate tensors along specific axis
(concat [[1 2]] [[3 4]] :axis 0) ; => Tensor f32[4] [ [1. 2.] [ 3. 4.]]
(concat [[1] [2]] [[3] [4]] :axis 1) ; => Tensor f32[4] [ [1. 3.] [ 2. 4.]]
```

---

### tril

**Type:** function  
**Signature:** `(tril x)`

Returns the lower triangular part of a matrix or a stack of matrices. Elements above the main diagonal are set to zero.

```sheaf
(tril [[1 2] [3 4]]) ; => [[1. 0.]
 [3. 4.]]
```

---

### where

**Type:** function  
**Signature:** `(where condition x y)`

Performs element-wise selection between `x` and `y` based on a boolean condition. If the condition is true, it selects the element from `x`, otherwise from `y`. This is the functionally pure equivalent of an if-else statement for tensors and is compatible with JIT compilation.

```sheaf
(where (> [1 3 2] 2) [10 20 30] 0) ; => [ 0. 20.  0.]
```

---

## Tensor Creation

### ones, zeros

**Type:** function  
**Signature:** `(ones shape)`, `(zeros shape)`

Fill tensor with 1.0 or 0.0. Shape must be a tuple/list (use quote for vector literals).

```sheaf
(ones '[2 3])        ; => [[1. 1. 1.]
 [1. 1. 1.]]
(zeros '[3])         ; => [0. 0. 0.]
```

---

### eye

**Type:** function  
**Signature:** `(eye n [m])`

Creates a 2D identity matrix of shape `(n, n)`. When a second argument `m` is provided, the output shape is `(n, m)` — a rectangular matrix with ones on the main diagonal and zeros elsewhere.

```sheaf
(eye 3)                ; => [[1. 0. 0.]
                        ;     [0. 1. 0.]
                        ;     [0. 0. 1.]]

(eye 2 4)              ; => [[1. 0. 0. 0.]
                        ;     [0. 1. 0. 0.]]
```

---

### arange

**Type:** function  
**Signature:** `(arange [start] stop [step])`

Generates a 1D tensor containing a sequence of evenly spaced values within a given interval. The sequence starts at `start` (default 0), increments by `step` (default 1), and ends before reaching `stop`.

```sheaf
(arange 5)           ; => [0 1 2 3 4]
(arange 2 7)         ; => [2 3 4 5 6]
(arange 0 10 2)      ; => [0 2 4 6 8]
```

---

### one-hot

**Type:** function  
**Signature:** `(one-hot indices num-classes)`

Converts integer indices into a one-hot representation. For an input of shape `(...)`, the output will have shape `(..., num-classes)`, where the specified indices are set to `1` and all other entries to `0`.

```sheaf
(one-hot 1 3)        ; => [0. 1. 0.]
(one-hot [0 2 1] 3)

; => [[1. 0. 0.]
; [0. 0. 1.]
; [0. 1. 0.]]
```

---

### index-update

**Type:** function  
**Signature:** `(index-update tensor idx value)`

Returns a new tensor identical to `tensor` except that the slice at position `idx` is replaced by `value`. The index must be a scalar integer — not a single-element vector. This is the standard way to perform functional (out-of-place) updates on tensors.

```sheaf
;; Replace one element in a 1D vector
(index-update [1 2 3 4 5] 2 99)     ; => [ 1.  2. 99.  4.  5.]

;; Replace one row in a 2D matrix
(index-update [[1 2] [3 4]] 0 [10 20])  ; => [[10. 20.]
                                         ;     [ 3.  4.]]
```

---

### tensor

**Type:** function  
**Signature:** `(tensor data)`

Converts a literal list or a dynamically-generated list into a tensor. This function is required when a sequence is built at runtime (via `cons`, `append`, etc.) and needs to enter the computational pipeline.

```sheaf
;; Conversion from a literal list
(tensor '[1 2 3])             ; => Tensor i32[3] = [1 2 3]
(tensor '[1. 2. 3.])          ; => Tensor f32[3] = [1. 2. 3.]

;; Conversion from a dynamic list
(let [lst (cons 1 '[2 3])]    ; 'cons' returns a list structure
  (tensor lst))               ; Explicitly lift to tensor context
; => Tensor i32[3] = [1 2 3]
```

---

### int, float

**Type:** function  
**Signature:** `(int x)`, `(float x)`

Cast a scalar tensor or number to a 32-bit integer or 32-bit float, respectively. The result is a scalar tensor. These are the primary way to extract a concrete numeric value from a tensor for use in indexing or list operations.

```sheaf
(int 3.7)                ; => 3
(int [1. 2. 3.] :f32)    ; => i32[3] = [1 2 3]

(float 42)               ; => 42.0
(float [1 2 3] :i32)     ; => f32[3] = [1. 2. 3.]
```

---

### cast

**Type:** function
**Signature:** `(cast expr :dtype)`

Converts a tensor, scalar, or DeviceBuffer to the specified dtype. Supported dtypes: `:f32`, `:bf16`, `:i32`. Sheaf never converts dtypes implicitly.

The `[vec] :dtype` syntax is syntactic sugar that desugars to `cast` at parse time.

```sheaf
(cast [1 2 3] :bf16)         ; => tensor bf16[3] = [1. 2. 3.]
(cast [1 2 3] :f32)          ; => tensor f32[3] = [1. 2. 3.]

;; Equivalent shorthand via type annotation
[1 2 3] :bf16                ; => tensor bf16[3] = [1. 2. 3.]

;; Cast an existing tensor
(def x [1 2 3])
(cast x :bf16)               ; => tensor bf16[3] = [1. 2. 3.]

;; Cast scalars
(cast 3.14 :bf16)            ; => tensor bf16[] = 3.14
```

---

## Initializers

Provided by `(use nn)`. All follow the `(init key shape)` convention, matching PyTorch naming.
Use a quoted vector for the shape (e.g., `'[256 128]`) to ensure it is treated as static data.

For zero/one initialization, use the builtins `zeros` and `ones` directly (no key needed).

### xavier-uniform, xavier-normal

**Type:** function (`nn` module)
**Signature:** `(xavier-uniform key shape)`, `(xavier-normal key shape)`

Xavier (Glorot) initialization. Scales weights so variance stays constant across layers. Effective with symmetric activations (`tanh`, `sigmoid`).

```sheaf
(use nn)
(xavier-uniform (random-key 42) '[256 128])  ; U(-limit, limit), limit = sqrt(6/(fan_in+fan_out))
(xavier-normal  (random-key 42) '[256 128])  ; N(0, sqrt(2/(fan_in+fan_out)))
```

---

### kaiming-uniform, kaiming-normal

**Type:** function (`nn` module)
**Signature:** `(kaiming-uniform key shape)`, `(kaiming-normal key shape)`

Kaiming (He) initialization. Accounts for ReLU non-linearity by scaling variance by `2/fan_in`.

```sheaf
(kaiming-normal (random-key 42) '[512 256])
```

---

### lecun-normal, lecun-uniform

**Type:** function (`nn` module)
**Signature:** `(lecun-normal key shape)`, `(lecun-uniform key shape)`

LeCun initialization. Scales by `1/fan_in`. Default for SELU networks.

```sheaf
(lecun-normal (random-key 42) '[256 128])
```

---

## Random Sampling

### random-key

**Type:** function  
**Signature:** `(random-key seed)`

Generates a Pseudo-Random Number Generator (PRNG) key starting from an integer seed. Unlike many other frameworks, randomness in Sheaf is explicit and deterministic: the state is never hidden, ensuring identical results across different runs and hardwares.

```sheaf
(random-key 42)      ; => key<fry>[] = Array((), overlaying: [ 0 42]
```

---

### random-split

**Type:** function  
**Signature:** `(random-split key [num])`

Splits a single PRNG key into a specified number `num` of independent sub-keys. This is a cornerstone of stochastic modeling in Sheaf: to maintain statistical independence, a key should be used for exactly one operation and then "discarded" in favor of its split children.

```sheaf
(let [key (random-key 0)
      [k1 k2 k3] (random-split key 3)]
  (dict :w1 (random-normal k1 '[10])
        :w2 (random-normal k2 '[10])
        :noise (random-uniform k3 '[10])))

; =>  {:w1 f32[10], :w2 f32[10], :noise f32[10]}
```

---

### random-normal

**Type:** function  
**Signature:** `(random-normal key shape [:dtype dtype])`

Samples random values from a standard normal distribution (mean 0, variance 1) with the specified shape.

```sheaf
(random-normal (random-key 0) '[100 100])  ; => Tensor i32[100x100]
```

---

### random-uniform

**Type:** function  
**Signature:** `(random-uniform key [shape])`

Samples random values from a uniform distribution over the semi-open interval `[0.0, 1.0)`. This is often used as a base for custom probability transforms or simple noise injection.

```sheaf
(random-uniform (random-key 0) '[10 10])  ; => Tensor i32[10x10]
```

---

### random-randint

**Type:** function  
**Signature:** `(random-randint key shape low high)`

Generates random integers sampled uniformly from the range `[low, high)`. This is useful for generating random indices, selecting data augmentations, or creating dummy datasets.

```sheaf
(random-randint (random-key 0) '[100] 0 10)  ; => ⇒ Tensor i32[100]
```

---

### choice

**Type:** function  
**Signature:** `(choice key a shape [:p p :replace replace])`

Generates a random sample from a given 1D array or a range.

- `a`: If an integer, the sample is taken from `(arange a)`. If a tensor, the sample is taken from its elements.

- `shape`: The shape of the output sample.

- `p`: An optional tensor of probabilities associated with each entry in `a`.

- `replace`: Whether the sample is with or without replacement.

```sheaf

;; Samples 5 random integers from the range [0, 100).
(choice (random-key 0) 100 '[5])  ; => [89  0 12 73 71]

;; Samples 10 indices from the range [0, 3) according to the
;; provided probability distribution.
(choice (random-key 0) 3 '[10] :p [0.1 0.3 0.6])  ; => [0 0 2 2 2 2 2 1 1 2]
```

---

## List Operations

### first, second, last

**Type:** function  
**Signature:** `(first seq)`, `(second seq)`, `(last seq)`

Provides positional access to elements within a list.
To access elements in a tensor, use `nth` or `slice`.

```sheaf
(first '[1 2 3])      ; => 1
(second '[1 2 3])     ; => 2
(last '[1 2 3])       ; => 3
```

---

### rest

**Type:** function  
**Signature:** `(rest seq)`

Returns a new sequence containing all elements of the input except for the first one. In Lisp terms, this is the `cdr` operation. If the sequence is empty, it returns an empty list.

```sheaf
(rest '[1 2 3])      ; => [2, 3]
```

---

### chars

**Type:** function  
**Signature:** `(chars s)`

Splits a string into a list of its individual characters. This is the standard way to iterate over a string character by character, e.g. to build a vocabulary or encode text.

```sheaf
(chars "hello")      ; => ["h" "e" "l" "l" "o"]
(count (chars "abc")) ; => 3
```

---

### nth

**Type:** function  
**Signature:** `(nth seq index)`

Retrieves the element at the specified zero-based `index`. This function is polymorphic in Sheaf: it performs a pointer-based lookup on lists and a coordinate-based extraction on tensors.

```sheaf
(nth '[10 20 30] 1)  ; => 20 (Sheaf list)
(nth [10 20 30] 1)   ; => f32[] 20.0 (Tensor)
```

---

### cons

**Type:** function  
**Signature:** `(cons elem seq)`

Constructs a new list by prepending `elem` to the front of `seq`. This is a constant-time O(1) operation for lists, making it the preferred way to build sequences dynamically.

```sheaf
(cons 0 '[1 2 3])    ; => [0, 1, 2, 3]
```

---

### append

**Type:** function  
**Signature:** `(append seq elem)`

Creates a new sequence by adding `elem` to the end of `seq`. Unlike cons, this operation requires traversing the entire sequence, resulting in O(N) complexity for lists.

```sheaf
(append '[1 2] 3)    ; => [1, 2, 3]

(append [1 2] 3)     ; => Tensor f32[3] = [1. 2. 3.]
```

---

### empty?

**Type:** function  
**Signature:** `(empty? seq)`

Returns `true` if the provided sequence (list or tensor) contains no elements. For tensors, this checks if any dimension in the shape is `0`.

```sheaf
(empty? '[])         ; => true
(empty? '[1])        ; => false
(empty? [1 2 3])     ; => false (tensor)
```

---

### sort

**Type:** function  
**Signature:** `(sort seq [:reverse] [:key f] [:axis n])`

Sorts a list or tensor. Polymorphic: available options depend on the input type.

| Option     | List | Tensor | Description                                                                  |
| ---------- | ---- | ------ | ---------------------------------------------------------------------------- |
| `:reverse` | yes  | yes    | Sort in descending order instead of ascending.                               |
| `:key`     | yes  | no     | A function applied to each element; elements are sorted by the return value. |
| `:axis`    | no   | yes    | The tensor axis along which to sort (default: last axis).                    |

```sheaf
;; Lists — alphabetical
(sort '("c" "a" "b"))                                  ; => ["a" "b" "c"]
(sort '("c" "a" "b") :reverse)                         ; => ["c" "b" "a"]

;; Lists — sort by key function (here: string length)
(sort '("hello" "hi" "hey" "a") :key count)            ; => ["a" "hi" "hey" "hello"]
(sort '("hello" "hi" "hey" "a") :key count :reverse)   ; => ["hello" "hey" "hi" "a"]

;; Tensors — sort along last axis (default)
(sort [3.0 1.0 2.0])                                   ; => [1. 2. 3.]
(sort [3.0 1.0 2.0] :reverse)                          ; => [3. 2. 1.]
(sort [[3 1] [2 4]] :axis 1)                           ; => [[1. 3.] [2. 4.]]
```

---

### filter

**Type:** function  
**Signature:** `(filter pred seq)`

Returns a new list containing only the elements of `seq` for which the predicate `pred` returns a truthy value. The result is always a tuple (static list). Works on both quoted lists and lists of strings built at runtime.

```sheaf
(filter (fn [x] (> x 2)) '[1 2 3 4 5])                ; => '[3, 4, 5]
(filter (fn [x] (> x 10)) '[1 2 3])                   ; => '[]

;; Filter a list of strings by length
(filter (fn [s] (> (count s) 3)) ["hi" "hello" "hey" "world"])  ; => ('hello', 'world')
```

---

### find

**Type:** function  
**Signature:** `(find pred seq)`

Returns the first element of `seq` for which `pred` returns a truthy value, or `nil` if no element matches. Traversal is left-to-right and stops at the first match.

```sheaf
(find (fn [x] (> x 3)) '[1 2 3 4 5])                  ; => 4
(find (fn [x] (> x 10)) '[1 2 3])                     ; => nil
```

---

### index-of

**Type:** function  
**Signature:** `(index-of seq val)`

Returns the zero-based index of the first occurrence of `val` in `seq`, or `-1` if `val` is not present. Works on both numbers and strings.

```sheaf
(index-of '[10 20 30 40] 30)                           ; => 2
(index-of '[10 20 30] 99)                              ; => -1

;; Strings — typical use: map a label to its one-hot index
(index-of ["apple" "banana" "cherry"] "banana")         ; => 1
```

---

## Higher-Order Functions

### map

**Type:** function  
**Signature:** `(map func seq)`

Applies a function to each element of a sequence and returns a list of the results. In Sheaf, `map` is the primary way to transform collections without using explicit loops. While `map` is sequential in logic, the underlying execution on tensors can often be optimized by the compiler.

```sheaf
;; Basic numeric transformation
(map (fn [x] (* x 2)) [1 2 3])

; => [2.0 4.0 6.0]

;; Mapping over a list of parameter dictionaries
(map (fn [p] (with-params [p] (+ W b)))
     [{:W 1 :b 0.1} {:W 2 :b 0.2}])

; => [1.1 2.2]
```

---

### tree-map

**Type:** function  
**Signature:** `(tree-map f tree1 [tree2 ...])`

Applies a function `f` to every leaf in a tree (or multiple trees) and returns a new tree with the same structure. This is the primary tool for manipulating model parameters, such as applying an optimizer update or scaling weights, without manually flattening the structures.

```sheaf
;; Square all elements in a nested structure
(let [params {:layer1 {:w [2.0 4.0] :b 0.5}
              :layer2 {:w [10.0]}}]
  (tree-map (fn [x] (* x x)) params))

; => {:layer1 {:w [4.0 16.0], :b 0.25}, :layer2 {:w [100.0]}}
```

---

### tree-map-zeros

**Type:** function  
**Signature:** `(tree-map-zeros tree)`

Creates a new tree with the same structure as the input, but where every leaf is replaced by a zero of the same numerical type. This is primarily used to initialize gradient accumulators or optimizer states.

````sheaf
;; Zero all elements in a nested structure
(let [params {:layer1 {:w [2.0 4.0] :b 0.5}
              :layer2 {:w [10.0]}}]
  (tree-map-zeros params))

;=> {:layer1 {:b 0.0 :w [0.0, 0.0]}, :layer2 {:w [0.0]}}

---

### flatten

**Type:** function
**Signature:** `(flatten tree)`

Flattens a PyTree into a pair of (leaves, tree-structure). The leaves are a flat list of all leaf values in the tree, and tree-structure is metadata that can be used to reconstruct the original tree. Useful for examining all parameter values or applying global operations.

```sheaf
;; Extract all leaves from a nested structure
(let [params {:layer1 {:w 1.0 :b 2.0} :layer2 {:w 3.0}}]
  (first (flatten params)))

; => [2.0 1.0 3.0]

;; Count total leaves
(len (first (flatten {:a 1 :b 2 :c 3}))) ; => 3
```

---

### tree-reduce

**Type:** function
**Signature:** `(tree-reduce func tree [init])`

Reduces all leaves of a PyTree to a single value by applying a binary function cumulatively. If an initial value is provided, it becomes the first accumulator; otherwise the first leaf is used. This is useful for computing aggregate statistics across all parameters, such as the total number of learnable weights or the sum of all gradients.

```sheaf
;; Sum all leaf values
(tree-reduce + {:a 1 :b 2 :c 3} 0) ; => 6

;; Multiply all values (starting from 1)
(tree-reduce * '[2 3 4] 1) ; => 24

;; Find maximum across all leaves
(tree-reduce maximum {:l1 [1.0 5.0] :l2 [2.0]} -1e9) ; => [2. 5.]
```

---

### reduce

**Type:** function
**Signature:** `(reduce func init seq)`

Reduces a collection to a single value by applying a binary function func cumulatively to the elements of seq, from left to right, starting with the init value. Each step takes the current accumulator and the next element to produce the new accumulator.

```sheaf
;; Simple sum: 0 + 1 + 2 + 3 + 4
(reduce + 0 [1 2 3 4]) ; => 10.

;; Finding the maximum value across multiple tensors
;; Useful for tracking the peak activation across different layers
(reduce maximum -1e9 [[1 5] [2 3] [8 4]]) ; => [8. 5.]

````

---

### apply

**Type:** function  
**Signature:** `(apply func seq)`

Calls the provided function func using the elements of seq as its individual arguments. It "unpacks" a list or a tensor so that each element becomes a separate argument to the function.

```sheaf
;; Unpacking a list for a variadic function
;; Equivalent to (+ 1 2 3 4)
(apply + [1 2 3 4])      ; => 10.0

;; Finding the global maximum of a list of scalars
;; Equivalent to (max [10 52 8])
(apply max [10 52 8])  ; => 52.0

;; Dynamic shape generation
;; Passing a list of dimensions to a constructor
;; Equivalent to (ones '[2 3 4])
(let [dims '[2 3 4]]
  (apply ones dims))

; => f32[2x3x4] (μ=1.000 min=1.000 max=1.000)

```

---

## Dictionary Operations

### dict

**Type:** special-form  
**Signature:** `(dict :key1 val1 :key2 val2 ...)`

Creates a dictionary (map) from a sequence of alternating keys and values. This is useful for dynamically constructing parameter maps or configurations where keys and values are computed or passed as arguments.

```sheaf
;; Basic construction
(dict :learning-rate 0.001 :batch-size 32)

; => {:learning-rate 0.001, :batch-size 32}

;; Constructing from variables
(let [w (ones '[10])
      b (zeros '[1])]
  (dict :W w :b b))

; => {:W f32[10], :b f32[1]}



```

---

### get

**Type:** special-form  
**Signature:** `(get coll key [default])`

Retrieves the value associated with a key in a map or an index in a tensor.

If the key/index exists, returns the corresponding value.
If it does not exist:

- Returns `nil` if no `default` is provided.
- Returns the `default` value if provided.

```sheaf
(get {:a 1 :b 2} :a)  ; => 1
(get {:a 1} :missing 99) ; => 99
```

---

### get-in

**Type:** function  
**Signature:** `(get-in collection key-vector [default])`

Navigates through nested data structures (maps, lists, or tensors) using a sequence of keys. If the path exists, returns the value at the destination; otherwise, returns nil or the optional not-found value.

```sheaf
;; Deep navigation in a parameter dictionary
(get-in {:layers {:l1 {:w 10}}} [:layers :l1 :w])

; => 10

;; Missing path with default value (very common for configs)
(get-in {:a 1} [:layers :l1 :depth] 12)

; => 12

;; Mixed navigation (Map and Tensors)
;; Accesses: layer1 -> weights -> 1st row
(get-in {:l1 {:w [[1 2] [3 4]]}} [:l1 :w 0])  ;  ==> [1. 2.]
(get-in {:a {:b {:c 5}}} [:a :b :c]) ;  ==> 5
```

---

### assoc

**Type:** special-form  
**Signature:** `(assoc dict :key1 val1 :key2 val2 ...)`

Creates a new dictionary with updated key-value pairs. Does not mutate the original dictionary (functional update).

```sheaf
;; Add a new key
(assoc {:a 1} :b 2)

; => {:a 1, :b 2}

;; Update an existing key
(assoc {:a 1 :b 2} :a 10)

; => {:a 10, :b 2}

;; Multiple updates at once
(assoc {:layer1 {:w w1 :b b1}} :learning-rate 0.001 :epochs 100)

; => {:layer1 {...}, :learning-rate 0.001, :epochs 100}

```

---

### dissoc

**Type:** function  
**Signature:** `(dissoc dict keys-list)`

Creates a new dictionary with specified keys removed. Does not mutate the original dictionary (functional update). Keys are passed as a list.

```sheaf
;; Remove a single key
(dissoc {:a 1 :b 2 :c 3} [:b])

; => {:a 1, :c 3}

;; Remove multiple keys
(dissoc {:a 1 :b 2 :c 3} [:a :c])

; => {:b 2}

;; Remove keys that don't exist (no error, returns same dict)
(dissoc {:a 1} [:x :y])

; => {:a 1}

;; Useful for filtering state in scan/reduce
(let [state {:p params :m momentum :v velocity :t t :loss 0.5}]
  (dissoc state [:loss]))  ; Remove loss before scan

; => {:p ..., :m ..., :v ..., :t ...}

```

---

### merge

**Type:** function  
**Signature:** `(merge dict1 dict2 ...)`

Merges multiple dictionaries into one. Later dictionaries override earlier ones for conflicting keys. Does not mutate the originals (creates new dict).

```sheaf
;; Merge two dicts
(merge {:a 1} {:b 2})

; => {:a 1, :b 2}

;; Override on conflict (later dict wins)
(merge {:a 1 :b 2} {:b 20 :c 30})

; => {:a 1, :b 20, :c 30}

;; Merge configuration with defaults
(let [defaults {:lr 0.001 :epochs 10 :batch-size 32}
      user-config {:lr 0.0001 :batch-size 64}]
  (merge defaults user-config))

; => {:lr 0.0001, :epochs 10, :batch-size 64}
```

---

### keys

**Type:** function  
**Signature:** `(keys dict)`

Returns a list containing all the keys present in the dictionary. The order of keys is generally consistent with the insertion order but should not be relied upon for critical logic.

```sheaf
(keys {:a 1 :b 2 :c 3})

; => ['a', 'b', 'c']
```

---

### vals

**Type:** function  
**Signature:** `(vals dict)`

Returns a list of all values stored in the dictionary. This is particularly useful when combined with `tree-map` or other sequence operations to process all parameters of a model layer simultaneously.

```sheaf
(vals {:a 1 :b 2 :c 3})

; => [1, 2, 3]
```

---

## Control Flow

### if

**Type:** special-form  
**Signature:** `(if condition then-expr [else-expr])`

The fundamental conditional operator. It evaluates the `condition`; if truthy, it evaluates and returns `then-expr`. If falsy, it evaluates and returns `else-expr` (or `nil` if no else branch is provided). In compiled JIT contexts, `if` is used for structural control flow, not for element-wise tensor masking (use `where` for that).

```sheaf
(if (> 1 0) :yes :no) ; => :yes
(if false :a :b)      ; => :b
```

---

### case

**Type:** special-form  
**Signature:** `(case x match1 result1 ... [default])`

Provides efficient multi-branch dispatch. It evaluates `x` and compares it against each `match` literal. When a match is found, the corresponding `result` is returned. If no matches succeed and a `default` value is provided at the end, it is returned; otherwise, the result is `nil`.

```sheaf
(case 2
  1 :low
  2 :mid
  3 :high
  :unknown)           ; => :mid
```

---

### do

**Type:** special-form  
**Signature:** `(do expr1 expr2 ... exprN)`

Evaluates a sequence of expressions in order and returns the value of the last one. The preceding expressions are evaluated for their side effects only (e.g. `print`, `io "save"`). This is the idiomatic way to sequence imperative actions inside an otherwise functional context.

```sheaf
(do
  (print "loading...")
  (print "done.")
  42)                  ; => 42  (prints both messages, returns 42)

;; Typical use: side effect before a value
(let [params (do
               (print "Training...")
               (train data config epochs))]
  params)

;; Inside a reduce body: log then return new state
(reduce (fn [state step]
          (do
            (if (== (mod step 10) 0)
              (print (str-call "format" "Step {}" step))
              nil)
            (train-step state data)))
        init-state
        (range 500))
```

---

### repeat

**Type:** special-form  
**Signature:** `(repeat [i n] [acc init] body)`

Loops exactly `n` times. `i` is the iteration index (0-based), `acc` is the accumulator that carries state across iterations. The body must return the new value of `acc`. Returns the final accumulator value.

```sheaf
;; Sum 0..9
(repeat [i 10] [sum 0]
  (+ sum i))            ; => 45

;; Build a list
(repeat [i 5] [lst '()]
  (append lst (* i i))) ; => [0 1 4 9 16]

;; Accumulator is a dict — typical for training loops
(repeat [step 100] [state {:params p :loss 0.0}]
  (let [result (train-step (get state :params) x y lr)]
    {:params (get result :p)
     :loss   (get result :loss)}))
;; => {:params <trained> :loss <final-loss>}
```

Use `repeat` when the iteration count is known at compile time. For loops that depend on a runtime condition, see `while`.

---

### while

**Type:** special-form  
**Signature:** `(while cond [acc init] body)`

Loops as long as `cond` is true. `acc` is the accumulator, visible both in `cond` and in `body`. The body must return the new value of `acc`. Returns the accumulator value at the point where `cond` becomes false.

```sheaf
;; Count up to 5
(while (< n 5) [n 0]
  (+ n 1))              ; => 5

;; Accumulate until sum exceeds 100
(while (< (get s :sum) 100)
  [s {:sum 0 :n 1}]
  {:sum (+ (get s :sum) (get s :n))
   :n   (+ (get s :n) 1)})
;; => {:sum 105 :n 15}  (1+2+...+14 = 105)

;; Training loop that stops on convergence
(while (> (get state :loss) 0.01)
  [state {:params init-params :loss 999.0 :key key}]
  (let [result (train-step (get state :params) x y lr)
        [k1 k2] (random-split (get state :key))]
    {:params (get result :p)
     :loss   (get result :loss)
     :key    k1}))
```

Use `while` when the stopping condition depends on runtime values (e.g. loss convergence). For a fixed number of iterations, prefer `repeat`.

---

### let

**Type:** special-form  
**Signature:** `(let [var1 val1 var2 val2 ...] body)`

Evaluates expressions in a local scope where each variable name is bound to its corresponding value sequentially. Each binding can reference previously defined variables within the same block before returning the result of the body.

```sheaf
(let [x 1] x)         ; => 1

(let [x 1 y 2]
  (+ x y))            ; => 3
```

---

## Function Definition

### defn

**Type:** special-form
**Signature:** `(defn name [params] body ...)`

Binds a global name to a function. Multiple body expressions are allowed (implicit `do`): all are evaluated in order, and the last value is returned.

```sheaf
(defn square [x]
  (* x x))

(square 5)            ; => 25

;; Multiple body forms with implicit do
(defn train-step [model lr]
  (print "training...")
  (sgd-update model lr))
```

---

### fn

**Type:** special-form  
**Signature:** `(fn [params] body)`

Defines an anonymous function (lambda) that captures no external state (pure function). It is primarily used for short-lived transformations, mapping operations, or as arguments to higher-order functions like `map` or `vmap`.

```sheaf
((fn [x] (+ x 1)) 10)         ; => 11

(map (fn [x] (* x 2)) [1 2])  ; => [2 4]

;; Binding an anonymous function to a local name
(let [double (fn [n] (* n 2))]
  (double 21))                ; => 42
```

---

## Macros

### defmacro

**Type:** special-form  
**Signature:** `(defmacro name [params] body)`

Defines a macro that performs code transformation at compile-time. Unlike functions, macros receive their arguments as unevaluated data (S-expressions) and return a new expression that replaces the macro call before execution. This is primarily used for syntax extension and code generation without runtime overhead.

```sheaf
;; Define a 'when' macro (syntactic sugar for if)
(defmacro when [cond body]
  `(if ~cond ~body nil))

;; At compile-time, this call:
(when (> x 0) (log x))

;; ...is expanded into:
(if (> x 0) (log x) nil)

;; Logic for ensuring positive values
(when (> 5 0)
  (+ 1 2))   ; => 3

(when (< 5 0)
  (+ 1 2))   ; => nil (condition false)
```

---

## Threading & Composition

### ->

**Type:** special-form  
**Signature:** `(-> x (f1) (f2 a) ...)`

Threads the expression `x` through the provided forms. It inserts `x` as the first argument of the first form, then inserts the result as the first argument of the next form, and so on. This macro linearizes nested function calls, improving readability for sequential data transformations.

```sheaf
;; Linear sequence: (f3 (f2 (f1 x a) b))
(-> x
  (f1 a)     ; expands to (f1 x a)
  (f2 b)     ; expands to (f2 (result-f1) b)
  (f3))      ; expands to (f3 (result-f2))

;; Compute operations on an array
(-> [1 2 3]
  (sum)             ; => 6.0
  (+ 10))           ; => 16.0

;; Practical tensor pipeline
(-> (arange 12)
  (reshape 3 4)     ; Reshape to [3, 4]
  (sum :axis 0)     ; Sum columns => [12 13 14 15]
  (+ 10))           ; Add bias => [22 23 24 25]

```

---

### as->

**Type:** special-form  
**Signature:** `(as-> init name form1 form2 ...)`

Threads the initial expression init through the provided forms by binding its result to a symbol name. At each step, name is updated with the result of the previous form, allowing the data to be placed at any position within the next expression. This is the preferred way to handle pipelines where functions do not accept the threaded data as their first argument.

```sheaf
;; Simple math pipeline
(as-> 5 x
  (+ x 1)           ; x is bound to 5, result is 6
  (* x 2)           ; x is now 6, result is 12
  (- x 3))          ; x is now 12, result is 9

;; Non-first position (data used as second arg):
(as-> [1 2 3] data
  (map (fn [x] (* x 2)) data)  ; [Array(2.) Array(4.) Array(6.)]
  (apply + data))              ; => 12.0 (sum of mapped values)

;; Multiple uses in one step:
(as-> {:a 1 :b 2} m
  {:sum (+ (get m :a) (get m :b))
   :prod (* (get m :a) (get m :b))}) ; => {:sum 3 :prod 2}
```

---

## Parameter Management

### with-params

**Type:** special-form  
**Signature:** `(with-params [dict] body)` or `(with-params [dict :key] body)`

Unpacks the keys of a dictionary into the local scope as bound variables. If a `:key` is provided, it destructures the nested dictionary located at `(get dict :key)`.

```sheaf
;; Access root level keys (params dict)
(with-params [params]
  (+ W b))          ; W and b from root of params

;; Destructure a nested sub-dictionary
;; Accesses weights stored in p[:layer1]
(with-params [params :l1]
  (+ (@ x W) b))    ; W and b from params[:l1]

;; Functional composition
;; Dynamically select a layer's parameters from a list
(let [layers [{:W [[1.0] [2.0]] :b [0.1]}  ; Layer 0
              {:W [[3.0] [4.0]] :b [0.5]}] ; Layer 1
      layer-id 1                           ; Pick Layer 1
      x [1.0 1.0]]                         ; Input
  (with-params (get layers layer-id)
    (relu (+ (@ x W) b))))

```

---

## Transforms

### vmap

**Type:** special-form  
**Signature:** `(vmap func)` or `(vmap func in-axes)`

Vectorized mapping over batches. Applies function independently to each element along specified axes. Optional second argument controls axis mapping for multiple parameters.

```sheaf
;; Apply function to each row (default axis=0)
((vmap (fn [x] (sum x))) [[1 2 3] [4 5 6]]) ; => [ 6. 15.]

;; Specify axis explicitly (axis=0)
((vmap (fn [x] (* x 2)) 0) [[1 2 3] [4 5 6]]) ; => [[ 2.  4.  6.]
 [ 8. 10. 12.]]

;; Vmap along axis 1 (columns)
((vmap (fn [x] (sum x)) 1) [[1 2] [3 4]]) ; => [4. 6.]

;; Multiple parameters: vmap first arg, keep second fixed
((vmap (fn [x w] (+ x w)) [0 nil]) [[1 2] [3 4]] 10) ; => [[11. 12.]
 [13. 14.]]
```

---

### scan

**Type:** special-form  
**Signature:** `(scan func init-state xs)`

Iterates over `xs` while carrying a state. At each step, `func` is applied to the current state and the next element of `xs`.

The function `func` must return a pair: `[new-state, output]`.

Returns a vector `[final-state, stacked-outputs]`.

**Example 1: Accumulator (running sum)**

```sheaf
;; Simple accumulator: state is the running sum
(scan (fn [state x]
        [(+ state x)          ; new-state: accumulated sum
         (+ state x)])        ; output: emit current sum
      0.0
      [1.0 2.0 3.0])

; => [6.0, [1.0 3.0 6.0]]  ;; final state=6, outputs=[1, 1+2, 1+2+3]
```

**Example 2: RNN-style recurrence (state threading)**

For RNNs, state is the hidden vector. Each step processes `(h_prev, x_t) -> (h_next, output)`:

```sheaf
;; Concrete RNN example: h_t+1 = tanh(W_hh @ h_t + W_xh @ x_t + b_h)
;; Parameters: W_hh=[0.5 0.1; 0.2 0.3], W_xh=[0.1 0.2; 0.3 0.4], b_h=[0 0]
(let [W_hh [[0.5 0.1] [0.2 0.3]]
      W_xh [[0.1 0.2] [0.3 0.4]]
      b_h [0.0 0.0]
      h0 [0.0 0.0]                          ; initial hidden state
      X [[1.0 0.0] [0.0 1.0] [1.0 1.0]]]   ; sequence of 3 inputs
  (let [[h_final outputs]
        (scan (fn [h_prev x_t]
                (let [h_next (tanh (+ (@ W_hh h_prev)
                                      (@ W_xh x_t)
                                      b_h))]
                  [h_next h_next]))          ; [new_state, output]
              h0 X)]
    outputs))                                ; Return all hidden states

; => [[0.09966799 0.36936549]
;     [0.29131263 0.47375143]
;     [0.36652097 0.52315444]]
```

**Example 3: Multiple state components (like LSTM)**

```sheaf
;; State is a dict: {:h hidden :c cell}
;; Output is a vector (just the hidden state)
(fn [{:keys [h c]} x_t]
  (let [h_new (tanh (+ (@ W_hh h) (@ W_xh x_t)))
        c_new (+ (* f_t c) (* i_t (tanh (+ ...))))]
    [{:h h_new :c c_new}   ; new_state: dict with both h and c
     h_new]))              ; output: emit only h
```

**Notes:**

- If `func` returns only one element, `scan` will fail
- If state and output structures mismatch between iterations, `scan` will also fail
- `scan` is fully differentiable; use with `value-and-grad` for learning

---

### value-and-grad

**Type:** function  
**Signature:** `(value-and-grad func)`

A higher-order function that transforms a scalar-valued function `func` into a new function. This new function returns a list containing two elements: the original result (value) and the gradient(s) with respect to the first argument of `func`. The gradients share the same structure (Pytree) as the input parameters.

```sheaf

;; Basic scalar optimization
;; f(x) = x² + 1  => f'(x) = 2x
(let [loss-fn (fn [x] (+ (* x x) 1))
      grad-fn (value-and-grad loss-fn)
      [val grad] (grad-fn 3.0)]
  [val grad])

: => [10. 6.]

;; Optimization on parameter dictionaries
(let [p {:w 2.0 :b 1.0}
      loss (fn [params] (with-params [params] (+ (* w w) b)))
      [val grads] ((value-and-grad loss) p)]
  [(get grads :w) (get grads :b)])

; => [4. 1.]  ;; 2*w = 4.0, derivative of b = 1.0
```

---

## Loss & Metrics

### sparse-cross-entropy

**Type:** function
**Signature:** `(sparse-cross-entropy logits targets :i32)`

Computes the categorical cross-entropy loss between logits (unnormalized predictions) and targets (integer class indices). This function internally applies a softmax to the logits, making it more numerically stable than manual computation.

Note: `targets` must be integers. Specify the `:i32` type to avoid type errors.

```sheaf
(sparse-cross-entropy [[0.9 0.1] [0.2 0.8]] [0 1] :i32) ; => 0.40429434
```

---

## Advanced Tensor Ops

### einsum

**Type:** function  
**Signature:** `(einsum pattern x1 x2 [...])`

Computes the Einstein summation of the input tensors according to the specified `pattern`.

```sheaf
;; Dot product: i,i->
(einsum "i,i->" [1 2 3] [4 5 6]) ; => 32.0

;; Matrix multiply: ij,jk->ik
(einsum "ij,jk->ik" [[1 2] [3 4]] [[5 6] [7 8]]) ; => [[19. 22.] [43. 50.]]

;; Batch multiply: bij,bjk->bik
(einsum "bij,bjk->bik" [[[1 2] [3 4]]] [[[5 6] [7 8]]]) ; => [[[19. 22.] [43. 50.]]]

;; Element-wise with wildcard: ...i,...i->...
(einsum "...i,...i->..." [1 2 3] [4 5 6]) ; => 32.0
```

---

### top_k

**Type:** function  
**Signature:** `(top_k x k)`

Finds the `k` largest elements and their corresponding indices in the input tensor along the last axis. This is the standard operation for selecting the most likely next tokens during language model decoding and inference.

```sheaf
(let [[vals idxs] (top_k [0.1 0.9 0.3 0.7] 2)]
  {:values vals :indices idxs})     ; => values and indices of top 2
```

---

### tensor-split

**Type:** function  
**Signature:** `(tensor-split x num-sections [:axis axis])`

Splits a tensor into `num-sections` sub-tensors along the specified axis. This function is often used to divide a large hidden state into separate Query, Key, and Value projections in Transformer layers.

```sheaf
(let [[a b c] (tensor-split [1 2 3 4 5 6] 3)]
  a)  ; => [ 1. 2.] (first third)
```

---

### dynamic-slice

**Type:** function  
**Signature:** `(dynamic-slice x start length)`

Extracts a slice of a fixed `length` starting from a dynamic `start` index. The output shape (length) remains constant even when the starting position is a computed value.

```sheaf
(dynamic-slice (arange 5) 1 3)  ; => [1 2 3]
```

---

## Utilities

### normalize

**Type:** functions

Scales the elements of a tensor so they sum to 1.0. This is a common utility for processing probability distributions or attention weights.

```sheaf
(normalize [1 2 3])  ; => [0.16666667 0.33333334 0.5       ]

```

---

### str, gensym, symbol?

**Type:** functions

- `str`: Convert to string
- `gensym`: Generate unique symbol
- `symbol?`: Is it a symbol?

```sheaf
(str 42)                ; => '42
(gensym "var")          ; => Unique symbol
(symbol? 'W)            ; => true
```

---

## Strings

### Escape sequences

Sheaf strings support standard escape sequences. The backslash `\` triggers interpretation of the following character:

| Sequence | Meaning              |
| -------- | -------------------- |
| `\n`     | Newline              |
| `\t`     | Tab                  |
| `\"`     | Literal double-quote |
| `\\`     | Literal backslash    |

Any other character after `\` is kept as-is (backslash + character).

The tricky case is `\\n`: the first `\\` resolves to a literal backslash, then `n` is just `n` — the result is the two characters `\n`, not a newline.

```sheaf
(print "hello\nworld")   ; prints hello and world on separate lines
(print "col1\tcol2")     ; prints with a tab between
(print "say \"hi\"")     ; prints: say "hi"
(print "path\\name")     ; prints: path\name
(print "literal\\n")     ; prints: literal\n  (backslash + n, not newline)
```

---

### str-call

**Type:** function  
**Signature:** `(str-call method target [args ...])`

Calls a string method on `target`. This is the primary way to manipulate strings in Sheaf. The method name is passed as a string; subsequent arguments are forwarded directly.

```sheaf
(str-call "replace" "hello world" "world" "sheaf")   ; => "hello sheaf"
(str-call "splitlines" "a\nb\nc")                    ; => ["a" "b" "c"]
(str-call "join" ", " '("a" "b" "c"))                ; => "a, b, c"
(str-call "format" "x={} y={}" 1 2)                  ; => "x=1 y=2"
```

---

### print

**Type:** function
**Signature:** `(print arg ...)`, `(print fmt arg1 arg2 ...)`

Prints values to stdout. With multiple arguments, values are space-separated. If the first argument is a format string (contains `{}` or `{:`), uses format-string interpolation instead.

```sheaf
(print "hello")                        ; prints: hello
(print 42)                             ; prints: 42

;; Variadic: space-separated
(print "x=" x "y=" y)                 ; prints: x= 42 y= 7

;; Format string: placeholders filled by subsequent arguments
(print "x={} y={}" 10 20)             ; prints: x=10 y=20
(print "loss={:.4f} step={}" 0.0532 100)  ; prints: loss=0.0532 step=100
```

---

## I/O

### io

**Type:** function  
**Signature:** `(io verb [path] [data] [format-hint])`

Single entry point for all file and system I/O. The verb selects the operation; format is inferred from the file extension unless overridden with a keyword hint.

#### Verbs

**load** — deserialize a file into a value (pytree, string, dict, or mmap'd handle).

```sheaf
(io "load" "weights.pkl")                  ; pickle
(io "load" "model.safetensors")            ; lazy tensor handle
(io "load" "config.json")                  ; dict
(io "load" "weights.dat" :safetensors)     ; explicit format hint
(io "load" "train.npy")                    ; NpyHandle — mmap'd, zero-copy
(io "load" "tokens/shard-*.bin" :i32)      ; ShardedHandle — virtual concat over glob
```

**save** — serialize a value to a file. Directories are created automatically.

```sheaf
(io "save" "out/weights.pkl" params)
(io "save" "out/model.safetensors" params)
```

**read** — read a file as a raw string.

```sheaf
(io "read" "data/shakespeare.txt")         ; => full text as string
```

**lines** — return a lazy line iterator (streaming, no full file in memory).

```sheaf
(io "lines" "data/large.txt")              ; => LazyLines iterator
```

**exists** — check whether a file exists.

```sheaf
(io "exists" "out/weights.pkl")            ; => true / false
```

**entropy** — read bytes from the OS entropy source (`/dev/urandom`). Returns an integer. Default is 4 bytes (fits `int32`, suitable for `random-key`). Pass an optional byte count for larger values.

```sheaf
(io "entropy")                             ; => random int32 (4 bytes)
(io "entropy" 16)                          ; => random int (16 bytes, UUID-scale)

;; Typical usage: non-deterministic seed
(random-key (io "entropy"))                ; => fresh PRNG key each run
```

#### Supported formats

| Extension          | Format      | Notes                                                       |
| ------------------ | ----------- | ----------------------------------------------------------- |
| `.safetensors`     | safetensors | Lazy loading via mmap, dtype kept                           |
| `.npy`             | npy         | NumPy array, mmap'd → `NpyHandle`                           |
| `.bin`             | raw         | Raw binary, mmap'd. Dtype flag required (`:i32`, `:f32`, …) |
| `.pkl` / `.pickle` | pickle      | Legacy, discouraged                                         |
| `.txt`             | text        | Plain text                                                  |
| `.json`            | JSON        | Eager load as dict                                          |
| `.jsonl`           | JSONL       | Streaming via `lines`                                       |

Sharded models: pass a glob pattern or a HuggingFace index file.

```sheaf
(io "load" "shards/model-*.safetensors")             ; glob → merged safetensors
(io "load" "model.safetensors.index.json")           ; HF shard index
```

#### NpyHandle — single `.npy` file

`(io "load" "file.npy")` returns a **NpyHandle**: a lazy, mmap'd view over the array. No data is read until you slice.

```sheaf
(let [dataset (io "load" "train.npy")]       ; NpyHandle, shape [50000 784]
  (dataset 0)                                ; => first row  f32[784]   (zero-copy)
  (dataset 100:200))                         ; => rows 100–199  f32[100 784]
```

Slicing is along axis 0 only. The handle stays open for the lifetime of the binding — no explicit `close` needed.

#### ShardedHandle — virtual concat over glob

When training data is split across many files, use a glob pattern. The result is a **ShardedHandle**: a single virtual tensor that spans all shards, backed by independent mmaps.

```sheaf
;; .npy shards — dtype is read from each header automatically
(let [data (io "load" "data/train-*.npy")]
  (len data)                                 ; => total rows across all shards
  (data 0)                                   ; => first row of first shard
  (data 9999:10002))                         ; => cross-shard slice, transparent

;; .bin shards — dtype must be explicit
(let [tokens (io "load" "tokens/shard-*.bin" :i32)]
  (len tokens)                               ; => total token count
  (tokens 0:4096))                           ; => first 4096 tokens
```

Shards are sorted lexicographically. Use zero-padded names (`shard-001.bin`, not `shard-1.bin`) to keep the order correct.

Available dtype flags: `:f32`, `:f16`, `:bf16`, `:i32`, `:i16`, `:u32`, `:bool`.

Binary lookup across shards is O(log N) — reading a single element from shard 10 000 costs the same as from shard 1.

---

## Module System

### use

**Type:** special-form  
**Signature:** `(use module-name)`

Loads a library module and imports its public functions into the current global namespace. Use ':registry' to see what functions have been imported.

```sheaf
(use nn)              ; Load neural network ops
(use optim)           ; Load optimizer ops
```

---

## Quoting & Metaprogramming

### ' (quote)

**Type:** reader macro  
**Signature:** `'expr` or `(quote expr)`

Prevents evaluation of `expr`. It treats the expression as raw data (S-expression) instead of code to be executed.

- `[1 2 3]` -> Evaluates immediately into a Tensor.
- `'[1 2 3]` -> Remains a List/Vector of constants.

Quotes are used to pass arguments like shapes to functions like `reshape` or `ones`, which do not accept Tensors as inputs.

```sheaf
'symbol              ; => symbol (not evaluated)
'[1 2 3]             ; => (1, 2, 3)
(ones [2 2])         ; Will fail as [2 2] is seen as a dynamic tensor
(ones '[2 2])        ; Success: shape is passed as static data
```

---

### `` ` `` (quasiquote)

**Type:** reader macro  
**Signature:** `` `expr `` or `(quasiquote expr)`

Quote with selective evaluation using `~` and `~@`.

```sheaf
`(+ ~x 1)            ; x is evaluated, + and 1 are not

;; In macros:
(defmacro add1 [x]
  `(+ ~x 1))

(add1 5)              ; Expands to (+ 5 1) => 6

;; Unquote-splicing:
`(list ~@(range 3))  ; => (list 0 1 2)
```

---

## Additional

### append-and-roll

**Type:** function  
**Signature:** `(append-and-roll buffer value)`

Append value to buffer and roll (for autoregressive state).

```sheaf
(append-and-roll [1 2 3] 4) ; => [2. 3. 4.]
```

---

### range

**Type:** function
**Signature:** `(range [start] stop [step])`

Alias for `arange`. Generate integer sequence.

```sheaf
(range 5)            ; => [0 1 2 3 4]
(range 10 25 5)      ; => [10 15 20]
```

---

### var

**Type:** function  
**Signature:** `(var x [:axis axis :keepdims])`

Computes the variance of the tensor elements along the specified axis. It measures the spread of the data around the mean, a core component of normalization layers.

```sheaf
(var [1 2 3])        ; => 0.6666667
```

---

## Standard Library Modules

### Neural Network Operations (nn.shf)

The `nn` module provides essential building blocks for neural network construction.

#### linear

**Type:** function  
**Signature:** `(linear x w b)`

Linear transformation: `y = x @ w + b`. Applies a fully-connected layer to input `x` using weight matrix `w` and bias vector `b`.

```sheaf
(let [x [1.0 2.0]
      w [[1.0 0.5] [0.0 1.0]]
      b [0.1 0.2]]
  (linear x w b))

; => [1.1 2.7]
```

#### cross-entropy-loss

**Type:** function  
**Signature:** `(cross-entropy-loss logits targets)`

Categorical cross-entropy loss. Logits are unnormalized predictions; targets are integer class indices.

```sheaf
(let [logits [[10.0 1.0 0.1] [1.0 10.0 0.1]]
      targets [0 1] :i32]
  (cross-entropy-loss logits targets))

; => 0.00017355366435367614
```

#### mse-loss

**Type:** function  
**Signature:** `(mse-loss predictions targets)`

Mean squared error loss: `mean((predictions - targets)²)`. Heavily penalizes large outliers.

```sheaf
(mse-loss [1.0 2.0 3.0] [1.1 1.9 3.1])

; => 0.009999996982514858  (average squared error)
```

#### mae-loss

**Type:** function  
**Signature:** `(mae-loss predictions targets)`

Mean absolute error loss: `mean(|predictions - targets|)`. Provides a more robust linear penalty for errors compared to the MSE.

```sheaf
(mae-loss [1.0 2.0 3.0] [1.2 1.8 3.1])

; => 0.1666666716337204  (average absolute error)
```

#### layer-norm

**Type:** function  
**Signature:** `(layer-norm x p axis)`

Applies Layer Normalization. It re-centers and re-scales the input `x` along the specified `axis` to improve training stability. It uses the parameters in `p` (expected keys: `:gamma` and `:beta).

Default epsilon is `1e-5`.

```sheaf
(let [x [1.0 2.0 3.0]
      p {:gamma (ones [3]) :beta (zeros [3])}]
  (layer-norm x p 0))

; => [-1.2247356  0.         1.2247356]
```

#### rms-norm

**Type:** function  
**Signature:** `(rms-norm x p axis)`

Applies RMS Normalization (Root Mean Square Layer Normalization). Normalizes by the RMS of the input values without centering and uses the parameters in `p` (expected keys: `:gamma` and `:beta`).

Default epsilon is `1e-6`.

```sheaf
(let [x [1.0 2.0 3.0 4.0]
      p {:gamma (ones '[4]) :beta (zeros '[4])}]
  (rms-norm x p -1))

; => [0.36514837 0.73029673 1.095445   1.4605935 ]
```

#### clamp

**Type:** function
**Signature:** `(clamp x lo hi)`

Clamps values element-wise to the range [lo, hi]. Supports broadcasting between scalars and tensors.

```sheaf
(clamp 300 0 255)           ; => 255
(clamp [-50 100 300] 0 255) ; => [0. 100. 255.]
```

#### xavier-uniform

**Type:** function
**Signature:** `(xavier-uniform key shape)`

Xavier (Glorot) uniform initialization for weight matrices. Draws from `U(-limit, limit)` where `limit = sqrt(6 / (fan_in + fan_out))`. See also `xavier-normal`, `kaiming-uniform`, `kaiming-normal`, `lecun-normal`, `lecun-uniform`.

```sheaf
(let [key (random-key 42)]
  (xavier-uniform key '[256 256]))

; => f32[256x256]  (suitable for dense layer weights)
```

---

### Optimizer Operations (optim.shf)

The `optim` module provides optimization utilities for training neural networks.

#### sgd-step

**Type:** function  
**Signature:** `(sgd-step params grads learning-rate)`

Updates model parameters using Stochastic Gradient Descent. It performs the operation `p = p - (lr * g)` across the entire Pytree of parameters, maintaining the model's structural integrity.

```sheaf
(let [params {:w [1.0 2.0] :b 0.5}
      grads {:w [0.1 0.2] :b 0.05}
      lr 0.01]
  (sgd-step params grads lr))

; => {:w [0.999 1.998], :b 0.4995}
```

#### adam-step

**Type:** function  
**Signature:** `(adam-step params grads m v t lr [beta1 beta2 eps])`
Adam optimizer step. Maintains first moment `m` (mean) and second moment `v` (variance). Returns `[new-params, new-m, new-v]`.

```sheaf
(let [p {:w 1.0} g {:w 0.1}  ; Params and gradients
      m {:w 0.0} v {:w 0.0}  ; Moments
      t 0]                   ; Steps
  (let [[p1 m1 v1 t1] (adam-step p g m v t 0.001 0.9 0.999 1e-8)]
    p1))

; => [{:w 0.999}, {:w 0.0001}, {:w 0.00001}]
```

#### global-norm

**Type:** function  
**Signature:** `(global-norm grads)`

Computes the Euclidean (L2) norm of all scalar values contained within a Pytree. This is primarily used for gradient clipping, preventing numerical instability by scaling down gradients if their total magnitude exceeds a threshold.

```sheaf
(global-norm {:w [3.0 4.0] :b 0.0})

; => 5.0  (sqrt(3^2 + 4^2))
```

#### clip-by-global-norm

**Type:** function  
**Signature:** `(clip-by-global-norm grads max-norm)`

Clips gradients by global norm to prevent gradient explosion. If global norm > max-norm, scales all gradients down proportionally.

```sheaf
(clip-by-global-norm {:w [3.0 4.0] :b 0.0} 2.0)

; => {:w [1.2 1.6] :b 0.0}  (scaled by 2.0/5.0)
```

---

### Macros (macros.shf)

Macros are powerful meta-programming tools that allow allow architectural patterns to be abstracted once and expanded at compile-time.

#### when

**Type:** macro  
**Signature:** `(when condition body)`

Conditional execution without else branch. Expands to `(if condition body nil)`.

```sheaf
(when (> x 0)
  (log x))

;; Expands to:
(if (> x 0) (log x) nil)
```

#### unless

**Type:** macro  
**Signature:** `(unless condition body)`

Negated conditional. Expands to `(if condition nil body)`.

```sheaf
(unless (zero? x)
  (/ 1 x))

;; Expands to:
(if (zero? x) nil (/ 1 x))
```

#### comment

**Type:** macro  
**Signature:** `(comment expr ...)`

Ignores expressions and returns nil. Useful for multi-line comments.

```sheaf
(comment
  "This is a long analysis of the algorithm"
  "Multiple lines can go here"
  "They're all discarded at compile-time")

; => nil
```

#### defmodel

**Type:** macro
**Signature:** `(defmodel name [input-params] (layer :name) (layer :name :activation) ...)`

Defines a neural network as a sequence of named layers with optional activations. Each layer spec generates a `with-params` block that binds `W` and `b` from the parameter dictionary, applies `(+ (@ _ W) b)`, and optionally wraps it in an activation function. The input is threaded through layers via `as->`.

```sheaf
(defmodel forward [x]
  (layer :hidden1 relu)
  (layer :output))

;; Expands to:
(defn forward [x p]
  (as-> x _
    (with-params [p :hidden1] (relu (+ (@ _ W) b)))
    (with-params [p :output] (+ (@ _ W) b))))
```

#### defbatch

**Type:** macro
**Signature:** `(defbatch name [params] [axes] & body)`

Defines a function that auto-vectorizes over batch dimensions via `vmap`. The axes spec indicates which arguments are batched (`0`) vs shared (`nil`).

```sheaf
(defbatch linear-layer [x w b] [0 nil nil]
  (+ (@ x w) b))

;; Expands to:
(defn linear-layer [x w b]
  ((vmap (lambda [x w b] (+ (@ x w) b)) [0 nil nil]) x w b))
```

---
