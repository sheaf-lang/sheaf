// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! StableHLO emitter - generates MLIR StableHLO from Sheaf AST

use crate::ast::SheafValue;
use std::fmt::Write;

/// StableHLO type representation
#[derive(Debug, Clone, PartialEq)]
pub enum StableHLOType {
    /// Scalar tensor: tensor<f32>
    ScalarF32,
    /// Scalar tensor: tensor<f64>
    ScalarF64,
    /// Scalar tensor: tensor<i64>
    ScalarI64,
    /// Scalar tensor: tensor<i1> (boolean)
    ScalarI1,
    /// Tensor with shape: tensor<2x3xf32>
    Tensor { shape: Vec<i64>, dtype: String },
    /// Tuple of types: tuple<tensor<2x3xf32>, tensor<8xf32>>
    Tuple(Vec<StableHLOType>),
}

impl StableHLOType {
    pub fn scalar_f32() -> Self {
        Self::ScalarF32
    }

    pub fn scalar_i64() -> Self {
        Self::ScalarI64
    }

    pub fn f32_tensor(shape: Vec<i64>) -> Self {
        Self::Tensor {
            shape,
            dtype: "f32".to_string(),
        }
    }

    pub fn i64_tensor(shape: Vec<i64>) -> Self {
        Self::Tensor {
            shape,
            dtype: "i64".to_string(),
        }
    }

    pub fn i1_tensor(shape: Vec<i64>) -> Self {
        Self::Tensor {
            shape,
            dtype: "i1".to_string(),
        }
    }

    /// Get the shape of this type, or empty vec for scalars/tuples
    pub fn shape(&self) -> Vec<i64> {
        match self {
            Self::ScalarF32
            | Self::ScalarF64
            | Self::ScalarI64
            | Self::ScalarI1
            | Self::Tuple(_) => vec![],
            Self::Tensor { shape, .. } => shape.clone(),
        }
    }

    /// Get the dtype string
    pub fn dtype(&self) -> &str {
        match self {
            Self::ScalarF32 => "f32",
            Self::ScalarF64 => "f64",
            Self::ScalarI64 => "i64",
            Self::ScalarI1 => "i1",
            Self::Tensor { dtype, .. } => dtype,
            Self::Tuple(_) => "tuple",
        }
    }

    pub fn to_mlir(&self) -> String {
        match self {
            Self::ScalarF32 => "tensor<f32>".to_string(),
            Self::ScalarF64 => "tensor<f64>".to_string(),
            Self::ScalarI64 => "tensor<i64>".to_string(),
            Self::ScalarI1 => "tensor<i1>".to_string(),
            Self::Tensor { shape, dtype } => {
                let shape_str = shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x");
                format!("tensor<{}x{}>", shape_str, dtype)
            }
            Self::Tuple(elems) => {
                let elems_str = elems
                    .iter()
                    .map(|t| t.to_mlir())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("tuple<{}>", elems_str)
            }
        }
    }
}

/// Register name in SSA form: %0, %1, etc. or %arg0, %arg1, etc.
#[derive(Debug, Clone, PartialEq)]
pub enum Register {
    /// Regular SSA register: %0, %1, etc.
    Reg(usize),
    /// Function argument: %arg0, %arg1, etc.
    Arg(usize),
}

impl Register {
    pub fn new(id: usize) -> Self {
        Self::Reg(id)
    }

    pub fn arg(id: usize) -> Self {
        Self::Arg(id)
    }

    pub fn to_mlir(&self) -> String {
        match self {
            Self::Reg(id) => format!("%{}", id),
            Self::Arg(id) => format!("%arg{}", id),
        }
    }
}

/// MLIR StableHLO emitter
pub struct StableHLOEmitter {
    counter: usize,
    pub(crate) body: Vec<String>,
}

impl StableHLOEmitter {
    pub fn new() -> Self {
        Self {
            counter: 0,
            body: Vec::new(),
        }
    }

    /// Generate a fresh register name
    pub fn fresh_register(&mut self) -> Register {
        let reg = Register::new(self.counter);
        self.counter += 1;
        reg
    }

    /// Add an instruction to the body
    pub fn emit_instruction(&mut self, instruction: String) {
        self.body.push(instruction);
    }

    /// Emit a constant scalar
    pub fn emit_constant_f32(&mut self, value: f64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::scalar_f32();
        // Format with .0 if integer value to satisfy IREE
        let value_str = if value.fract() == 0.0 && value.is_finite() {
            format!("{:.1}", value)
        } else {
            format!("{}", value)
        };
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value_str,
            ty.to_mlir()
        ));
        reg
    }

    /// Emit a constant integer
    pub fn emit_constant_i64(&mut self, value: i64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::ScalarI64;
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value,
            ty.to_mlir()
        ));
        reg
    }

    /// Emit a tensor constant from a nested vector
    /// For example: [[1.0, 2.0], [3.0, 4.0]] -> tensor<2x2xf32>
    pub fn emit_tensor_constant(&mut self, values: &[Vec<f64>]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        // Infer shape from nested structure
        let rows = values.len();
        let cols = if rows > 0 { values[0].len() } else { 0 };
        let shape = vec![rows as i64, cols as i64];
        let ty = StableHLOType::f32_tensor(shape);

        // Build nested structure for dense representation
        let rows_str: Vec<String> = values
            .iter()
            .map(|row| {
                let row_values: Vec<String> = row
                    .iter()
                    .map(|&v| {
                        if v.fract() == 0.0 && v.is_finite() {
                            format!("{:.1}", v)
                        } else {
                            format!("{}", v)
                        }
                    })
                    .collect();
                format!("[{}]", row_values.join(", "))
            })
            .collect();

        let values_str = rows_str.join(", ");

        self.body.push(format!(
            "    {} = stablehlo.constant dense<[{}]> : {}",
            reg.to_mlir(),
            values_str,
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit a binary operation with broadcasting support
    pub fn emit_binop(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let stablehlo_op = match op {
            "+" => "stablehlo.add",
            "-" => "stablehlo.subtract",
            "*" => "stablehlo.multiply",
            "/" => "stablehlo.divide",
            "**" => "stablehlo.power",
            "//" => "stablehlo.floor_divide",
            "%" | "mod" => "stablehlo.remainder",
            "min" => "stablehlo.minimum",
            "max" => "stablehlo.maximum",
            _ => panic!("Unsupported binop: {}", op),
        };

        // Determine result type (broadcast if needed)
        let result_ty = self.broadcast_types(lhs_ty, rhs_ty);

        // Check if we need to broadcast operands
        let (actual_lhs, actual_rhs) =
            self.maybe_broadcast_operands(lhs, rhs, lhs_ty, rhs_ty, &result_ty);

        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = {} {}, {} : {}",
            reg.to_mlir(),
            stablehlo_op,
            actual_lhs.to_mlir(),
            actual_rhs.to_mlir(),
            result_ty.to_mlir()
        ));
        (reg, result_ty)
    }

    /// Emit a comparison operation
    /// Returns a tensor of i1 (boolean) values
    pub fn emit_compare(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let comparison_direction = match op {
            "=" | "==" => "EQ",
            "!=" => "NE",
            "<" => "LT",
            "<=" => "LE",
            ">" => "GT",
            ">=" => "GE",
            _ => panic!("Unsupported comparison: {}", op),
        };

        // Determine result shape (broadcast if needed)
        let operand_ty = self.broadcast_types(lhs_ty, rhs_ty);

        // Check if we need to broadcast operands
        let (actual_lhs, actual_rhs) =
            self.maybe_broadcast_operands(lhs, rhs, lhs_ty, rhs_ty, &operand_ty);

        // Result type: same shape as operands but with i1 dtype
        let result_ty = if operand_ty.shape().is_empty() {
            // Scalar comparison returns scalar i1
            StableHLOType::ScalarI1
        } else {
            // Tensor comparison returns tensor of same shape with i1 elements
            StableHLOType::i1_tensor(operand_ty.shape())
        };

        let reg = self.fresh_register();
        // Use generic form with attributes in braces
        self.body.push(format!(
            "    {} = \"stablehlo.compare\"({}, {}) {{",
            reg.to_mlir(),
            actual_lhs.to_mlir(),
            actual_rhs.to_mlir()
        ));
        self.body.push(format!(
            "      comparison_direction = #stablehlo<comparison_direction {}>,",
            comparison_direction
        ));
        self.body.push(format!(
            "      compare_type = #stablehlo<comparison_type FLOAT>"
        ));
        self.body.push(format!(
            "    }} : ({}, {}) -> {}",
            operand_ty.to_mlir(),
            operand_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Maybe broadcast operands to match result shape
    /// Returns (lhs_reg, rhs_reg) which may be the originals or broadcasted versions
    fn maybe_broadcast_operands(
        &mut self,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
        result_ty: &StableHLOType,
    ) -> (Register, Register) {
        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();
        let result_shape = result_ty.shape();

        // Broadcast lhs if needed
        let actual_lhs = if lhs_shape != result_shape && !result_shape.is_empty() {
            self.emit_broadcast(lhs, lhs_ty, result_ty)
        } else {
            lhs.clone()
        };

        // Broadcast rhs if needed
        let actual_rhs = if rhs_shape != result_shape && !result_shape.is_empty() {
            self.emit_broadcast(rhs, rhs_ty, result_ty)
        } else {
            rhs.clone()
        };

        (actual_lhs, actual_rhs)
    }

    /// Emit broadcast_in_dim to convert from_ty to to_ty
    fn emit_broadcast(
        &mut self,
        operand: &Register,
        from_ty: &StableHLOType,
        to_ty: &StableHLOType,
    ) -> Register {
        let from_shape = from_ty.shape();
        let to_shape = to_ty.shape();

        // Determine broadcast dimensions
        // For [8] -> [4, 8], broadcast on dimension 1
        // For [] -> [4, 8], broadcast scalar (no dims needed)
        let dims = if from_shape.is_empty() {
            // Scalar broadcast
            vec![]
        } else if from_shape.len() == 1 && to_shape.len() == 2 {
            // Vector to matrix: broadcast on last dimension
            vec![1]
        } else {
            // Default: assume trailing dimensions match
            let offset = to_shape.len() - from_shape.len();
            (offset..to_shape.len()).collect()
        };

        let reg = self.fresh_register();
        if dims.is_empty() {
            // Scalar broadcast
            self.body.push(format!(
                "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
                reg.to_mlir(),
                operand.to_mlir(),
                from_ty.to_mlir(),
                to_ty.to_mlir()
            ));
        } else {
            let dims_str = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            self.body.push(format!(
                "    {} = stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}",
                reg.to_mlir(),
                operand.to_mlir(),
                dims_str,
                from_ty.to_mlir(),
                to_ty.to_mlir()
            ));
        }

        reg
    }

    /// Broadcast types: choose result type for binary op
    /// For now, simple heuristic: larger shape wins, scalars broadcast
    fn broadcast_types(&self, lhs: &StableHLOType, rhs: &StableHLOType) -> StableHLOType {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        // If both are scalars, return scalar
        if lhs_shape.is_empty() && rhs_shape.is_empty() {
            return lhs.clone();
        }

        // If one is scalar, return the other (scalar broadcasts)
        if lhs_shape.is_empty() {
            return rhs.clone();
        }
        if rhs_shape.is_empty() {
            return lhs.clone();
        }

        // Otherwise, prefer lhs shape (TODO: proper numpy-style broadcasting)
        lhs.clone()
    }

    /// Emit a matrix multiply (dot_general)
    /// For simple 2D matrix multiply: [M, K] @ [K, N] -> [M, N]
    pub fn emit_matmul(
        &mut self,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        // Shape inference for matmul
        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();

        // Simple 2D matmul for now: [M, K] @ [K, N] -> [M, N]
        let result_shape = if lhs_shape.len() == 2 && rhs_shape.len() == 2 {
            vec![lhs_shape[0], rhs_shape[1]]
        } else {
            // Fallback: assume result is same as lhs
            lhs_shape.clone()
        };

        let result_ty = StableHLOType::f32_tensor(result_shape);
        let reg = self.fresh_register();

        // dot_general with contracting dimensions
        // For [M,K] @ [K,N]: contract on dimension 1 of lhs and 0 of rhs
        self.body.push(format!(
            "    {} = stablehlo.dot_general {}, {}, contracting_dims = [1] x [0] : ({}, {}) -> {}",
            reg.to_mlir(),
            lhs.to_mlir(),
            rhs.to_mlir(),
            lhs_ty.to_mlir(),
            rhs_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit select operation (conditional): select(pred, on_true, on_false)
    pub fn emit_select(
        &mut self,
        pred: &Register,
        on_true: &Register,
        on_false: &Register,
        pred_ty: &StableHLOType,
        on_true_ty: &StableHLOType,
        _on_false_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        // Result type is the type of the branches (assume they match)
        let result_ty = on_true_ty.clone();

        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.select {}, {}, {} : {}, {}",
            reg.to_mlir(),
            pred.to_mlir(),
            on_true.to_mlir(),
            on_false.to_mlir(),
            pred_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit boolean binary operation (and, or)
    pub fn emit_bool_binop(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let stablehlo_op = match op {
            "and" => "stablehlo.and",
            "or" => "stablehlo.or",
            _ => panic!("Unsupported boolean binop: {}", op),
        };

        // Determine result type (broadcast if needed)
        let result_ty = self.broadcast_types(lhs_ty, rhs_ty);

        // Check if we need to broadcast operands
        let (actual_lhs, actual_rhs) =
            self.maybe_broadcast_operands(lhs, rhs, lhs_ty, rhs_ty, &result_ty);

        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = {} {}, {} : {}",
            reg.to_mlir(),
            stablehlo_op,
            actual_lhs.to_mlir(),
            actual_rhs.to_mlir(),
            result_ty.to_mlir()
        ));
        (reg, result_ty)
    }

    /// Emit unary operation (relu, sigmoid, tanh, etc.)
    pub fn emit_unary(&mut self, op: &str, operand: &Register, ty: &StableHLOType) -> Register {
        let reg = self.fresh_register();

        match op {
            "relu" => {
                // ReLU = max(x, 0)
                let zero_reg = self.emit_constant_f32(0.0);
                let zero_ty = StableHLOType::scalar_f32();

                // Broadcast zero to match operand shape if needed
                let broadcasted_zero = if ty.shape() != zero_ty.shape() && !ty.shape().is_empty() {
                    self.emit_broadcast(&zero_reg, &zero_ty, ty)
                } else {
                    zero_reg
                };

                self.body.push(format!(
                    "    {} = stablehlo.maximum {}, {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    broadcasted_zero.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "sigmoid" => {
                // sigmoid(x) = 1 / (1 + exp(-x))
                // Step 1: negate x
                let neg_reg = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.negate {} : {}",
                    neg_reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));

                // Step 2: exp(-x)
                let exp_reg = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.exponential {} : {}",
                    exp_reg.to_mlir(),
                    neg_reg.to_mlir(),
                    ty.to_mlir()
                ));

                // Step 3: 1 + exp(-x)
                let one_reg = self.emit_constant_f32(1.0);
                let one_ty = StableHLOType::scalar_f32();
                let broadcasted_one = if ty.shape() != one_ty.shape() && !ty.shape().is_empty() {
                    self.emit_broadcast(&one_reg, &one_ty, ty)
                } else {
                    one_reg.clone()
                };

                let one_plus_exp = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.add {}, {} : {}",
                    one_plus_exp.to_mlir(),
                    broadcasted_one.to_mlir(),
                    exp_reg.to_mlir(),
                    ty.to_mlir()
                ));

                // Step 4: 1 / (1 + exp(-x))
                self.body.push(format!(
                    "    {} = stablehlo.divide {}, {} : {}",
                    reg.to_mlir(),
                    broadcasted_one.to_mlir(),
                    one_plus_exp.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "tanh" => {
                self.body.push(format!(
                    "    {} = stablehlo.tanh {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "sqrt" => {
                self.body.push(format!(
                    "    {} = stablehlo.sqrt {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "exp" => {
                self.body.push(format!(
                    "    {} = stablehlo.exponential {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "log" => {
                self.body.push(format!(
                    "    {} = stablehlo.log {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "not" => {
                // Boolean NOT
                self.body.push(format!(
                    "    {} = stablehlo.not {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "abs" => {
                self.body.push(format!(
                    "    {} = stablehlo.abs {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            _ => panic!("Unsupported unary op: {}", op),
        }

        reg
    }

    /// Emit zeros tensor: (zeros [M N]) -> tensor<MxNxf32>
    pub fn emit_zeros(&mut self, shape: &[i64]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.constant dense<0.0> : {}",
            reg.to_mlir(),
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit random-normal tensor: (random-normal key [M N])
    /// For now, we emit a constant with small values (placeholder)
    /// TODO: Proper RNG with seed/key
    pub fn emit_random_normal(&mut self, shape: &[i64]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        // Placeholder: emit constant with 0.01 (will need proper RNG later)
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0.01> : {}",
            reg.to_mlir(),
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit ones tensor: (ones [M N]) -> tensor<MxNxf32>
    pub fn emit_ones(&mut self, shape: &[i64]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.constant dense<1.0> : {}",
            reg.to_mlir(),
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit reshape: (reshape tensor [M N]) -> tensor<MxNxf32>
    pub fn emit_reshape(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        new_shape: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let result_ty = StableHLOType::f32_tensor(new_shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            reg.to_mlir(),
            operand.to_mlir(),
            operand_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit transpose: (transpose tensor [1 0]) -> permutes dimensions
    pub fn emit_transpose(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        permutation: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        // Compute result shape by applying permutation
        let operand_shape = operand_ty.shape();
        let result_shape: Vec<i64> = permutation
            .iter()
            .map(|&i| operand_shape[i as usize])
            .collect();

        let result_ty = StableHLOType::f32_tensor(result_shape);

        // Format permutation as [0, 1, 2]
        let perm_str = permutation
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        self.body.push(format!(
            "    {} = stablehlo.transpose {}, dims = [{}] : ({}) -> {}",
            reg.to_mlir(),
            operand.to_mlir(),
            perm_str,
            operand_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit iota (arange): (arange N) -> tensor<Nxf32> with values [0, 1, 2, ..., N-1]
    /// For now, creates a 1D tensor. Can be extended to multi-dimensional later.
    pub fn emit_iota(&mut self, shape: &[i64], dimension: i64) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.iota dim = {} : {}",
            reg.to_mlir(),
            dimension,
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit concatenate: (concat [tensor1 tensor2 ...] axis)
    pub fn emit_concatenate(
        &mut self,
        operands: &[Register],
        operand_types: &[StableHLOType],
        dimension: i64,
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        // Compute result shape: same as first operand except for concat dimension
        let first_shape = operand_types[0].shape();
        let mut result_shape = first_shape.clone();

        // Sum the sizes along the concatenation dimension
        let concat_dim_size: i64 = operand_types
            .iter()
            .map(|ty| ty.shape()[dimension as usize])
            .sum();

        result_shape[dimension as usize] = concat_dim_size;
        let result_ty = StableHLOType::f32_tensor(result_shape);

        // Format operands as %0, %1, %2
        let operands_str = operands
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        // Format types as (tensor<2x3xf32>, tensor<2x3xf32>)
        let types_str = operand_types
            .iter()
            .map(|ty| ty.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        self.body.push(format!(
            "    {} = stablehlo.concatenate {}, dim = {} : ({}) -> {}",
            reg.to_mlir(),
            operands_str,
            dimension,
            types_str,
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit where (conditional selection): (where condition x y)
    /// Selects elements from x when condition is true, from y when false
    pub fn emit_where(
        &mut self,
        condition: &Register,
        x: &Register,
        y: &Register,
        condition_ty: &StableHLOType,
        x_ty: &StableHLOType,
        y_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        // Result type is the same as x and y (they should match)
        let result_ty = x_ty.clone();

        // Use stablehlo.select: select(pred, on_true, on_false)
        self.emit_select(condition, x, y, condition_ty, x_ty, y_ty)
    }

    /// Emit swapaxes: (swapaxes x axis1 axis2)
    /// Interchanges two axes using transpose
    pub fn emit_swapaxes(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        axis1: i64,
        axis2: i64,
    ) -> (Register, StableHLOType) {
        let operand_shape = operand_ty.shape();
        let rank = operand_shape.len();

        // Build permutation that swaps axis1 and axis2
        let mut permutation: Vec<i64> = (0..rank as i64).collect();
        permutation[axis1 as usize] = axis2;
        permutation[axis2 as usize] = axis1;

        // Use transpose with the permutation
        self.emit_transpose(operand, operand_ty, &permutation)
    }

    /// Emit tril (lower triangular): (tril x)
    /// Returns the lower triangular part of a matrix, zeros above diagonal
    pub fn emit_tril(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let result_ty = operand_ty.clone();

        // Create a triangular mask using iota operations
        // For a [M, N] matrix, we need:
        // mask[i, j] = (i >= j) ? 1.0 : 0.0
        let shape = operand_ty.shape();
        if shape.len() != 2 {
            panic!("tril requires a 2D tensor");
        }

        let m = shape[0];
        let n = shape[1];

        // Create row indices: iota dim=0 creates [[0], [1], [2], ...]
        let row_iota = self.fresh_register();
        let row_iota_ty = StableHLOType::f32_tensor(vec![m, n]);
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 0 : {}",
            row_iota.to_mlir(),
            row_iota_ty.to_mlir()
        ));

        // Create col indices: iota dim=1 creates [[0, 1, 2, ...], [0, 1, 2, ...], ...]
        let col_iota = self.fresh_register();
        let col_iota_ty = StableHLOType::f32_tensor(vec![m, n]);
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 1 : {}",
            col_iota.to_mlir(),
            col_iota_ty.to_mlir()
        ));

        // Compare: row_idx >= col_idx (i >= j for lower triangle)
        let mask = self.fresh_register();
        let mask_ty = StableHLOType::i1_tensor(vec![m, n]); // Comparison returns i1 (boolean)
        self.body.push(format!(
            "    {} = \"stablehlo.compare\"({}, {}) {{",
            mask.to_mlir(),
            row_iota.to_mlir(),
            col_iota.to_mlir()
        ));
        self.body
            .push("      comparison_direction = #stablehlo<comparison_direction GE>,".to_string());
        self.body
            .push("      compare_type = #stablehlo<comparison_type FLOAT>".to_string());
        self.body.push(format!(
            "    }} : ({}, {}) -> {}",
            row_iota_ty.to_mlir(),
            col_iota_ty.to_mlir(),
            mask_ty.to_mlir()
        ));

        // Create zero tensor for the false branch
        let zero_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0.0> : {}",
            zero_reg.to_mlir(),
            result_ty.to_mlir()
        ));

        // Select: where mask is true, use operand, else use zero
        self.body.push(format!(
            "    {} = stablehlo.select {}, {}, {} : {}, {}",
            reg.to_mlir(),
            mask.to_mlir(),
            operand.to_mlir(),
            zero_reg.to_mlir(),
            mask_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit get_tuple_element: extract field from a tuple at a given index
    pub fn emit_get_tuple_element(
        &mut self,
        tuple_reg: &Register,
        tuple_ty: &StableHLOType,
        index: usize,
        result_ty: &StableHLOType,
    ) -> Register {
        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.get_tuple_element {} [{index}] : ({}) -> {}",
            reg.to_mlir(),
            tuple_reg.to_mlir(),
            tuple_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        reg
    }

    /// Emit stablehlo.reduce to compute sum along one axis.
    ///
    /// input: tensor<...xf32>, axis: which dimension to reduce
    /// returns: tensor with that dimension removed (or size 1 if keepdims)
    pub fn emit_reduce_sum(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        // Treat scalar as 1D tensor of size 1
        let shape = if shape.is_empty() { vec![1i64] } else { shape };
        let ndim = shape.len();
        let axis_usize = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };
        let axis_usize = axis_usize.min(ndim.saturating_sub(1));

        // Result shape: remove the reduced axis
        let reduced_shape: Vec<i64> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis_usize)
            .map(|(_, &d)| d)
            .collect();

        let result_ty = if reduced_shape.is_empty() {
            StableHLOType::scalar_f32()
        } else {
            StableHLOType::f32_tensor(reduced_shape.clone())
        };

        // Zero initializer (scalar)
        let zero_reg = self.emit_constant_f32(0.0);
        let result_reg = self.fresh_register();

        self.body.push(format!(
            "    {} = stablehlo.reduce({} init: {}) applies stablehlo.add across dimensions = [{}] : ({}, {}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            zero_reg.to_mlir(),
            axis_usize,
            input_ty.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
            result_ty.to_mlir(),
        ));

        if keepdims {
            // Re-insert the reduced dimension as size 1
            let mut keepdim_shape = shape.clone();
            keepdim_shape[axis_usize] = 1;
            let keepdim_ty = StableHLOType::f32_tensor(keepdim_shape);

            // broadcast_in_dim to restore the axis
            let dims: Vec<String> = (0..ndim)
                .filter(|i| *i != axis_usize)
                .map(|i| i.to_string())
                .collect();
            let keepdim_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                keepdim_reg.to_mlir(),
                result_reg.to_mlir(),
                result_ty.to_mlir(),
                keepdim_ty.to_mlir(),
            ));
            (keepdim_reg, keepdim_ty)
        } else {
            (result_reg, result_ty)
        }
    }

    /// Emit stablehlo.reduce to compute mean along one axis.
    pub fn emit_reduce_mean(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();
        let axis_usize = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };
        let n = shape[axis_usize] as f64;

        let (sum_reg, sum_ty) = self.emit_reduce_sum(input, input_ty, axis, keepdims);

        // Divide by n
        let n_reg = self.emit_constant_f32(n);

        // Broadcast n to match sum shape if needed
        let result_reg = self.fresh_register();
        if sum_ty == StableHLOType::scalar_f32() {
            self.body.push(format!(
                "    {} = stablehlo.divide {}, {} : {}",
                result_reg.to_mlir(),
                sum_reg.to_mlir(),
                n_reg.to_mlir(),
                sum_ty.to_mlir(),
            ));
        } else {
            let n_broadcast_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
                n_broadcast_reg.to_mlir(),
                n_reg.to_mlir(),
                StableHLOType::scalar_f32().to_mlir(),
                sum_ty.to_mlir(),
            ));
            self.body.push(format!(
                "    {} = stablehlo.divide {}, {} : {}",
                result_reg.to_mlir(),
                sum_reg.to_mlir(),
                n_broadcast_reg.to_mlir(),
                sum_ty.to_mlir(),
            ));
        }
        (result_reg, sum_ty)
    }

    /// Emit a return statement
    pub fn emit_return(&mut self, reg: &Register, ty: &StableHLOType) {
        self.body
            .push(format!("    return {} : {}", reg.to_mlir(), ty.to_mlir()));
    }

    /// Emit a multi-value return: `return %0, %1 : type0, type1`
    pub fn emit_return_multi(&mut self, regs: &[Register], types: &[StableHLOType]) {
        let vals = regs
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");
        let tys = types
            .iter()
            .map(|t| t.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");
        self.body.push(format!("    return {} : {}", vals, tys));
    }

    /// Emit a function with multiple return values.
    ///
    /// `-> (t1, t2, t3)` syntax in StableHLO.
    pub fn emit_func_declaration_multi(
        &mut self,
        name: &str,
        param_types: &[StableHLOType],
        return_types: &[StableHLOType],
        body_instructions: &[String],
    ) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();

        let params: Vec<String> = param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("%arg{}: {}", i, ty.to_mlir()))
            .collect();
        let params_str = params.join(", ");

        let ret_str = if return_types.len() == 1 {
            return_types[0].to_mlir()
        } else {
            format!(
                "({})",
                return_types
                    .iter()
                    .map(|t| t.to_mlir())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        writeln!(
            output,
            "  func.func @{}({}) -> {} {{",
            sanitized_name, params_str, ret_str
        )
        .unwrap();

        for line in body_instructions {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        output
    }

    /// Emit a function declaration (func.func)
    ///
    /// Generates: func.func @name(%arg0: type0, %arg1: type1) -> return_type { ... }
    pub fn emit_func_declaration(
        &mut self,
        name: &str,
        param_types: &[StableHLOType],
        return_type: &StableHLOType,
        body_instructions: &[String],
    ) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();

        // Generate parameter list: %arg0: tensor<f32>, %arg1: tensor<f32>
        let params: Vec<String> = param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("%arg{}: {}", i, ty.to_mlir()))
            .collect();
        let params_str = params.join(", ");

        writeln!(
            output,
            "  func.func @{}({}) -> {} {{",
            sanitized_name,
            params_str,
            return_type.to_mlir()
        )
        .unwrap();

        // Emit body instructions
        for line in body_instructions {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        output
    }

    /// Emit a function call (func.call)
    ///
    /// Generates: %result = func.call @name(%arg0, %arg1) : (type0, type1) -> return_type
    pub fn emit_func_call(
        &mut self,
        name: &str,
        arg_registers: &[Register],
        arg_types: &[StableHLOType],
        return_type: &StableHLOType,
    ) -> Register {
        let sanitized_name = Self::sanitize_func_name(name);
        let reg = self.fresh_register();

        // Build argument list: %0, %1, %2
        let args_str = arg_registers
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        // Build type signature: (tensor<f32>, tensor<f32>) -> tensor<f32>
        let arg_types_str = arg_types
            .iter()
            .map(|ty| ty.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        self.body.push(format!(
            "    {} = func.call @{}({}) : ({}) -> {}",
            reg.to_mlir(),
            sanitized_name,
            args_str,
            arg_types_str,
            return_type.to_mlir()
        ));

        reg
    }

    /// Compile an expression to a register
    pub fn compile_expr(&mut self, expr: &SheafValue) -> (Register, StableHLOType) {
        match expr {
            // Constants
            SheafValue::Float(x, _) => {
                let reg = self.emit_constant_f32(*x);
                (reg, StableHLOType::scalar_f32())
            }
            SheafValue::Integer(n, _) => {
                // For now, treat integers as floats for compatibility
                let reg = self.emit_constant_f32(*n as f64);
                (reg, StableHLOType::scalar_f32())
            }

            // Vectors as tensor literals: [1.0 2.0] or [[1.0 2.0] [3.0 4.0]]
            SheafValue::Vector(elems, _) => {
                // Check if nested vector (matrix)
                if !elems.is_empty() && matches!(elems[0], SheafValue::Vector(_, _)) {
                    // Matrix literal: [[row1] [row2] ...]
                    let rows: Vec<Vec<f64>> = elems
                        .iter()
                        .map(|row| {
                            if let SheafValue::Vector(row_elems, _) = row {
                                row_elems
                                    .iter()
                                    .map(|e| match e {
                                        SheafValue::Float(x, _) => *x,
                                        SheafValue::Integer(n, _) => *n as f64,
                                        _ => panic!("Matrix element must be number"),
                                    })
                                    .collect()
                            } else {
                                panic!("Matrix rows must be vectors")
                            }
                        })
                        .collect();
                    self.emit_tensor_constant(&rows)
                } else {
                    // 1D vector - treat as 1xN tensor
                    let values: Vec<f64> = elems
                        .iter()
                        .map(|e| match e {
                            SheafValue::Float(x, _) => *x,
                            SheafValue::Integer(n, _) => *n as f64,
                            _ => panic!("Vector element must be number"),
                        })
                        .collect();
                    self.emit_tensor_constant(&vec![values])
                }
            }

            // List forms: function calls, special ops
            SheafValue::List(elems, _) if !elems.is_empty() => {
                if let Some(op) = elems[0].as_symbol() {
                    match op {
                        // Binary operations: (+ a b), (- a b), (* a b), (/ a b)
                        "+" | "-" | "*" | "/" if elems.len() == 3 => {
                            let (lhs_reg, lhs_ty) = self.compile_expr(&elems[1]);
                            let (rhs_reg, rhs_ty) = self.compile_expr(&elems[2]);
                            self.emit_binop(op, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty)
                        }

                        // Matmul: (@ A B)
                        "@" if elems.len() == 3 => {
                            let (lhs_reg, lhs_ty) = self.compile_expr(&elems[1]);
                            let (rhs_reg, rhs_ty) = self.compile_expr(&elems[2]);
                            self.emit_matmul(&lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty)
                        }

                        // Unary operations: (relu x), (sigmoid x), (tanh x), etc.
                        "relu" | "sigmoid" | "tanh" | "sqrt" | "exp" | "log"
                            if elems.len() == 2 =>
                        {
                            let (operand_reg, operand_ty) = self.compile_expr(&elems[1]);
                            let result_reg = self.emit_unary(op, &operand_reg, &operand_ty);
                            (result_reg, operand_ty)
                        }

                        // zeros: (zeros [M N])
                        "zeros" if elems.len() == 2 => {
                            let shape = self.parse_shape_vector(&elems[1]);
                            self.emit_zeros(&shape)
                        }

                        // random-normal: (random-normal key [M N])
                        "random-normal" if elems.len() == 3 => {
                            // Ignore key for now (placeholder implementation)
                            let shape = self.parse_shape_vector(&elems[2]);
                            self.emit_random_normal(&shape)
                        }

                        _ => panic!("Unsupported operation: {}", op),
                    }
                } else {
                    panic!("First element of list must be a symbol: {}", expr)
                }
            }

            _ => panic!("Unsupported expression: {}", expr),
        }
    }

    /// Parse a shape vector like [2 8] into vec![2, 8]
    fn parse_shape_vector(&self, expr: &SheafValue) -> Vec<i64> {
        if let SheafValue::Vector(elems, _) = expr {
            elems
                .iter()
                .map(|e| match e {
                    SheafValue::Integer(n, _) => *n,
                    _ => panic!("Shape element must be integer"),
                })
                .collect()
        } else {
            panic!("Shape must be a vector")
        }
    }

    /// Sanitize function name for MLIR (replace dashes with underscores)
    fn sanitize_func_name(name: &str) -> String {
        name.replace('-', "_")
    }

    /// Generate a complete MLIR module with a function body already emitted
    pub fn emit_function_body(&self, name: &str, result_ty: &StableHLOType) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf Rust compiler").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();
        writeln!(
            output,
            "  func.func @{}() -> {} {{",
            sanitized_name,
            result_ty.to_mlir()
        )
        .unwrap();

        for line in &self.body {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        writeln!(output, "}}").unwrap();

        output
    }

    /// Generate a complete MLIR module with a function
    pub fn emit_function(&mut self, name: &str, expr: &SheafValue) -> String {
        let (result_reg, result_ty) = self.compile_expr(expr);
        self.emit_return(&result_reg, &result_ty);

        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf Rust compiler").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output, "// Source: {}", expr).unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();
        writeln!(
            output,
            "  func.func @{}() -> {} {{",
            sanitized_name,
            result_ty.to_mlir()
        )
        .unwrap();

        for line in &self.body {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        writeln!(output, "}}").unwrap();

        output
    }

    /// Generate a complete MLIR module with multiple function declarations
    ///
    /// Takes a vector of function declaration strings (from emit_func_declaration)
    /// and wraps them in a module { ... } block
    pub fn emit_module(func_declarations: &[String]) -> String {
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf Rust compiler").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();

        for func_decl in func_declarations {
            write!(output, "{}", func_decl).unwrap();
        }

        writeln!(output, "}}").unwrap();
        output
    }
}

impl Default for StableHLOEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SourceLocation;

    fn make_int(n: i64) -> SheafValue {
        SheafValue::Integer(n, SourceLocation::unknown())
    }

    fn make_float(x: f64) -> SheafValue {
        SheafValue::Float(x, SourceLocation::unknown())
    }

    fn make_symbol(s: &str) -> SheafValue {
        SheafValue::Symbol(s.to_string(), SourceLocation::unknown())
    }

    fn make_list(elems: Vec<SheafValue>) -> SheafValue {
        SheafValue::List(elems, SourceLocation::unknown())
    }

    #[test]
    fn test_emit_constant() {
        let mut emitter = StableHLOEmitter::new();
        let reg = emitter.emit_constant_f32(42.0);
        assert_eq!(reg.to_mlir(), "%0");
        assert_eq!(emitter.body.len(), 1);
        assert!(emitter.body[0].contains("dense<42.0>"));
    }

    #[test]
    fn test_emit_add() {
        let mut emitter = StableHLOEmitter::new();
        // (+ 1 2)
        let expr = make_list(vec![make_symbol("+"), make_int(1), make_int(2)]);
        let mlir = emitter.emit_function("add", &expr);

        assert!(mlir.contains("stablehlo.constant"));
        assert!(mlir.contains("stablehlo.add"));
        assert!(mlir.contains("@add"));
    }

    #[test]
    fn test_emit_nested() {
        let mut emitter = StableHLOEmitter::new();
        // (* (+ 1 2) 4)
        let expr = make_list(vec![
            make_symbol("*"),
            make_list(vec![make_symbol("+"), make_int(1), make_int(2)]),
            make_int(4),
        ]);
        let mlir = emitter.emit_function("nested", &expr);

        assert!(mlir.contains("stablehlo.add"));
        assert!(mlir.contains("stablehlo.multiply"));
        assert!(mlir.contains("@nested"));
    }

    #[test]
    fn test_emit_float() {
        let mut emitter = StableHLOEmitter::new();
        let expr = make_float(3.14);
        let mlir = emitter.emit_function("pi", &expr);

        assert!(mlir.contains("dense<3.14>"));
    }
}
