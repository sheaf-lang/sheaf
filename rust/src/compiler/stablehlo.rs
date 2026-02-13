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
    /// Tensor with shape: tensor<2x3xf32>
    Tensor { shape: Vec<i64>, dtype: String },
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

    /// Get the shape of this type, or empty vec for scalars
    pub fn shape(&self) -> Vec<i64> {
        match self {
            Self::ScalarF32 | Self::ScalarF64 | Self::ScalarI64 => vec![],
            Self::Tensor { shape, .. } => shape.clone(),
        }
    }

    /// Get the dtype string
    pub fn dtype(&self) -> &str {
        match self {
            Self::ScalarF32 => "f32",
            Self::ScalarF64 => "f64",
            Self::ScalarI64 => "i64",
            Self::Tensor { dtype, .. } => dtype,
        }
    }

    pub fn to_mlir(&self) -> String {
        match self {
            Self::ScalarF32 => "tensor<f32>".to_string(),
            Self::ScalarF64 => "tensor<f64>".to_string(),
            Self::ScalarI64 => "tensor<i64>".to_string(),
            Self::Tensor { shape, dtype } => {
                let shape_str = shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x");
                format!("tensor<{}x{}>", shape_str, dtype)
            }
        }
    }
}

/// Register name in SSA form: %0, %1, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct Register(usize);

impl Register {
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    pub fn to_mlir(&self) -> String {
        format!("%{}", self.0)
    }
}

/// MLIR StableHLO emitter
pub struct StableHLOEmitter {
    counter: usize,
    body: Vec<String>,
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
            "    {} = \"stablehlo.constant\"() {{value = dense<{}> : {}}} : () -> {}",
            reg.to_mlir(),
            value_str,
            ty.to_mlir(),
            ty.to_mlir()
        ));
        reg
    }

    /// Emit a constant integer
    pub fn emit_constant_i64(&mut self, value: i64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::ScalarI64;
        self.body.push(format!(
            "    {} = \"stablehlo.constant\"() {{value = dense<{}> : {}}} : () -> {}",
            reg.to_mlir(),
            value,
            ty.to_mlir(),
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
            "    {} = \"stablehlo.constant\"() {{value = dense<[{}]> : {}}} : () -> {}",
            reg.to_mlir(),
            values_str,
            ty.to_mlir(),
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit a binary operation
    pub fn emit_binop(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        ty: &StableHLOType,
    ) -> Register {
        let stablehlo_op = match op {
            "+" => "stablehlo.add",
            "-" => "stablehlo.subtract",
            "*" => "stablehlo.multiply",
            "/" => "stablehlo.divide",
            _ => panic!("Unsupported binop: {}", op),
        };

        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = \"{}\"({}, {}) : ({}, {}) -> {}",
            reg.to_mlir(),
            stablehlo_op,
            lhs.to_mlir(),
            rhs.to_mlir(),
            ty.to_mlir(),
            ty.to_mlir(),
            ty.to_mlir()
        ));
        reg
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
            "    {} = \"stablehlo.dot_general\"({}, {}) {{
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >
    }} : ({}, {}) -> {}",
            reg.to_mlir(),
            lhs.to_mlir(),
            rhs.to_mlir(),
            lhs_ty.to_mlir(),
            rhs_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit a return statement
    pub fn emit_return(&mut self, reg: &Register, ty: &StableHLOType) {
        self.body
            .push(format!("    return {} : {}", reg.to_mlir(), ty.to_mlir()));
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

            // Binary operations: (+ a b), (- a b), (* a b), (/ a b)
            SheafValue::List(elems, _) if elems.len() == 3 => {
                if let Some(op) = elems[0].as_symbol() {
                    if matches!(op, "+" | "-" | "*" | "/") {
                        let (lhs_reg, lhs_ty) = self.compile_expr(&elems[1]);
                        let (rhs_reg, _rhs_ty) = self.compile_expr(&elems[2]);
                        let result_reg = self.emit_binop(op, &lhs_reg, &rhs_reg, &lhs_ty);
                        return (result_reg, lhs_ty);
                    }
                }
                panic!("Unsupported list form: {}", expr)
            }

            _ => panic!("Unsupported expression: {}", expr),
        }
    }

    /// Generate a complete MLIR module with a function body already emitted
    pub fn emit_function_body(&self, name: &str, result_ty: &StableHLOType) -> String {
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf Rust compiler").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();
        writeln!(
            output,
            "  func.func @{}() -> {} {{",
            name,
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

        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf Rust compiler").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output, "// Source: {}", expr).unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();
        writeln!(
            output,
            "  func.func @{}() -> {} {{",
            name,
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
