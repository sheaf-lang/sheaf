// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! StableHLO MLIR types, registers, and expression emitter.

mod broadcast;
mod compare_ops;
mod constants;
mod dot_ops;
mod emit;
mod indexing_ops;
mod operations;
mod random_ops;
mod reduce_ops;
mod shape_ops;
mod tensor_ops;
mod types;

use crate::core::ast::SheafValue;
use std::collections::HashMap;

pub use types::{Register, StableHLOType};

/// Stateful StableHLO instruction emitter.
pub struct StableHLOEmitter {
    counter: usize,
    pub(crate) body: Vec<String>,
    /// Cached mean reductions reused while emitting variance operations.
    reduce_mean_cache: HashMap<(Register, usize, bool), (Register, StableHLOType)>,
    /// Tuples remain register groups and are never materialized as MLIR operations.
    virtual_tuples: HashMap<Register, Vec<(Register, StableHLOType)>>,
    /// Constants used to resolve shape operands during code generation.
    known_scalars: HashMap<Register, f64>,
    known_tensors: HashMap<Register, Vec<f64>>,
}

impl StableHLOEmitter {
    pub fn new() -> Self {
        Self {
            counter: 0,
            body: Vec::new(),
            reduce_mean_cache: HashMap::new(),
            virtual_tuples: HashMap::new(),
            known_scalars: HashMap::new(),
            known_tensors: HashMap::new(),
        }
    }

    pub fn set_known_scalar(&mut self, reg: Register, value: f64) {
        self.known_scalars.insert(reg, value);
    }

    pub fn known_scalar_value(&self, reg: &Register) -> Option<f64> {
        self.known_scalars.get(reg).copied()
    }

    pub fn set_known_tensor(&mut self, reg: Register, values: Vec<f64>) {
        self.known_tensors.insert(reg, values);
    }

    pub fn known_tensor_values(&self, reg: &Register) -> Option<&Vec<f64>> {
        self.known_tensors.get(reg)
    }

    pub fn fresh_register(&mut self) -> Register {
        let reg = Register::new(self.counter);
        self.counter += 1;
        reg
    }

    pub fn emit_instruction(&mut self, instruction: String) {
        self.body.push(instruction);
    }

    /// Maps tuple leaves to flat arguments and nested tuples to virtual registers.
    pub fn register_virtual_param(&mut self, ty: &StableHLOType, flat_idx: &mut usize) -> Register {
        match ty {
            StableHLOType::Tuple(elems, _) => {
                let vreg = self.fresh_register();
                let constituents: Vec<_> = elems.iter()
                    .map(|elem_ty| {
                        let sub_reg = self.register_virtual_param(elem_ty, flat_idx);
                        (sub_reg, elem_ty.clone())
                    })
                    .collect();
                self.virtual_tuples.insert(vreg, constituents);
                vreg
            }
            _ => {
                let reg = Register::arg(*flat_idx);
                *flat_idx += 1;
                reg
            }
        }
    }

    /// Flattens a virtual tuple into its leaf registers.
    pub fn virtual_tuple_elements(&self, reg: &Register) -> Option<&[(Register, StableHLOType)]> {
        self.virtual_tuples.get(reg).map(Vec::as_slice)
    }

    pub(crate) fn replace_virtual_leaf(
        &mut self, mut root: Register, indices: &[usize], reg: Register, ty: StableHLOType,
    ) -> bool {
        let Some((&last, parents)) = indices.split_last() else { return false };
        for &index in parents {
            let Some((child, _)) = self.virtual_tuples.get(&root)
                .and_then(|elements| elements.get(index)) else { return false };
            root = *child;
        }
        let Some(element) = self.virtual_tuples.get_mut(&root)
            .and_then(|elements| elements.get_mut(last)) else { return false };
        *element = (reg, ty);
        true
    }

    pub fn collect_virtual_leaves(&self, reg: Register, ty: &StableHLOType) -> Vec<(Register, StableHLOType)> {
        if let Some(constituents) = self.virtual_tuples.get(&reg) {
            constituents.iter()
                .flat_map(|(r, t)| self.collect_virtual_leaves(*r, t))
                .collect()
        } else {
            vec![(reg, ty.clone())]
        }
    }

    pub fn compile_expr(&mut self, expr: &SheafValue) -> (Register, StableHLOType) {
        match expr {
            SheafValue::Float(x, _) => {
                let reg = self.emit_constant_f32(*x);
                (reg, StableHLOType::scalar_f32())
            }
            SheafValue::Integer(n, _) => {
                let reg = self.emit_constant_f32(*n as f64);
                (reg, StableHLOType::scalar_f32())
            }

            SheafValue::Vector(elems, _) => {
                if !elems.is_empty() && matches!(elems[0], SheafValue::Vector(_, _)) {
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
                    let values: Vec<f64> = elems
                        .iter()
                        .map(|e| match e {
                            SheafValue::Float(x, _) => *x,
                            SheafValue::Integer(n, _) => *n as f64,
                            _ => panic!("Vector element must be number"),
                        })
                        .collect();
                    self.emit_tensor_constant(&[values])
                }
            }

            SheafValue::List(elems, _) if !elems.is_empty() => {
                if let Some(op) = elems[0].as_symbol() {
                    match op {
                        "+" | "-" | "*" | "/" if elems.len() == 3 => {
                            let (lhs_reg, lhs_ty) = self.compile_expr(&elems[1]);
                            let (rhs_reg, rhs_ty) = self.compile_expr(&elems[2]);
                            self.emit_binop(op, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty)
                        }

                        "@" if elems.len() == 3 => {
                            let (lhs_reg, lhs_ty) = self.compile_expr(&elems[1]);
                            let (rhs_reg, rhs_ty) = self.compile_expr(&elems[2]);
                            self.emit_matmul(&lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty)
                        }

                        "relu" | "sigmoid" | "tanh" | "sqrt" | "exp" | "log"
                            if elems.len() == 2 =>
                        {
                            let (operand_reg, operand_ty) = self.compile_expr(&elems[1]);
                            let result_reg = self.emit_unary(op, &operand_reg, &operand_ty);
                            (result_reg, operand_ty)
                        }

                        "zeros" if elems.len() == 2 => {
                            let shape = self.parse_shape_vector(&elems[1]);
                            self.emit_zeros(&shape)
                        }

                        "random-normal" if elems.len() == 3 => {
                            let (key_reg, key_ty) = self.compile_expr(&elems[1]);
                            let shape = self.parse_shape_vector(&elems[2]);
                            self.emit_random_normal(&key_reg, &key_ty, &shape)
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
        let expr = make_list(vec![make_symbol("+"), make_int(1), make_int(2)]);
        let mlir = emitter.emit_function("add", &expr);

        assert!(mlir.contains("stablehlo.constant"));
        assert!(mlir.contains("stablehlo.add"));
        assert!(mlir.contains("@add"));
    }

    #[test]
    fn test_emit_nested() {
        let mut emitter = StableHLOEmitter::new();
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
        let expr = make_float(3.125);
        let mlir = emitter.emit_function("float", &expr);

        assert!(mlir.contains("dense<3.125>"));
    }
}
