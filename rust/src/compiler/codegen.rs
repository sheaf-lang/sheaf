// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Code generation - translate CompiledExpr to StableHLO MLIR

use crate::compiler::stablehlo::{Register, StableHLOEmitter, StableHLOType};
use crate::core::compiler::{CompiledExpr, CompilerContext};
use crate::core::error::{SheafError, SheafResult};
use std::collections::HashMap;

/// Code generator - converts CompiledExpr to StableHLO
pub struct CodeGenerator {
    emitter: StableHLOEmitter,
    /// Map from variable names to registers
    bindings: HashMap<String, Register>,
}

impl CodeGenerator {
    pub fn new() -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
        }
    }

    /// Generate StableHLO for a compiled expression
    pub fn generate(&mut self, expr: &CompiledExpr) -> SheafResult<(Register, StableHLOType)> {
        match expr {
            CompiledExpr::Integer(n) => {
                // Treat integers as floats for now (matches Python behavior)
                let reg = self.emitter.emit_constant_f32(*n as f64);
                Ok((reg, StableHLOType::scalar_f32()))
            }

            CompiledExpr::Float(x) => {
                let reg = self.emitter.emit_constant_f32(*x);
                Ok((reg, StableHLOType::scalar_f32()))
            }

            CompiledExpr::FunctionCall { name, args } => {
                self.generate_function_call(name, args)
            }

            CompiledExpr::FunctionRef(name) => {
                Err(SheafError::Compile {
                    message: format!("Cannot generate code for bare function reference: {}", name),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }

            _ => Err(SheafError::Compile {
                message: format!("Code generation not yet implemented for: {:?}", expr),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }
    }

    /// Generate code for a function call
    fn generate_function_call(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        // For now, only support binary arithmetic operations
        if matches!(name, "+" | "-" | "*" | "/") && args.len() == 2 {
            let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
            let (rhs_reg, _rhs_ty) = self.generate(&args[1])?;
            let result_reg = self.emitter.emit_binop(name, &lhs_reg, &rhs_reg, &lhs_ty);
            Ok((result_reg, lhs_ty))
        } else {
            Err(SheafError::Compile {
                message: format!("Function call not yet supported: {}", name),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    /// Emit a complete function module
    pub fn emit_function(mut self, name: &str, expr: &CompiledExpr) -> SheafResult<String> {
        let (result_reg, result_ty) = self.generate(expr)?;
        self.emitter.emit_return(&result_reg, &result_ty);

        Ok(self.emitter.emit_function_body(name, &result_ty))
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_constant() {
        let mut codegen = CodeGenerator::new();
        let expr = CompiledExpr::Integer(42);
        let result = codegen.generate(&expr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_binop() {
        let mut codegen = CodeGenerator::new();
        let expr = CompiledExpr::FunctionCall {
            name: "+".to_string(),
            args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(2)],
        };
        let result = codegen.generate(&expr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_emit_function() {
        let codegen = CodeGenerator::new();
        let expr = CompiledExpr::FunctionCall {
            name: "+".to_string(),
            args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(2)],
        };
        let mlir = codegen.emit_function("test", &expr);
        assert!(mlir.is_ok());
        let mlir_str = mlir.unwrap();
        assert!(mlir_str.contains("stablehlo.add"));
        assert!(mlir_str.contains("@test"));
    }
}
