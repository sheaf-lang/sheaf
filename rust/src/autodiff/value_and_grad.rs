// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

/// value_and_grad: generate a single MLIR function that returns (loss, grad_p1, grad_p2, ...).
///
/// The generated function shares a single SSA counter across the forward and
/// all backward passes, so register names never collide.
use crate::autodiff::grad_simplified;
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::stablehlo::{Register, StableHLOEmitter, StableHLOType};
use crate::core::compiler::{CompiledExpr, FunctionDef};
use crate::core::error::SheafResult;
use std::collections::HashMap;

// Descriptor for one differentiable parameter.
pub struct GradParam {
    /// Name of the function parameter (matches the `CompiledExpr::Symbol` in the body)
    pub name: String,
    /// StableHLO type of the gradient output (same shape as the parameter)
    pub ty: StableHLOType,
}

/// Generate a complete MLIR module containing a value-and-grad function.
///
/// # Arguments
/// - `fn_name`     : name of the generated function (e.g. `"train_step"`)
/// - `param_names` : ordered list of all function parameter names
/// - `param_types` : StableHLO type for each parameter (same order)
/// - `loss_expr`   : compiled expression that computes the scalar loss
/// - `wrt`         : which parameters to differentiate with respect to
/// - `registry`    : function registry (for inlining user-defined helpers)
///
/// # Returns
/// A complete MLIR module string ready for `iree-compile`.
pub fn emit_value_and_grad_module(
    fn_name: &str,
    param_names: &[String],
    param_types: &[StableHLOType],
    loss_expr: &CompiledExpr,
    wrt: &[GradParam],
    registry: HashMap<String, FunctionDef>,
) -> SheafResult<String> {
    let func_decl =
        emit_value_and_grad_func(fn_name, param_names, param_types, loss_expr, wrt, registry)?;
    Ok(StableHLOEmitter::emit_module(&[func_decl]))
}

/// Generate a single `func.func` declaration for value-and-grad.
///
/// The function body computes:
///   1. Forward pass: `loss = loss_expr(...)`
///   2. Backward passes: `grad_p = grad(loss_expr, p)` for each p in `wrt`
///   3. Returns `(loss, grad_p1, grad_p2, ...)`
pub fn emit_value_and_grad_func(
    fn_name: &str,
    param_names: &[String],
    param_types: &[StableHLOType],
    loss_expr: &CompiledExpr,
    wrt: &[GradParam],
    registry: HashMap<String, FunctionDef>,
) -> SheafResult<String> {
    // One shared CodeGenerator — SSA counter is shared across forward + all backward passes.
    let mut codegen = CodeGenerator::with_function_params(registry, param_names, param_types);

    // Forward pass
    let (loss_reg, loss_ty) = codegen.generate(loss_expr)?;

    // Backward passes
    let mut grad_regs: Vec<Register> = Vec::new();
    let mut grad_types: Vec<StableHLOType> = Vec::new();
    for param in wrt {
        let grad_expr = grad_simplified(loss_expr, &param.name);
        let (grad_reg, grad_ty) = codegen.generate(&grad_expr)?;
        grad_regs.push(grad_reg);
        grad_types.push(grad_ty);
    }

    // Collect all return values: (loss, grad1, grad2, ...)
    let mut all_regs = vec![loss_reg];
    all_regs.extend(grad_regs);
    let mut all_types = vec![loss_ty];
    all_types.extend(grad_types);

    let decl = codegen.finish_multi(fn_name, param_types, &all_regs, &all_types);
    Ok(decl)
}
