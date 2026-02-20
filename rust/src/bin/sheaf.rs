// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf compiler CLI
//!
//! Compiles Sheaf source to StableHLO MLIR
//!
//! Usage:
//!   sheaf input.shf -o output.mlir
//!   sheaf input.shf -o output.vmfb --iree

use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, exit};

use sheaf_compiler::autodiff::value_and_grad::{GradParam, emit_value_and_grad_func};
use sheaf_compiler::core::compiler::CompiledExpr;
use sheaf_compiler::core::inference::infer_function_signature_with_known;
use sheaf_compiler::{CodeGenerator, CompilerContext, StableHLOEmitter, StableHLOType, parse};

#[derive(Parser)]
#[command(name = "sheaf")]
#[command(about = "Sheaf - A Functional Language for Differentiable Computation", long_about = None)]
struct Cli {
    /// Input file (.shf or .json for AST)
    input: PathBuf,

    /// Output file (.mlir or .vmfb)
    #[arg(short, long)]
    output: PathBuf,

    /// Compile to VMFB using IREE (requires iree-compile in PATH or IREE_COMPILE env)
    #[arg(long)]
    iree: bool,

    /// IREE target backend (default: llvm-cpu)
    #[arg(long, default_value = "llvm-cpu")]
    iree_backend: String,

    /// Function name to compile (default: main)
    #[arg(long, default_value = "main")]
    function: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Resolve `CompiledExpr::ValueAndGrad` nodes into MLIR func.func declarations.
/// Walks all compiled expressions (including inside `Do` blocks) and performs
/// the codegen that was previously done inline by `ValueAndGradForm`.
fn resolve_value_and_grad_decls(
    compiler: &CompilerContext,
    compiled_exprs: &[CompiledExpr],
) -> Vec<String> {
    let mut vag_nodes = Vec::new();
    for expr in compiled_exprs {
        collect_vag_nodes(expr, &mut vag_nodes);
    }

    let mut decls = Vec::new();
    for (fn_name, src_fn_name, wrt_params, shape_config) in vag_nodes {
        let func_def = match compiler.registry.get(src_fn_name) {
            Some(fd) => fd,
            None => continue,
        };

        let body_compiled = match &func_def.body_compiled {
            Some(b) => b,
            None => continue,
        };

        let known_types: Vec<(String, StableHLOType)> = shape_config
            .iter()
            .map(|(name, dims)| {
                let ty = if dims.is_empty() {
                    StableHLOType::scalar_f32()
                } else {
                    StableHLOType::f32_tensor(dims.clone())
                };
                (name.clone(), ty)
            })
            .collect();

        let signature = if !known_types.is_empty() {
            match infer_function_signature_with_known(
                compiler,
                &func_def.params,
                body_compiled,
                &known_types,
            ) {
                Ok(sig) => sig,
                Err(e) => {
                    eprintln!("value-and-grad '{}': signature inference failed: {}", fn_name, e);
                    exit(1);
                }
            }
        } else {
            match &func_def.signature {
                Some(sig) => sig.clone(),
                None => {
                    eprintln!("value-and-grad '{}': function '{}' has no inferred signature", fn_name, src_fn_name);
                    exit(1);
                }
            }
        };

        let grad_params: Vec<GradParam> = wrt_params
            .iter()
            .map(|wrt_name| {
                let idx = func_def
                    .params
                    .iter()
                    .position(|p| p == wrt_name)
                    .unwrap_or_else(|| {
                        eprintln!(
                            "value-and-grad '{}': '{}' is not a parameter of '{}'",
                            fn_name, wrt_name, src_fn_name
                        );
                        exit(1);
                    });
                GradParam {
                    name: wrt_name.clone(),
                    ty: signature.param_types[idx].clone(),
                }
            })
            .collect();

        let func_decl = emit_value_and_grad_func(
            fn_name,
            &func_def.params,
            &signature.param_types,
            body_compiled,
            &grad_params,
            compiler.registry.clone(),
        )
        .unwrap_or_else(|e| {
            eprintln!("value-and-grad '{}': code generation failed: {}", fn_name, e);
            exit(1);
        });

        decls.push(func_decl);
    }
    decls
}

/// Recursively collect ValueAndGrad nodes from compiled expression trees.
fn collect_vag_nodes<'a>(
    expr: &'a CompiledExpr,
    out: &mut Vec<(&'a str, &'a str, &'a Vec<String>, &'a Vec<(String, Vec<i64>)>)>,
) {
    match expr {
        CompiledExpr::ValueAndGrad {
            fn_name,
            src_fn_name,
            wrt_params,
            shape_config,
        } => {
            out.push((fn_name, src_fn_name, wrt_params, shape_config));
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                collect_vag_nodes(e, out);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                collect_vag_nodes(v, out);
            }
            collect_vag_nodes(body, out);
        }
        _ => {}
    }
}

fn main() {
    let cli = Cli::parse();

    // Read input
    let source = fs::read_to_string(&cli.input).unwrap_or_else(|e| {
        eprintln!("Error reading input file: {}", e);
        exit(1);
    });

    if cli.verbose {
        println!("Parsing {}...", cli.input.display());
    }

    let exprs = parse(&source, cli.input.to_str().unwrap()).unwrap_or_else(|e| {
        eprintln!("Parse error: {}", e);
        exit(1);
    });

    if exprs.is_empty() {
        eprintln!("Error: No expressions found in input");
        exit(1);
    }

    // Compile all expressions (register defn, defparams, etc.)
    if cli.verbose {
        println!("Compiling expressions...");
    }

    let mut compiler = CompilerContext::new();
    let mut compiled_exprs = Vec::new();

    for expr in &exprs {
        match compiler.compile(expr) {
            Ok(compiled) => compiled_exprs.push(compiled),
            Err(e) => {
                eprintln!("Compilation error: {}", e);
                exit(1);
            }
        }
    }

    // Resolve value-and-grad nodes into MLIR declarations
    let extra_decls = resolve_value_and_grad_decls(&compiler, &compiled_exprs);

    // Generate StableHLO for the target function
    if cli.verbose {
        println!("Looking for function '{}'...", cli.function);
    }

    if cli.verbose {
        println!("Generating StableHLO...");
    }

    let mlir = if let Some(func_def) = compiler.registry.get(&cli.function).cloned() {
        // Function found in registry - use its compiled body and signature
        let body = func_def.body_compiled.unwrap_or_else(|| {
            eprintln!("Error: function '{}' has no compiled body", cli.function);
            exit(1);
        });
        let sig = func_def.signature.unwrap_or_else(|| {
            eprintln!(
                "Error: function '{}' has no inferred signature",
                cli.function
            );
            exit(1);
        });

        let codegen = CodeGenerator::with_function_params(
            compiler.registry.clone(),
            &func_def.params,
            &sig.param_types,
        );
        let main_decl = codegen
            .emit_func_declaration(&cli.function, &body, &sig.param_types, &sig.return_type)
            .unwrap_or_else(|e| {
                eprintln!("Code generation error: {}", e);
                exit(1);
            });

        let mut all_decls = extra_decls;
        all_decls.push(main_decl);
        StableHLOEmitter::emit_module(&all_decls)
    } else if !extra_decls.is_empty() {
        // Function was emitted by a module-level form (e.g. value-and-grad)
        if cli.verbose {
            println!(
                "Function '{}' found in extra declarations (emitted by module-level form)",
                cli.function
            );
        }
        StableHLOEmitter::emit_module(&extra_decls)
    } else {
        // No function found — compile first non-defn expression as a standalone main
        if cli.verbose {
            println!(
                "Function '{}' not found in registry, compiling first expression",
                cli.function
            );
        }
        let standalone = exprs
            .iter()
            .find(|e| {
                e.as_list()
                    .and_then(|l| l.first())
                    .and_then(|h| h.as_symbol())
                    .map(|s| s != "defn" && s != "defparams")
                    .unwrap_or(true)
            })
            .unwrap_or_else(|| {
                eprintln!("Error: no expressions to compile in '{}'", cli.function);
                exit(1);
            });

        let compiled = compiler.compile(standalone).unwrap_or_else(|e| {
            eprintln!("Compilation error: {}", e);
            exit(1);
        });

        CodeGenerator::with_registry(compiler.registry.clone())
            .emit_function(&cli.function, &compiled)
            .unwrap_or_else(|e| {
                eprintln!("Code generation error: {}", e);
                exit(1);
            })
    };

    // Determine output path
    let mlir_path = if cli.iree {
        cli.output.with_extension("mlir")
    } else {
        cli.output.clone()
    };

    // Write MLIR
    fs::write(&mlir_path, &mlir).unwrap_or_else(|e| {
        eprintln!("Error writing MLIR: {}", e);
        exit(1);
    });

    if cli.verbose {
        println!("Wrote MLIR to {}", mlir_path.display());
    }

    // If IREE compilation requested
    if cli.iree {
        if cli.verbose {
            println!("Compiling with IREE...");
        }

        let iree_compile = std::env::var("IREE_COMPILE").unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|home| format!("{}/bin/iree-build/tools/iree-compile", home))
                .unwrap_or_else(|_| "iree-compile".to_string())
        });

        let status = Command::new(&iree_compile)
            .arg(&mlir_path)
            .arg(format!("--iree-hal-target-backends={}", cli.iree_backend))
            .arg("-o")
            .arg(&cli.output)
            .status()
            .unwrap_or_else(|e| {
                eprintln!("Error running iree-compile: {}", e);
                eprintln!("Make sure iree-compile is in PATH or set IREE_COMPILE env variable");
                exit(1);
            });

        if !status.success() {
            eprintln!("IREE compilation failed");
            exit(1);
        }

        if cli.verbose {
            println!("Compiled to {}", cli.output.display());
        }

        // Clean up temp .mlir if different from output
        if mlir_path != cli.output {
            let _ = fs::remove_file(&mlir_path);
        }
    }

    if cli.verbose {
        println!("Done!");
    }
}
