// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf compiler CLI - Phase 1
//!
//! Compiles Sheaf AST (from JSON) to StableHLO MLIR
//!
//! Usage:
//!   sheaf-compile input.json -o output.mlir
//!   sheaf-compile input.json -o output.vmfb --iree

use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, exit};

use sheaf_compiler::{CodeGenerator, CompilerContext, SheafValue, StableHLOEmitter, parse};

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

    for expr in &exprs {
        if let Err(e) = compiler.compile(expr) {
            eprintln!("Compilation error: {}", e);
            exit(1);
        }
    }

    // Generate StableHLO for the target function
    if cli.verbose {
        println!("Looking for function '{}'...", cli.function);
    }

    // Generate StableHLO
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
        let func_decl = codegen
            .emit_func_declaration(&cli.function, &body, &sig.param_types, &sig.return_type)
            .unwrap_or_else(|e| {
                eprintln!("Code generation error: {}", e);
                exit(1);
            });
        StableHLOEmitter::emit_module(&[func_decl])
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
        // If IREE compilation requested, write to temp .mlir first
        let temp_mlir = cli.output.with_extension("mlir");
        temp_mlir
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
