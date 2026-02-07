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

use sheaf_compiler::{CodeGenerator, CompilerContext, SheafValue, parse};

/// Find a function definition by name
/// Looks for (defn name [...] body) in the expression list
fn find_function<'a>(exprs: &'a [SheafValue], name: &str) -> Option<&'a SheafValue> {
    for expr in exprs {
        if let Some(list) = expr.as_list() {
            if list.len() >= 3 && list[0].is_symbol("defn") && list[1].is_symbol(name) {
                // Found the function definition
                // Return the body (skip defn, name, params)
                if list.len() > 3 {
                    return Some(&list[3]);
                }
            }
        }
    }
    None
}

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

    // Look for the target function (defn main [...] body)
    // or use the first expression if it's a standalone expr
    if cli.verbose {
        println!("Looking for function '{}'...", cli.function);
    }

    let target_expr = find_function(&exprs, &cli.function).unwrap_or_else(|| {
        if cli.verbose {
            println!(
                "Function '{}' not found, compiling first expression",
                cli.function
            );
        }
        &exprs[0]
    });

    // Compile all expressions (register defn, etc.)
    if cli.verbose {
        println!("Compiling expressions...");
    }

    let mut compiler = CompilerContext::new();

    // Compile all expressions to populate registry
    for expr in &exprs {
        if let Err(e) = compiler.compile(expr) {
            eprintln!("Compilation error: {}", e);
            exit(1);
        }
    }

    // Now compile the target function body
    let compiled = compiler.compile(target_expr).unwrap_or_else(|e| {
        eprintln!("Compilation error: {}", e);
        exit(1);
    });

    // Generate StableHLO
    if cli.verbose {
        println!("Generating StableHLO...");
    }

    let codegen = CodeGenerator::new();
    let mlir = codegen
        .emit_function(&cli.function, &compiled)
        .unwrap_or_else(|e| {
            eprintln!("Code generation error: {}", e);
            exit(1);
        });

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
