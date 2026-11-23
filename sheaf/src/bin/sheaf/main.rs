// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf command-line entry point.

mod doc;
mod pretty;
mod repl;

use sheaf_compiler::sheaf_msg;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tail = &args[1..];

    // Help and version take precedence over all other arguments.
    if tail.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    if tail.iter().any(|a| a == "--version") {
        println!("Sheaf {}", sheaf_compiler::SHEAF_VERSION);
        return;
    }

    let mut verbosity: u8 = 0;
    let mut device: Option<String> = None;
    let mut jit_profile = false;
    let mut jit_strict = false;
    let mut blame = false;
    let mut trace_enabled = false;
    let mut trace_scope: Option<Vec<String>> = None;
    let mut guard_specs: Vec<String> = Vec::new();
    let mut mem_profile = false;
    let mut remaining: Vec<String> = Vec::new();
    let mut i = 0;
    while i < tail.len() {
        match tail[i].as_str() {
            "-v" | "--verbose" => verbosity += 1,
            "-vv" => verbosity = 2,
            "--device" => {
                i += 1;
                match tail.get(i) {
                    Some(d) => device = Some(d.clone()),
                    None => {
                        sheaf_msg!("sheaf: --device requires an argument");
                        exit(1);
                    }
                }
            }
            "--jit-profile" => jit_profile = true,
            "--jit-strict" => jit_strict = true,
            "--blame" => blame = true,
            "--mem-profile" => mem_profile = true,
            "--trace" => {
                trace_enabled = true;
                if tail.get(i + 1).map(|a| !a.starts_with('-')).unwrap_or(false) {
                    i += 1;
                    trace_scope = Some(tail[i].split(',').map(|s| s.to_string()).collect());
                }
            }
            "--guard" => {
                i += 1;
                match tail.get(i) {
                    Some(spec) => guard_specs.push(spec.clone()),
                    None => {
                        sheaf_msg!("sheaf: --guard requires a SPEC argument");
                        exit(1);
                    }
                }
            }
            other => remaining.push(other.to_string()),
        }
        i += 1;
    }
    if jit_profile && trace_enabled {
        sheaf_msg!("sheaf: --jit-profile and --trace are mutually exclusive");
        exit(1);
    }

    sheaf_compiler::core::config::init_with_jit_strict(verbosity, device, jit_profile, jit_strict);

    let first_positional = remaining.iter().position(|a| !a.starts_with('-'));

    let cli_guards: Vec<_> = guard_specs.iter().map(|spec| {
        match parse_guard_spec(spec) {
            Ok(guard) => guard,
            Err(msg) => {
                sheaf_msg!("sheaf: invalid guard spec '{}': {}", spec, msg);
                exit(1);
            }
        }
    }).collect();

    match first_positional.map(|i| remaining[i].as_str()) {
        Some("init-ai") => run_init_ai(),

        Some(file) if file.ends_with(".shf") || std::path::Path::new(file).exists() => {
            let pos = first_positional.unwrap();
            run_file_v2(
                &remaining[pos],
                blame,
                trace_enabled,
                trace_scope,
                cli_guards,
                mem_profile,
            );
        }

        _ => {
            if let Some(ci) = remaining.iter().position(|a| a == "-c") {
                let expr = remaining[ci + 1..].iter()
                    .filter(|a| !a.starts_with('-') || a.parse::<f64>().is_ok())
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ");
                if expr.is_empty() {
                    sheaf_msg!("sheaf: -c requires an expression");
                    exit(1);
                }
                run_expr_v2(&expr, blame, trace_enabled, trace_scope, cli_guards, mem_profile);
            } else if remaining.is_empty() {
                run_repl();
            } else {
                sheaf_msg!("sheaf: unknown command '{}'", remaining[0]);
                eprintln!("Run 'sheaf --help' for usage.");
                exit(1);
            }
        }
    }

    sheaf_compiler::runtime::report_jit_profile();
}

fn print_help() {
    println!(
        "Sheaf {} - A Functional Language for Differentiable Computation

Usage:
    sheaf                              Launch interactive REPL
    sheaf [OPTIONS] FILE               Run a Sheaf file
    sheaf -c EXPR                      Evaluate an expression
    sheaf init-ai                      Generate context file for AI assistants

Options:
    --trace [FUNCTIONS]    Log calls with tensor shapes and timing
    --trace-level LEVEL    Trace verbosity: fast, normal (default), verbose
    --trace-out FORMAT     Trace output: console (default), json
    --guard SPEC           Runtime guard: [scope:]variable:check (repeatable)
    --blame                Profile execution and print timing report
    --mem-profile          Memory profiling (RSS at key checkpoints)

Advanced options:
    --device DEVICE        Run on specific device: cpu, metal, cuda, vulkan (default: auto)
    -v / -vv               Verbose JIT output (-vv for full MLIR dumps)
    --jit-profile          Show JIT dispatch timing breakdown
    --jit-strict           Fail on unsupported JIT compilation instead of interpreting

Examples:
    sheaf script.shf
    sheaf -c '(+ 1 2)'
    sheaf --blame train.shf
    sheaf --guard no-nan train.shf
    sheaf --mem-profile gpt2.shf
    sheaf train.shf --guard loss:range:0:20",
        sheaf_compiler::SHEAF_VERSION
    );
}

fn is_silent_result(val: &sheaf_compiler::interpreter::value::Value) -> bool {
    use sheaf_compiler::interpreter::value::Value;
    match val {
        Value::Nil => true,
        Value::List(items) => items.iter().all(|v| matches!(v, Value::Nil)),
        _ => false,
    }
}

fn print_result(result: Result<sheaf_compiler::interpreter::value::Value, sheaf_compiler::core::error::SheafError>) {
    match result {
        Ok(val) => {
            if !is_silent_result(&val) {
                println!("{}", val);
            }
        }
        Err(e) => {
            sheaf_msg!("{}", sheaf_compiler::core::error_format::format_error(&e));
            exit(1);
        }
    }
}

fn run_expr_v2(
    source: &str,
    blame: bool,
    trace_enabled: bool,
    trace_scope: Option<Vec<String>>,
    cli_guards: Vec<sheaf_compiler::interpreter::tracer::CliGuard>,
    mem_profile: bool,
) {
    use sheaf_compiler::interpreter::eval::{eval_source, eval_source_with_blame, eval_source_with_blame_mem, eval_source_with_mem, eval_source_with_tracing};
    use sheaf_compiler::interpreter::tracer::{LogFormat, TraceLevel, TracerConfig};

    let needs_tracing = trace_enabled || !cli_guards.is_empty();
    if blame {
        let tracer_config = if needs_tracing {
            Some(TracerConfig {
                enabled: trace_enabled,
                scope_filter: trace_scope,
                level: TraceLevel::Normal,
                format: LogFormat::Console,
                cli_guards,
            })
        } else {
            None
        };
        if mem_profile {
            let (val, mem_report) = eval_source_with_blame_mem(source, None, tracer_config)
                .unwrap_or_else(|e| {
                    sheaf_msg!("{}", sheaf_compiler::core::error_format::format_error(&e));
                    exit(1);
                });
            if !is_silent_result(&val) {
                println!("{}", val);
            }
            if !mem_report.is_empty() {
                eprintln!("{}", mem_report);
            }
        } else {
            eval_source_with_blame(source, None, tracer_config)
                .unwrap_or_else(|e| {
                    sheaf_msg!("{}", sheaf_compiler::core::error_format::format_error(&e));
                    exit(1);
                });
        }
    } else if mem_profile {
        let (val, mem_report) = eval_source_with_mem(source, None)
            .unwrap_or_else(|e| {
                sheaf_msg!("{}", sheaf_compiler::core::error_format::format_error(&e));
                exit(1);
            });
        if !is_silent_result(&val) {
            println!("{}", val);
        }
        if !mem_report.is_empty() {
            eprintln!("{}", mem_report);
        }
    } else if needs_tracing {
        let config = TracerConfig {
            enabled: trace_enabled,
            scope_filter: trace_scope,
            level: TraceLevel::Normal,
            format: LogFormat::Console,
            cli_guards,
        };
        print_result(eval_source_with_tracing(source, None, config));
    } else {
        print_result(eval_source(source));
    }
}

fn run_file_v2(
    path: &str,
    blame: bool,
    trace_enabled: bool,
    trace_scope: Option<Vec<String>>,
    cli_guards: Vec<sheaf_compiler::interpreter::tracer::CliGuard>,
    mem_profile: bool,
) {
    use std::path::PathBuf;
    use sheaf_compiler::interpreter::eval::{eval_source_with_blame, eval_source_with_blame_mem, eval_source_with_path, eval_source_with_mem, eval_source_with_tracing};
    use sheaf_compiler::interpreter::tracer::{LogFormat, TraceLevel, TracerConfig};

    let abs_path = PathBuf::from(path)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(path));

    let source = std::fs::read_to_string(&abs_path).unwrap_or_else(|e| {
        sheaf_msg!("sheaf: cannot read '{}': {}", path, e);
        exit(1);
    });
    sheaf_compiler::core::error_format::register_source(
        abs_path.to_str().unwrap_or(path),
        &source,
    );

    let needs_tracing = trace_enabled || !cli_guards.is_empty();
    if blame {
        let tracer_config = if needs_tracing {
            Some(TracerConfig {
                enabled: trace_enabled,
                scope_filter: trace_scope,
                level: TraceLevel::Normal,
                format: LogFormat::Console,
                cli_guards,
            })
        } else {
            None
        };
        if mem_profile {
            let (val, mem_report) = eval_source_with_blame_mem(&source, Some(&abs_path), tracer_config)
                .unwrap_or_else(|e| {
                    sheaf_msg!("{}", sheaf_compiler::core::error_format::format_error(&e));
                    exit(1);
                });
            if !is_silent_result(&val) {
                println!("{}", val);
            }
            if !mem_report.is_empty() {
                eprintln!("{}", mem_report);
            }
        } else {
            eval_source_with_blame(&source, Some(&abs_path), tracer_config)
                .unwrap_or_else(|e| {
                    sheaf_msg!("{}", sheaf_compiler::core::error_format::format_error(&e));
                    exit(1);
                });
        }
    } else if mem_profile {
        let (val, mem_report) = eval_source_with_mem(&source, Some(&abs_path))
            .unwrap_or_else(|e| {
                sheaf_msg!("{}", sheaf_compiler::core::error_format::format_error(&e));
                exit(1);
            });
        if !is_silent_result(&val) {
            println!("{}", val);
        }
        if !mem_report.is_empty() {
            eprintln!("{}", mem_report);
        }
    } else if needs_tracing {
        let config = TracerConfig {
            enabled: trace_enabled,
            scope_filter: trace_scope,
            level: TraceLevel::Normal,
            format: LogFormat::Console,
            cli_guards,
        };
        print_result(eval_source_with_tracing(&source, Some(&abs_path), config));
    } else {
        print_result(eval_source_with_path(&source, Some(&abs_path)));
    }
}

/// Parses `[scope:]check[:args]`, for example `loss:no-nan` or `range:0:10`.
fn parse_guard_spec(spec: &str) -> Result<sheaf_compiler::interpreter::tracer::CliGuard, String> {
    use sheaf_compiler::core::expr::GuardCheck;
    use sheaf_compiler::interpreter::tracer::CliGuard;

    let parts: Vec<&str> = spec.split(':').collect();
    match parts.as_slice() {
        ["no-nan"] => Ok(CliGuard { scope: None, check: GuardCheck::NoNan }),
        [scope, "no-nan"] => Ok(CliGuard { scope: Some(scope.to_string()), check: GuardCheck::NoNan }),
        ["range", lo, hi] => {
            let lo: f64 = lo.parse().map_err(|_| format!("invalid lo: {}", lo))?;
            let hi: f64 = hi.parse().map_err(|_| format!("invalid hi: {}", hi))?;
            Ok(CliGuard { scope: None, check: GuardCheck::Range { lo, hi } })
        }
        [scope, "range", lo, hi] => {
            let lo: f64 = lo.parse().map_err(|_| format!("invalid lo: {}", lo))?;
            let hi: f64 = hi.parse().map_err(|_| format!("invalid hi: {}", hi))?;
            Ok(CliGuard { scope: Some(scope.to_string()), check: GuardCheck::Range { lo, hi } })
        }
        _ => Err("expected [scope:]no-nan or [scope:]range:lo:hi".to_string()),
    }
}

fn run_repl() {
    repl::run();
}

fn run_init_ai() {
    const CONTEXT: &str = include_str!("../../../assets/sheaf-context.md");

    let output = std::path::Path::new("sheaf-context.md");
    if output.exists() {
        eprintln!("sheaf-context.md already exists: overwrite? [y/N] ");
        let mut answer = String::new();
        if std::io::stdin().read_line(&mut answer).is_err() || !answer.trim().eq_ignore_ascii_case("y") {
            eprintln!("Aborted.");
            exit(0);
        }
    }

    std::fs::write(output, CONTEXT).unwrap_or_else(|e| {
        sheaf_msg!("sheaf: cannot write '{}': {}", output.display(), e);
        exit(1);
    });

    sheaf_msg!("Wrote sheaf-context.md ({} bytes)", CONTEXT.len());
    sheaf_msg!("Add this file to your AI assistant context (e.g. CLAUDE.md, .cursorrules, etc.)");
}
