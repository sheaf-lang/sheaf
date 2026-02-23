// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf CLI
//!
//! Usage:
//!   sheaf                               Launch interactive REPL
//!   sheaf file.shf                      Interpret a Sheaf file
//!   sheaf -c '(+ 1 2)'                 Evaluate an expression
//!   sheaf build file.shf -o out.vmfb   Compile to VMFB (requires Sheaf SDK)
//!   sheaf build file.shf -o out.mlir -S  Emit MLIR only (no SDK required)

use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("build") => run_build(&args[2..]),

        Some("-c") => {
            let expr = args[2..].join(" ");
            if expr.is_empty() {
                eprintln!("sheaf: -c requires an expression");
                exit(1);
            }
            run_expr(&expr);
        }

        Some("--help") | Some("-h") => print_help(),

        Some("--version") => println!("Sheaf {}", env!("CARGO_PKG_VERSION")),

        Some(arg) if !arg.starts_with('-') => run_file(&args[1..]),

        None => run_repl(),

        Some(arg) => {
            eprintln!("sheaf: unknown command '{}'", arg);
            eprintln!("Run 'sheaf --help' for usage.");
            exit(1);
        }
    }
}

fn print_help() {
    println!(
        "Sheaf {} - A Functional Language for Differentiable Computation

Usage:
    sheaf                              Launch interactive REPL
    sheaf FILE [OPTIONS]               Interpret a Sheaf file
    sheaf -c EXPR [OPTIONS]            Evaluate an expression
    sheaf build FILE -o OUTPUT         Compile to VMFB (requires Sheaf SDK)
    sheaf build FILE -o OUTPUT -S      Emit MLIR only (no SDK required)

Interpreter options:
    --trace [FUNCTIONS]    Trace execution (optionally scoped to functions)
    --guard SPEC           Runtime guard: [scope:]variable:check (repeatable)

Build options:
    -o OUTPUT              Output file (.vmfb or .mlir)
    -S                     Emit MLIR only, do not invoke iree-compile
    --backend BACKEND      IREE target backend (default: llvm-cpu)
    -v, --verbose          Verbose output

Examples:
    sheaf script.shf
    sheaf -c '(+ 1 2)'
    sheaf script.shf --trace forward
    sheaf build model.shf -o model.vmfb
    sheaf build model.shf -o model.mlir -S
    sheaf build model.shf -o model.vmfb --backend cuda

SDK:
    'sheaf build' without -S requires iree-compile from the Sheaf SDK.
    Set IREE_COMPILE=/path/to/iree-compile to override the default location.",
        env!("CARGO_PKG_VERSION")
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

fn run_expr(source: &str) {
    use sheaf_compiler::interpreter::eval::eval_source;
    match eval_source(source) {
        Ok(val) => println!("{}", val),
        Err(e) => {
            eprintln!("{}", e);
            exit(1);
        }
    }
}

fn run_file(args: &[String]) {
    use std::path::PathBuf;
    use sheaf_compiler::interpreter::eval::eval_source_with_path;
    use sheaf_compiler::interpreter::value::Value;

    let path = &args[0];

    // Parse interpreter-only options; reject build options
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--trace" => {
                if args.get(i + 1).map(|a| !a.starts_with('-')).unwrap_or(false) {
                    i += 1;
                }
            }
            "--guard" => {
                i += 1;
            }
            arg => {
                eprintln!("sheaf: unknown option '{}' for file mode", arg);
                eprintln!("Run 'sheaf --help' for usage.");
                exit(1);
            }
        }
        i += 1;
    }

    let abs_path = PathBuf::from(path)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(path));

    let source = match std::fs::read_to_string(&abs_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("sheaf: cannot read '{}': {}", path, e);
            exit(1);
        }
    };
    match eval_source_with_path(&source, Some(&abs_path)) {
        Ok(val) => {
            if !is_silent_result(&val) {
                println!("{}", val);
            }
        }
        Err(e) => {
            eprintln!("{}", e);
            exit(1);
        }
    }
}

fn run_repl() {
    use rustyline::DefaultEditor;
    use rustyline::error::ReadlineError;
    use sheaf_compiler::interpreter::eval::Interpreter;
    use sheaf_compiler::interpreter::value::Value;

    println!("Sheaf {} — interactive REPL", env!("CARGO_PKG_VERSION"));
    println!("Type :quit or Ctrl-D to exit.\n");

    let history_file = std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".sheaf_history"));

    let mut rl = DefaultEditor::new().expect("failed to init line editor");
    // On macOS/libedit, Ctrl-D needs an explicit binding to trigger EndOfFile
    rl.bind_sequence(
        rustyline::KeyEvent(rustyline::KeyCode::Char('d'), rustyline::Modifiers::CTRL),
        rustyline::Cmd::EndOfFile,
    );
    if let Some(ref path) = history_file {
        let _ = rl.load_history(path);
    }

    let mut interp = Interpreter::new();

    loop {
        match rl.readline("sheaf> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == ":quit" || trimmed == ":q" {
                    break;
                }
                rl.add_history_entry(trimmed).ok();
                match interp.eval(trimmed) {
                    Ok(val) => {
                        if !is_silent_result(&val) {
                            println!("{}", val);
                        }
                    }
                    Err(e) => eprintln!("error: {}", e),
                }
            }
            Err(ReadlineError::Interrupted) => {
                eprintln!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("\nBye!");
                break;
            }
            Err(e) => {
                eprintln!("sheaf: read error: {}", e);
                break;
            }
        }
    }

    if let Some(ref path) = history_file {
        let _ = rl.save_history(path);
    }
}

fn run_build(args: &[String]) {
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    use sheaf_compiler::compiler::{
        build_index_map, collect_effects, format_effects, json_to_stablehlo_type, lower_get_calls,
    };
    use sheaf_compiler::core::compiler::CompilerContext;
    use sheaf_compiler::{CodeGenerator, StableHLOEmitter, parse};

    if args.first().map(|a| a == "--help" || a == "-h").unwrap_or(false) {
        println!(
            "Usage: sheaf build FILE -o OUTPUT [-S] [--config JSON] [--backend BACKEND] [-v]

    FILE            Input Sheaf source file (.shf)
    -o OUTPUT       Output file (.vmfb or .mlir with -S)
    -S              Emit MLIR only; do not invoke iree-compile
    --config JSON   Shape config for dict params, e.g. '{{\"p\":{{\"l1\":{{\"W\":[2,8],\"b\":[8]}}}}}}'
    --backend B     IREE target backend (default: llvm-cpu)
    -v, --verbose   Verbose output

Without -S, requires iree-compile (Sheaf SDK).
Set IREE_COMPILE=/path/to/iree-compile to override."
        );
        return;
    }

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut emit_mlir_only = false;
    let mut backend = "llvm-cpu".to_string();
    let mut verbose = false;
    let mut config_json: Option<serde_json::Value> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("sheaf build: -o requires an argument");
                    exit(1);
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "-S" => emit_mlir_only = true,
            "--backend" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("sheaf build: --backend requires an argument");
                    exit(1);
                }
                backend = args[i].clone();
            }
            "-v" | "--verbose" => verbose = true,
            "--config" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("sheaf build: --config requires a JSON argument");
                    exit(1);
                }
                config_json = Some(serde_json::from_str(&args[i]).unwrap_or_else(|e| {
                    eprintln!("sheaf build: --config: invalid JSON: {}", e);
                    exit(1);
                }));
            }
            // Reject interpreter-only options explicitly
            "--trace" | "--guard" => {
                eprintln!(
                    "sheaf build: '{}' is an interpreter option and cannot be used with 'build'",
                    args[i]
                );
                exit(1);
            }
            arg if !arg.starts_with('-') => {
                input = Some(PathBuf::from(arg));
            }
            arg => {
                eprintln!("sheaf build: unknown option '{}'", arg);
                eprintln!("Run 'sheaf build --help' for usage.");
                exit(1);
            }
        }
        i += 1;
    }

    let input = input.unwrap_or_else(|| {
        eprintln!("sheaf build: no input file specified");
        exit(1);
    });
    let output = output.unwrap_or_else(|| {
        eprintln!("sheaf build: no output file specified (-o)");
        exit(1);
    });

    // Detect output format from extension when -S not set
    let emit_mlir_only = emit_mlir_only
        || output.extension().and_then(|e| e.to_str()) == Some("mlir");

    let source = fs::read_to_string(&input).unwrap_or_else(|e| {
        eprintln!("sheaf: cannot read '{}': {}", input.display(), e);
        exit(1);
    });

    if verbose {
        println!("Parsing {}...", input.display());
    }

    let exprs = parse(&source, input.to_str().unwrap()).unwrap_or_else(|e| {
        eprintln!("parse error: {}", e);
        exit(1);
    });

    if exprs.is_empty() {
        eprintln!("sheaf: no expressions found in '{}'", input.display());
        exit(1);
    }

    if verbose {
        println!("Compiling...");
    }

    let mut compiler = CompilerContext::new();
    if let Some(dir) = input.canonicalize().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        compiler.current_dir = Some(dir);
    }
    let mut compiled_exprs = Vec::new();
    for expr in &exprs {
        match compiler.compile(expr) {
            Ok(c) => compiled_exprs.push(c),
            Err(e) => {
                eprintln!("compilation error: {}", e);
                exit(1);
            }
        }
    }

    let extra_decls = resolve_vag_decls(&compiler, &compiled_exprs, verbose);

    if verbose {
        println!("Generating StableHLO...");
    }

    // Emit all user-defined functions in the registry as a single MLIR module.
    // Functions from (use ...) imports are included only if they were defined
    // in the source file itself (tracked via compiled_exprs).
    let mut all_decls = extra_decls;

    // Collect function names defined directly in this file (i.e. top-level defn forms)
    let file_functions: Vec<String> = exprs
        .iter()
        .filter_map(|e| {
            e.as_list()
                .and_then(|l| l.first())
                .and_then(|h| h.as_symbol())
                .filter(|&s| s == "defn")
                .and_then(|_| {
                    e.as_list()
                        .and_then(|l| l.get(1))
                        .and_then(|n| n.as_symbol())
                        .map(|s| s.to_string())
                })
        })
        .collect();

    if file_functions.is_empty() {
        eprintln!("error: no functions defined in '{}'", input.display());
        exit(1);
    }

    // Build per-param config from --config JSON:
    // config_json top level: {"param_name": {dict structure}, ...}
    // e.g. {"p": {"l1": {"W": [2,8], "b": [8]}, "l2": {"W": [8,1], "b": [1]}}}
    let param_configs: Vec<(String, serde_json::Value)> = match &config_json {
        Some(serde_json::Value::Object(map)) => map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        Some(_) => {
            eprintln!("sheaf build: --config must be a JSON object {{\"param\": {{...}}}}");
            exit(1);
        }
        None => vec![],
    };

    for name in &file_functions {
        let func_def = match compiler.registry.get(name).cloned() {
            Some(f) => f,
            None => continue,
        };
        let mut body = match func_def.body_compiled {
            Some(b) => b,
            None => {
                if verbose { eprintln!("warning: '{}' has no compiled body, skipping", name); }
                continue;
            }
        };
        let mut sig = match func_def.signature {
            Some(s) => s,
            None => {
                if verbose { eprintln!("warning: '{}' has no inferred signature, skipping", name); }
                continue;
            }
        };

        // Refuse functions with side effects: they cannot be emitted as StableHLO.
        let effects = collect_effects(&body);
        if !effects.is_empty() {
            eprintln!(
                "error: '{}' has side effects ({}) and cannot be compiled",
                name,
                format_effects(&effects)
            );
            eprintln!("  Only side-effect-free functions can be compiled to StableHLO.");
            exit(1);
        }

        // Apply dict-to-tuple lowering for each configured param that appears
        // in this function's parameter list.
        let mut known_types: Vec<(String, sheaf_compiler::StableHLOType)> = Vec::new();
        for (param_name, param_config) in &param_configs {
            if !func_def.params.contains(param_name) {
                continue;
            }
            let tuple_ty = match json_to_stablehlo_type(param_config) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("sheaf build: --config error for '{}': {}", param_name, e);
                    exit(1);
                }
            };
            let index_map = build_index_map(param_config);
            if verbose {
                println!("  Lowering '{}' param '{}' → {}", name, param_name, tuple_ty.to_mlir());
            }
            body = lower_get_calls(&body, param_name, &index_map);
            known_types.push((param_name.clone(), tuple_ty));
        }

        // Re-infer signature if we have new known types from the config
        if !known_types.is_empty() {
            use sheaf_compiler::core::inference::infer_function_signature_with_known;
            sig = infer_function_signature_with_known(
                &compiler,
                &func_def.params,
                &body,
                &known_types,
            ).unwrap_or_else(|e| {
                eprintln!("type inference error in '{}': {}", name, e);
                exit(1);
            });
            // Override param types for configured params (inference may default to scalar)
            for (param_name, tuple_ty) in &known_types {
                if let Some(idx) = func_def.params.iter().position(|p| p == param_name) {
                    sig.param_types[idx] = tuple_ty.clone();
                }
            }
        }

        let codegen = CodeGenerator::with_function_params(
            compiler.registry.clone(),
            &func_def.params,
            &sig.param_types,
        );
        match codegen.emit_func_declaration(name, &body, &sig.param_types, &sig.return_type) {
            Ok(decl) => all_decls.push(decl),
            Err(e) => {
                if verbose {
                    eprintln!("warning: skipping '{}': {}", name, e);
                } else {
                    eprintln!("warning: skipping '{}' (use -v for details)", name);
                }
            }
        }
    }

    if all_decls.is_empty() {
        eprintln!("error: nothing to emit from '{}'", input.display());
        exit(1);
    }

    let mlir = StableHLOEmitter::emit_module(&all_decls);

    if emit_mlir_only {
        fs::write(&output, &mlir).unwrap_or_else(|e| {
            eprintln!("error writing '{}': {}", output.display(), e);
            exit(1);
        });
        if verbose {
            println!("Wrote {}", output.display());
        }
        return;
    }

    // VMFB path — requires iree-compile
    let iree_compile = find_iree_compile();

    // Write intermediate MLIR to a temp file
    let mlir_path = output.with_extension("mlir");
    fs::write(&mlir_path, &mlir).unwrap_or_else(|e| {
        eprintln!("error writing '{}': {}", mlir_path.display(), e);
        exit(1);
    });

    if verbose {
        println!("Running iree-compile ({})...", iree_compile);
    }

    let status = Command::new(&iree_compile)
        .arg(&mlir_path)
        .arg(format!("--iree-hal-target-backends={}", backend))
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("error running iree-compile '{}': {}", iree_compile, e);
            eprintln!("Install the Sheaf SDK or set IREE_COMPILE=/path/to/iree-compile");
            exit(1);
        });

    // Clean up temp MLIR
    let _ = fs::remove_file(&mlir_path);

    if !status.success() {
        eprintln!("iree-compile failed");
        exit(1);
    }

    if verbose {
        println!("Wrote {}", output.display());
        println!("Done.");
    }
}

fn find_iree_compile() -> String {
    // 1. Explicit env var
    if let Ok(path) = std::env::var("IREE_COMPILE") {
        return path;
    }
    // 2. Standard SDK install location
    if let Ok(home) = std::env::var("HOME") {
        let candidate = format!("{}/bin/iree-build/tools/iree-compile", home);
        if std::path::Path::new(&candidate).exists() {
            return candidate;
        }
    }
    // 3. PATH
    if let Some(path) = which("iree-compile") {
        return path;
    }
    eprintln!("error: 'sheaf build' requires the Sheaf SDK (iree-compile not found)");
    eprintln!("  Install the SDK or set IREE_COMPILE=/path/to/iree-compile");
    eprintln!("  To emit MLIR only (no SDK): sheaf build FILE -o FILE.mlir -S");
    exit(1);
}

fn which(name: &str) -> Option<String> {
    std::env::var("PATH").ok().and_then(|path_var| {
        path_var.split(':').find_map(|dir| {
            let candidate = format!("{}/{}", dir, name);
            if std::path::Path::new(&candidate).exists() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}

fn resolve_vag_decls(
    compiler: &sheaf_compiler::core::compiler::CompilerContext,
    compiled_exprs: &[sheaf_compiler::core::compiler::CompiledExpr],
    verbose: bool,
) -> Vec<String> {
    use sheaf_compiler::autodiff::value_and_grad::{GradParam, emit_value_and_grad_func};
    use sheaf_compiler::core::inference::infer_function_signature_with_known;
    use sheaf_compiler::StableHLOType;

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
                    eprintln!(
                        "value-and-grad '{}': signature inference failed: {}",
                        fn_name, e
                    );
                    exit(1);
                }
            }
        } else {
            match &func_def.signature {
                Some(sig) => sig.clone(),
                None => {
                    eprintln!(
                        "value-and-grad '{}': function '{}' has no inferred signature",
                        fn_name, src_fn_name
                    );
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

        if verbose {
            println!("Emitting value-and-grad '{}'...", fn_name);
        }

        let func_decl = emit_value_and_grad_func(
            fn_name,
            &func_def.params,
            &signature.param_types,
            body_compiled,
            &grad_params,
            compiler.registry.clone(),
        )
        .unwrap_or_else(|e| {
            eprintln!("value-and-grad '{}': codegen failed: {}", fn_name, e);
            exit(1);
        });

        decls.push(func_decl);
    }
    decls
}

fn collect_vag_nodes<'a>(
    expr: &'a sheaf_compiler::core::compiler::CompiledExpr,
    out: &mut Vec<(&'a str, &'a str, &'a Vec<String>, &'a Vec<(String, Vec<i64>)>)>,
) {
    use sheaf_compiler::core::compiler::CompiledExpr;
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
