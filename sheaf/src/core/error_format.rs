// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Pretty error formatting with code context, carets, and hints.
//!
//! Inspired by the Rust compiler's diagnostic style and Sheaf V1's error handler.
//! The `format_error` function is the rich CLI format; `SheafError::Display` remains
//! the simple one-liner for logs and tests.

use std::cell::RefCell;
use std::collections::HashMap;

use super::error::SheafError;

thread_local! {
    static SOURCE_REGISTRY: RefCell<HashMap<String, String>> = RefCell::new(HashMap::new());
}

/// Register source text for a filename (called at every file read).
pub fn register_source(filename: &str, source: &str) {
    SOURCE_REGISTRY.with(|reg| {
        reg.borrow_mut()
            .insert(filename.to_string(), source.to_string());
    });
}

fn get_source(filename: &str) -> Option<String> {
    SOURCE_REGISTRY.with(|reg| reg.borrow().get(filename).cloned())
}

/// Format a SheafError with code context, carets, and hints.
pub fn format_error(error: &SheafError) -> String {
    let mut parts = Vec::new();

    match error {
        SheafError::Parse { message, location } => {
            parts.push(format!("error: {}", message));
            parts.push(format!(
                " --> {}:{}",
                location.filename, location.line
            ));
            append_code_context(
                &mut parts,
                &location.filename,
                location.line,
                location.column,
            );
            if let Some(hint) = get_parse_hint(message, &location.filename) {
                parts.push(format!("  = hint: {}", hint));
            }
        }
        SheafError::Compile { message, location }
        | SheafError::CompilerUnsupported { message, location, .. }
        | SheafError::CompilerInvariant { message, location }
        | SheafError::InvalidSourceForm { message, location }
        | SheafError::InvalidShapeDimension { message, location } => {
            parts.push(format!("error: {}", message));
            parts.push(format!(
                " --> {}:{}",
                location.filename, location.line
            ));
            append_code_context(
                &mut parts,
                &location.filename,
                location.line,
                location.column,
            );
            if let Some(hint) = get_compile_hint(message) {
                parts.push(format!("  = hint: {}", hint));
            }
        }
        SheafError::Runtime {
            message,
            location: Some(loc),
        } => {
            parts.push(format!("error: {}", message));
            parts.push(format!(" --> {}:{}", loc.filename, loc.line));
            append_code_context(&mut parts, &loc.filename, loc.line, loc.column);
            if let Some(hint) = get_runtime_hint(message) {
                parts.push(format!("  = hint: {}", hint));
            }
        }
        SheafError::Runtime {
            message,
            location: None,
        } => {
            parts.push(format!("error: {}", message));
            if let Some(hint) = get_runtime_hint(message) {
                parts.push(format!("  = hint: {}", hint));
            }
        }
        SheafError::AutodiffMissingRule {
            operation,
            location: Some(loc),
        } => {
            parts.push(format!(
                "error: no differentiation rule for operation '{}'",
                operation
            ));
            parts.push(format!(" --> {}:{}", loc.filename, loc.line));
            append_code_context(&mut parts, &loc.filename, loc.line, loc.column);
        }
        SheafError::AutodiffMissingRule {
            operation,
            location: None,
        } => {
            parts.push(format!(
                "error: no differentiation rule for operation '{}'",
                operation
            ));
        }
        SheafError::AutodiffMissingGradientOutput { symbol } => {
            parts.push(format!(
                "error: missing gradient output for symbol '{}'",
                symbol
            ));
        }
        SheafError::Io(msg) => {
            parts.push(format!("io error: {}", msg));
        }
    }

    parts.push(String::new());
    parts.join("\n")
}

// Code context with line numbers and carets

fn append_code_context(parts: &mut Vec<String>, filename: &str, line: usize, column: usize) {
    let source = match get_source(filename) {
        Some(s) => s,
        None => return,
    };

    let lines: Vec<&str> = source.lines().collect();
    if line == 0 || line > lines.len() {
        return;
    }

    let context = 2;
    let start = line.saturating_sub(context).max(1);
    let end = (line + context).min(lines.len());
    let gutter = format!("{}", end).len();

    parts.push(format!("{:>w$} |", "", w = gutter));

    for i in start..=end {
        let text = lines[i - 1];
        if i == line {
            parts.push(format!("{:>w$} | {}", i, text, w = gutter));
            let col = if column > 0 { column - 1 } else {
                // No column info: point to first non-whitespace
                text.len() - text.trim_start().len()
            };
            let caret_len = token_length_at(text, col);
            parts.push(format!(
                "{:>w$} | {}{}",
                "",
                " ".repeat(col),
                "^".repeat(caret_len.max(1)),
                w = gutter
            ));
        } else {
            parts.push(format!("{:>w$} | {}", i, text, w = gutter));
        }
    }

    parts.push(format!("{:>w$} |", "", w = gutter));
}

fn token_length_at(line: &str, col: usize) -> usize {
    let bytes = line.as_bytes();
    if col >= bytes.len() {
        return 1;
    }
    let delimiters = b" \t\n()[]{}\"';,";
    let mut len = 0;
    for &ch in &bytes[col..] {
        if delimiters.contains(&ch) {
            break;
        }
        len += 1;
    }
    len.max(1)
}

// Hint system

fn known_mistake_hint(symbol: &str) -> Option<&'static str> {
    match symbol {
        "define" => Some("Sheaf uses 'def' for constants and 'defn' for functions."),
        "lambda" => Some("Sheaf uses 'fn' for anonymous functions: (fn [args] body)"),
        "set!" => {
            Some("Sheaf is purely functional. Use 'let' for bindings or 'assoc' for dicts.")
        }
        "return" => {
            Some("Sheaf is expression-based. The last expression in a body is the return value.")
        }
        "import" | "require" => Some("Sheaf uses 'use' to load modules: (use nn)"),
        "print!" => Some("Sheaf uses 'print': (print value)."),
        "var" | "const" => Some("Sheaf uses 'let' for local bindings: (let [x 1] body)"),
        "nil?" => Some("Use (= x nil) to check for nil."),
        "list" => Some("Use a quoted vector: '[1 2 3]."),
        "cond" => {
            Some("Use 'case' for multi-branch: (case pred1 val1 pred2 val2 default)")
        }
        "true?" | "false?" => Some("Booleans are truthy/falsy directly. Use them in 'if' or 'and'."),
        "begin" | "progn" => Some("Sheaf uses 'do' for sequencing: (do expr1 expr2 ...)"),
        "car" | "cdr" => Some("Use 'first' and 'rest': (first xs), (rest xs)"),
        "defvar" | "defparameter" => {
            Some("Sheaf uses 'def' for constants: (def name value)")
        }
        "loop" | "recur" => Some("Use 'repeat' for counted loops or 'reduce' for accumulation."),
        "inc" => Some("Use (+ x 1) for incrementation."),
        "dec" => Some("Use (- x 1) for decrementation."),
        "format" => Some("Use print with format strings: (print \"x={:.4f}\" x)"),
        "for" | "foreach" | "for-each" => Some("Use 'map', 'reduce', or 'repeat' for iteration."),
        "while!" | "loop!" => Some("Use 'while' for conditional loops: (while [state init] cond body)"),
        "type" | "typeof" | "type-of" => Some("Use (shape x) for tensor shape, (ndim x) for rank."),
        "append!" | "push" | "push!" => Some("Use 'append' (functional): (append lst item)"),
        "reverse" | "flip" => Some("Use (flip x) to reverse a tensor or list along axis 0, or (flip x :axis N) for a specific axis."),
        _ => None,
    }
}

fn get_compile_hint(message: &str) -> Option<String> {
    // Undefined symbol
    if let Some(symbol) = message.strip_prefix("Undefined symbol: ") {
        let sym = symbol.trim().trim_matches('\'');
        if let Some(hint) = known_mistake_hint(sym) {
            return Some(hint.to_string());
        }
        return Some("Check for typos in function or variable names.".to_string());
    }

    // with-params binding vector
    if message.contains("with-params") && message.contains("binding vector") {
        return Some(
            "with-params requires a binding vector: (with-params [dict] body) or (with-params [dict :key] body)"
                .to_string(),
        );
    }

    // defn body compilation
    if message.contains("defn") && message.contains("expected") {
        return None; // message already contains usage info
    }

    get_shape_hint(message)
}

fn get_shape_hint(message: &str) -> Option<String> {
    if message.contains("shape") && message.contains("integer") && message.contains("float") {
        Some("Use // for integer division, or convert explicitly with (int ...).".to_string())
    } else {
        None
    }
}

fn get_runtime_hint(message: &str) -> Option<String> {
    // Unknown/undefined function
    if message.starts_with("Unknown function: ") || message.starts_with("Undefined function: ") {
        let name = message.split(':').nth(1).map(|s| s.trim()).unwrap_or("");
        if let Some(hint) = known_mistake_hint(name) {
            return Some(hint.to_string());
        }
        return Some("Check for typos. Use (use module) to import functions.".to_string());
    }

    // Undefined symbol
    if message.starts_with("Undefined symbol: ") {
        let sym = message.strip_prefix("Undefined symbol: ").unwrap_or("").trim();
        if let Some(hint) = known_mistake_hint(sym) {
            return Some(hint.to_string());
        }
        return Some("Check for typos in variable names.".to_string());
    }

    // Collection type errors
    if message.contains("expected list or tensor") {
        return Some(
            "This function operates on lists or tensors. Check argument types and order."
                .to_string(),
        );
    }
    if message.contains("expected list") && !message.contains("or tensor") {
        return Some("This function requires a list argument.".to_string());
    }

    // Broadcasting
    if message.contains("broadcast") || message.contains("Broadcasting") {
        return Some(
            "Use (shape x) to inspect dimensions. Did you forget to transpose?".to_string(),
        );
    }

    if let Some(hint) = get_shape_hint(message) {
        return Some(hint);
    }

    // Index out of bounds
    if message.contains("is out of bounds for axis") {
        // Extract index and size for a more helpful hint
        if let (Some(_idx), Some(size)) = (
            message.split("index ").nth(1).and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse::<usize>().ok()),
            message.split("with size ").nth(1).and_then(|s| s.parse::<usize>().ok()),
        ) {
            return Some(format!("Indices are 0-based. Valid range is [0, {}].", size - 1));
        }
        return Some("Index exceeds dimension size. Use (shape x) to check dimensions.".to_string());
    }
    if message.contains("index out of bounds") || message.contains("out of range") {
        return Some(
            "Index exceeds collection length. Use (len xs) to check bounds.".to_string(),
        );
    }

    // Integer indexer
    if message.contains("index must be int") || message.contains("indexer must have integer") {
        return Some("Indices must be plain integers. Wrap with (int idx).".to_string());
    }

    // Not a function
    if message.contains("Expected a function, got") || message.contains("not callable") {
        return Some("Check that you're calling a function, not a value.".to_string());
    }

    // get on wrong type
    if message.contains("get:") && message.contains("expected dict") {
        return Some("'get' operates on dicts, lists, or tensors.".to_string());
    }

    // Key not found
    if message.contains("key not found") || message.contains("Key not found") {
        return Some(
            "Verify the key exists. Use (keys dict) to list available keys.".to_string(),
        );
    }

    // value-and-grad scalar
    if message.contains("value-and-grad") && message.contains("scalar") {
        return Some(
            "The loss function must return a single scalar, not a tensor or tuple.".to_string(),
        );
    }

    None
}

fn get_parse_hint(message: &str, filename: &str) -> Option<String> {
    let msg = message.to_lowercase();

    // Extra closing paren (skip if parser already provided a hint)
    if (msg.contains("unexpected closing") || msg.contains("unmatched")) && !msg.contains("hint:") {
        if let Some(source) = get_source(filename) {
            let info = find_unmatched_paren(&source);
            if let Some(line) = info.excess_line {
                let lines: Vec<&str> = source.lines().collect();
                if line <= lines.len() {
                    return Some(format!(
                        "Extra closing paren on line {}:\n       {} | {}",
                        line, line, lines[line - 1]
                    ));
                }
            }
            if !info.suspect_lines.is_empty() {
                let (ln, _, close_count) = info
                    .suspect_lines
                    .iter()
                    .copied()
                    .max_by_key(|&(_, _, c)| c)
                    .unwrap();
                let lines: Vec<&str> = source.lines().collect();
                if ln <= lines.len() {
                    return Some(format!(
                        "Suspicious line with {} closing parens:\n       {} | {}",
                        close_count, ln, lines[ln - 1]
                    ));
                }
            }
        }
        return Some("Check for extra closing parentheses.".to_string());
    }

    // Unclosed paren (skip if parser already provided a hint)
    if (msg.contains("unclosed") || msg.contains("unexpected end")) && !msg.contains("hint:") {
        if let Some(source) = get_source(filename) {
            let info = find_unmatched_paren(&source);
            let mut hints = Vec::new();
            if let Some(line) = info.culprit_line {
                let ctx = info
                    .culprit_context
                    .map(|c| format!(" (`{}`)", c))
                    .unwrap_or_default();
                hints.push(format!(
                    "Probable culprit: opening paren at line {}{}",
                    line, ctx
                ));
            }
            if info.max_depth > 4 {
                hints.push(format!(
                    "Max nesting depth: {}. Consider extracting sub-expressions.",
                    info.max_depth
                ));
            }
            if !hints.is_empty() {
                return Some(hints.join("\n  = hint: "));
            }
        }
        return Some("Check for missing closing parentheses.".to_string());
    }

    // Unterminated string
    if msg.contains("unterminated string") {
        return Some("Missing closing quote. Strings use double quotes: \"text\"".to_string());
    }

    None
}

// Paren analysis (ported from V1's _find_unmatched_paren)

struct ParenInfo {
    culprit_line: Option<usize>,
    culprit_context: Option<String>,
    excess_line: Option<usize>,
    suspect_lines: Vec<(usize, usize, usize)>, // (line, open_count, close_count)
    max_depth: usize,
}

fn find_unmatched_paren(source: &str) -> ParenInfo {
    let mut stack: Vec<(usize, Option<String>)> = Vec::new();
    let mut max_depth = 0;
    let mut excess_line = None;
    let mut suspect_lines = Vec::new();

    for (line_idx, line) in source.lines().enumerate() {
        let line_num = line_idx + 1;
        let mut line_open = 0usize;
        let mut line_close = 0usize;
        let bytes = line.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let ch = bytes[i];

            // Skip comments
            if ch == b';' {
                break;
            }

            // Skip strings
            if ch == b'"' {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
                continue;
            }

            if ch == b'(' {
                line_open += 1;
                // Extract context token
                let rest = &line[i + 1..];
                let rest = rest.trim_start();
                let token: String = rest
                    .chars()
                    .take_while(|c| !matches!(c, ' ' | '\t' | '(' | ')' | '[' | ']' | '{' | '}'))
                    .collect();
                let ctx = if token.is_empty() { None } else { Some(token) };
                stack.push((line_num, ctx));
                if stack.len() > max_depth {
                    max_depth = stack.len();
                }
            } else if ch == b')' {
                line_close += 1;
                if !stack.is_empty() {
                    stack.pop();
                } else if excess_line.is_none() {
                    excess_line = Some(line_num);
                }
            }

            i += 1;
        }

        if line_close >= 4 && line_close > line_open {
            suspect_lines.push((line_num, line_open, line_close));
        }
    }

    let culprit_line = stack.first().map(|(l, _)| *l);
    let culprit_context = stack.first().and_then(|(_, c)| c.clone());

    ParenInfo {
        culprit_line,
        culprit_context,
        excess_line,
        suspect_lines,
        max_depth,
    }
}
