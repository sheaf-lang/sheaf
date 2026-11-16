// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf interactive REPL with autocompletion, multiline, help, and tracing.

use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Editor, Helper};
use sheaf_compiler::forms::special_forms_registry;
use sheaf_compiler::interpreter::eval::Interpreter;
use sheaf_compiler::core::color;
use sheaf_compiler::core::signature::OpId;
use sheaf_compiler::interpreter::value::Value;
use std::borrow::Cow;
use std::collections::HashSet;
use std::cell::Cell;
use unicode_width::UnicodeWidthStr;

use crate::doc;
use crate::pretty;

struct MatchingBracketHighlighter {
    bracket: Cell<Option<(u8, usize)>>,
}

impl MatchingBracketHighlighter {
    fn new() -> Self {
        Self { bracket: Cell::new(None) }
    }
}

impl Highlighter for MatchingBracketHighlighter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        if line.len() <= 1 {
            return Cow::Borrowed(line);
        }
        if let Some((bracket, pos)) = self.bracket.get()
            && let Some((matching, idx)) = find_matching_bracket(line, pos, bracket)
        {
            let mut copy = line.to_owned();
            copy.replace_range(idx..=idx, &format!("\x1b[7m{}\x1b[0m", matching as char));
            return Cow::Owned(copy);
        }
        Cow::Borrowed(line)
    }

    fn highlight_char(&self, line: &str, pos: usize, forced: bool) -> bool {
        if forced {
            self.bracket.set(None);
            return false;
        }
        self.bracket.set(check_bracket(line, pos));
        self.bracket.get().is_some()
    }
}

fn find_matching_bracket(line: &str, pos: usize, bracket: u8) -> Option<(u8, usize)> {
    let matching = matching_bracket(bracket);
    let bytes = line.as_bytes();
    if is_open_bracket(bracket) {
        let mut unmatched = 1i32;
        let mut idx = pos + 1;
        while idx < bytes.len() {
            if !(in_string_or_comment(bytes, idx)) {
                let b = bytes[idx];
                if b == matching {
                    unmatched -= 1;
                    if unmatched == 0 {
                        return Some((matching, idx));
                    }
                } else if b == bracket {
                    unmatched += 1;
                }
            }
            idx += 1;
        }
    } else {
        let mut unmatched = 1i32;
        let mut idx = pos as i64 - 1;
        while idx >= 0 {
            let i = idx as usize;
            if !(in_string_or_comment(bytes, i)) {
                let b = bytes[i];
                if b == matching {
                    unmatched -= 1;
                    if unmatched == 0 {
                        return Some((matching, i));
                    }
                } else if b == bracket {
                    unmatched += 1;
                }
            }
            idx -= 1;
        }
    }
    None
}

fn in_string_or_comment(bytes: &[u8], pos: usize) -> bool {
    let mut in_string = false;
    let mut i = 0;
    while i < pos {
        let b = bytes[i];
        if in_string {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
        } else {
            if b == b'"' {
                in_string = true;
            } else if b == b';' {
                while i < pos {
                    if bytes[i] == b'\n' {
                        break;
                    }
                    i += 1;
                }
                continue;
            }
        }
        i += 1;
    }
    in_string
}

fn check_bracket(line: &str, pos: usize) -> Option<(u8, usize)> {
    if line.is_empty() { return None; }
    let mut pos = pos;
    let bytes = line.as_bytes();
    if pos >= line.len() {
        pos = line.len() - 1;
        let b = bytes[pos];
        if is_close_bracket(b) && !in_string_or_comment(bytes, pos) {
            return Some((b, pos));
        }
        return None;
    }
    let mut under_cursor = true;
    loop {
        if in_string_or_comment(bytes, pos) {
            return None;
        }
        let b = bytes[pos];
        if is_close_bracket(b) {
            return if pos == 0 { None } else { Some((b, pos)) };
        } else if is_open_bracket(b) {
            return if pos + 1 == line.len() { None } else { Some((b, pos)) };
        } else if under_cursor && pos > 0 {
            under_cursor = false;
            pos -= 1;
        } else {
            return None;
        }
    }
}

const fn matching_bracket(bracket: u8) -> u8 {
    match bracket {
        b'{' => b'}', b'}' => b'{',
        b'[' => b']', b']' => b'[',
        b'(' => b')', b')' => b'(',
        b => b,
    }
}

const fn is_open_bracket(b: u8) -> bool { matches!(b, b'{' | b'[' | b'(') }
const fn is_close_bracket(b: u8) -> bool { matches!(b, b'}' | b']' | b')') }

  const REPL_COMMANDS: &[&str] = &[
    ":help", ":h", ":?",
    ":list",
    ":quit", ":q",
    ":env",
    ":registry", ":reg",
    ":show",
    ":clear",
];

struct SheafHelper {
    names: Vec<String>,
    term_cols: usize,
    last_was_empty_tab: std::cell::Cell<bool>,
    bracket_hl: MatchingBracketHighlighter,
}

impl SheafHelper {
    fn new() -> Self {
        let mut names: Vec<String> = special_forms_registry()
            .keys()
            .map(|k| k.to_string())
            .collect();
        names.sort();
        Self { names, term_cols: 80, last_was_empty_tab: std::cell::Cell::new(false), bracket_hl: MatchingBracketHighlighter::new() }
    }

    fn refresh(&mut self, interp: &Interpreter) {
        let mut set = HashSet::new();
        for name in special_forms_registry().keys() {
            set.insert(name.to_string());
        }
        for name in interp.registry_names() {
            set.insert(name);
        }
        for name in interp.env().all_names() {
            set.insert(name);
        }
        let mut sorted: Vec<String> = set
            .into_iter()
            .filter(|name| OpId::resolve(name).is_none_or(|op| op.is_public()))
            .collect();
        sorted.sort();
        self.names = sorted;
    }
}

impl Completer for SheafHelper {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<String>)> {
        let delimiters = b" \t\n()[]{}";
        let start = line[..pos]
            .rfind(|c: char| delimiters.contains(&(c as u8)))
            .map(|i| i + 1)
            .unwrap_or(0);
        let word = &line[start..pos];

        let matches = if word.is_empty() {
            if self.last_was_empty_tab.get() {
                print_columns(&self.names, self.term_cols);
                print!("sheaf> ");
                let _ = std::io::Write::flush(&mut std::io::stdout());
                self.last_was_empty_tab.set(false);
            } else {
                self.last_was_empty_tab.set(true);
            }
            vec![]
        } else if word.starts_with(':') {
            self.last_was_empty_tab.set(false);
            REPL_COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(word))
                .map(|s| s.to_string())
                .collect()
        } else {
            self.last_was_empty_tab.set(false);
            self.names
                .iter()
                .filter(|name| name.starts_with(word))
                .cloned()
                .collect()
        };

        Ok((start, matches))
    }
}

impl Validator for SheafHelper {}

impl Hinter for SheafHelper {
    type Hint = String;
    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for SheafHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.bracket_hl.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize, forced: bool) -> bool {
        self.bracket_hl.highlight_char(line, pos, forced)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(&'s self, prompt: &'p str, default: bool) -> Cow<'b, str> {
        if default {
            Cow::Owned(format!("{}{}{}", color::color(), prompt, color::reset()))
        } else {
            // Continuation prompt for multi-line input
            Cow::Owned(format!("{}     > {}", color::color(), color::reset()))
        }
    }
}

impl Helper for SheafHelper {}

fn print_columns(names: &[String], term_width: usize) {
    if names.is_empty() {
        return;
    }
    let col_width = names.iter().map(|n| n.len()).max().unwrap_or(4) + 2;
    let num_cols = (term_width / col_width).max(1);
    let num_rows = names.len().div_ceil(num_cols);
    println!();
    for row in 0..num_rows {
        for col in 0..num_cols {
            let idx = col * num_rows + row;
            if idx < names.len() {
                print!("{:<width$}", names[idx], width = col_width);
            }
        }
        println!();
    }
}

fn is_balanced(input: &str) -> bool {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if !in_string => in_string = true,
            '"' if in_string => in_string = false,
            '\\' if in_string => { chars.next(); }
            ';' if !in_string => {
                // Skip to end of line
                for c2 in chars.by_ref() {
                    if c2 == '\n' { break; }
                }
            }
            '(' | '[' | '{' if !in_string => depth += 1,
            ')' | ']' | '}' if !in_string => depth -= 1,
            _ => {}
        }
    }
    depth <= 0
}

fn is_silent_result(val: &Value) -> bool {
    match val {
        Value::Nil => true,
        Value::List(items) => items.iter().all(|v| matches!(v, Value::Nil)),
        _ => false,
    }
}

fn format_repl_result(val: &Value) -> String {
    let prefix = match val.repl_type_prefix() {
        Some(value_type) => format!("=> {} = ", value_type),
        None => "=> ".to_string(),
    };
    let continuation = format!("\n{}", " ".repeat(UnicodeWidthStr::width(prefix.as_str())));
    format!("{}{}", prefix, val.to_string().replace('\n', &continuation))
}

pub fn run() {
    eprintln!("Sheaf {} (:h for help)", sheaf_compiler::SHEAF_VERSION);

    let history_file = std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".sheaf_history"));

    let helper = SheafHelper::new();
    let config = rustyline::Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .completion_prompt_limit(500)
        .behavior(rustyline::config::Behavior::PreferTerm)
        .build();
    let mut rl = Editor::with_config(config).expect("failed to init line editor");
    let term_cols = rl.dimensions().map(|(c, _)| c).unwrap_or(80);
    let mut helper = helper;
    helper.term_cols = term_cols;
    rl.set_helper(Some(helper));
    rl.bind_sequence(
        rustyline::KeyEvent(rustyline::KeyCode::Char('d'), rustyline::Modifiers::CTRL),
        rustyline::Cmd::EndOfFile,
    );
    if let Some(ref path) = history_file {
        let _ = rl.load_history(path);
    }

    let mut interp = Interpreter::new();

    // Initial refresh with builtins + stdlib (all stdlib loaded by CompilerContext)
    rl.helper_mut().unwrap().refresh(&interp);

    loop {
        match rl.readline("sheaf> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if trimmed.starts_with(':') {
                    rl.add_history_entry(&line).ok();
                    if handle_command(trimmed, &mut interp) {
                        break;
                    }
                    continue;
                }

                // Accumulate lines until parentheses are balanced
                let mut input = line.clone();
                while !is_balanced(&input) {
                    match rl.readline("     > ") {
                        Ok(cont) => {
                            input.push('\n');
                            input.push_str(&cont);
                        }
                        Err(ReadlineError::Interrupted) => {
                            eprintln!("^C");
                            input.clear();
                            break;
                        }
                        Err(_) => {
                            input.clear();
                            break;
                        }
                    }
                }

                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }

                // Preserve multi-line input verbatim for history recall so
                // that re-editing a multi-line form keeps its line breaks.
                rl.add_history_entry(trimmed).ok();

                match interp.eval(trimmed) {
                    Ok(val) => {
                        if !is_silent_result(&val) {
                            println!("{}", format_repl_result(&val));
                        }
                    }
                    Err(e) => eprint!("{}", sheaf_compiler::core::error_format::format_error(&e)),
                }

                rl.helper_mut().unwrap().refresh(&interp);
            }
            Err(ReadlineError::Interrupted) => {
                eprintln!("^C (use :quit to exit)");
                continue;
            }
            Err(ReadlineError::Eof) => {
                eprintln!("\nBye!");
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

/// Handle a REPL command. Returns true if the REPL should exit.
fn handle_command(input: &str, interp: &mut Interpreter) -> bool {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        ":quit" | ":q" => return true,

        ":help" | ":h" => {
            if arg.is_empty() {
                print_general_help();
            } else if !doc::doc_show(arg) {
                eprintln!("No help found for '{}'.", arg);
            }
        }

        ":?" => {
            if arg.is_empty() {
                println!("Usage: :? <symbol>");
            } else if !doc::doc_show(arg) {
                eprintln!("No help found for '{}'.", arg);
            }
        }

        ":list" => {
            doc::doc_list();
        }

        ":env" => {
            let reg_names = interp.registry_names();
            let var_names = interp.env().all_names();
            if !reg_names.is_empty() {
                println!("Functions:");
                for name in &reg_names {
                    println!("  {}", name);
                }
            }
            let user_vars: Vec<&String> = var_names
                .iter()
                .filter(|n| {
                    if let Ok(val) = interp.env().get(n) {
                        !matches!(val, Value::BuiltinFn { .. })
                    } else {
                        false
                    }
                })
                .collect();
            if !user_vars.is_empty() {
                println!("Variables:");
                for name in &user_vars {
                    if let Ok(val) = interp.env().get(name) {
                        println!("  {} = {}", name, format_env_value(&val));
                    }
                }
            }
            if reg_names.is_empty() && user_vars.is_empty() {
                println!("(empty environment)");
            }
        }

        ":registry" | ":reg" => {
            let names = interp.registry_names();
            if names.is_empty() {
                println!("(no user-defined functions)");
            } else {
                for name in &names {
                    println!("  {}", name);
                }
            }
        }

        ":show" => {
            if arg.is_empty() {
                println!("Usage: :show <name>");
            } else if let Some(fdef) = interp.registry_get(arg) {
                // Registered defn (stdlib or user-defined): show its source.
                println!("{}\n", pretty::format_defn(&fdef.name, &fdef.params, &fdef.body));
            } else if let Ok(val) = interp.env().get(arg) {
                println!("{}", val);
            } else {
                eprintln!("Unknown symbol: {}", arg);
            }
        }

        ":clear" => {
            print!("\x1b[2J\x1b[H");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }

        _ => {
            eprintln!("Unknown command: {}", cmd);
        }
    }

    false
}

fn format_env_value(val: &Value) -> String {
    match val {
        Value::Function { params, .. } => format!("<fn/{}>", params.len()),
        Value::BuiltinFn { name, .. } => format!("<builtin:{}>", name),
        Value::Tensor { data, .. } => {
            let shape: Vec<usize> = data.shape().to_vec();
            let shape_str = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
            format!("f64[{}]", shape_str)
        }
        Value::Dict(map) => format!("dict({})", map.len()),
        Value::List(items) => format!("list({})", items.len()),
        other => format!("{}", other),
    }
}

fn print_general_help() {
    println!("Sheaf console usage:
  :help, :h              Show this help
  :help, :h <name>       Help for a specific function or form
  :list                  List all built-in functions by category
  :env                   List all functions and variables
  :registry, :reg        List user-defined functions
  :show <name>           Show a variable's value or a function's source
  :clear                 Clear screen
  :quit, :q              Exit");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use std::sync::Arc;

    #[test]
    fn completion_respects_operation_visibility() {
        let mut interp = Interpreter::new();
        for &op in OpId::ALL {
            for name in std::iter::once(op.name()).chain(op.aliases().iter().copied()) {
                interp.env_mut().set(name, Value::Nil);
            }
        }
        for name in ["user-value", "__user-value", "@-user-value"] {
            interp.env_mut().set(name, Value::Nil);
        }
        let mut helper = SheafHelper::new();
        helper.refresh(&interp);

        for &op in OpId::ALL {
            for name in std::iter::once(op.name()).chain(op.aliases().iter().copied()) {
                assert_eq!(
                    helper.names.iter().any(|candidate| candidate == name),
                    op.is_public(),
                    "unexpected completion visibility for {name}",
                );
            }
        }
        for name in ["user-value", "__user-value", "@-user-value"] {
            assert!(helper.names.iter().any(|candidate| candidate == name));
        }
    }

    #[test]
    fn repl_prefix_is_applied_to_every_output_line() {
        let tensor = Value::Tensor {
            data: Arc::new(ArrayD::from_shape_vec(
                IxDyn(&[2, 2]),
                vec![1.0, 2.0, 3.0, 4.0],
            ).unwrap()),
            dtype: sheaf_compiler::interpreter::value::Dtype::F32,
        };

        assert_eq!(
            format_repl_result(&tensor),
            "=> tensor f32[2x2] = [[1. 2.]\n                      [3. 4.]]",
        );
    }
}
