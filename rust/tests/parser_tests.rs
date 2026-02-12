// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Integration tests for the Sheaf parser

use sheaf_compiler::{SheafValue, parse};

#[test]
fn test_parse_defn() {
    let source = r#"
(defn add [x y]
  (+ x y))
"#;
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 1);

    // Check it's a list starting with 'defn'
    match &exprs[0] {
        SheafValue::List(elems, _) => {
            assert!(elems[0].is_symbol("defn"));
            assert!(elems[1].is_symbol("add"));
            assert!(elems[2].is_vector());
        }
        _ => panic!("Expected list"),
    }
}

#[test]
fn test_parse_let() {
    let source = r#"
(let [x 1
      y 2]
  (+ x y))
"#;
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 1);

    match &exprs[0] {
        SheafValue::List(elems, _) => {
            assert!(elems[0].is_symbol("let"));
            assert!(elems[1].is_vector());
        }
        _ => panic!("Expected list"),
    }
}

#[test]
fn test_parse_dict() {
    let source = "{:a 1 :b 2 :c 3}";
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 1);

    match &exprs[0] {
        SheafValue::Dict(pairs, _) => {
            assert_eq!(pairs.len(), 3);
        }
        _ => panic!("Expected dict"),
    }
}

#[test]
fn test_parse_nested_expr() {
    let source = "(* (+ 1 2) (- 4 3))";
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 1);

    match &exprs[0] {
        SheafValue::List(elems, _) => {
            assert!(elems[0].is_symbol("*"));
            assert!(elems[1].is_list());
            assert!(elems[2].is_list());
        }
        _ => panic!("Expected list"),
    }
}

#[test]
fn test_parse_f_string() {
    let source = r#"(print "Epoch {:>3} | Loss: {:.6f}" epoch loss)"#;
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 1);

    match &exprs[0] {
        SheafValue::List(elems, _) => {
            assert!(elems[0].is_symbol("print"));
            match &elems[1] {
                SheafValue::String(s, _) => {
                    assert!(s.contains("{:>3}"));
                    assert!(s.contains("{:.6f}"));
                }
                _ => panic!("Expected string"),
            }
        }
        _ => panic!("Expected list"),
    }
}

#[test]
fn test_parse_quasiquote() {
    let source = "`(foo ~x ~@rest)";
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 1);

    match &exprs[0] {
        SheafValue::Quasiquote(inner, _) => match &**inner {
            SheafValue::List(elems, _) => {
                assert!(elems[0].is_symbol("foo"));
                assert!(matches!(elems[1], SheafValue::Unquote(_, _)));
                assert!(matches!(elems[2], SheafValue::UnquoteSplicing(_, _)));
            }
            _ => panic!("Expected list inside quasiquote"),
        },
        _ => panic!("Expected quasiquote"),
    }
}

#[test]
fn test_parse_numbers() {
    let source = "42 3.14 -17 -2.5";
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 4);

    assert!(matches!(exprs[0], SheafValue::Integer(42, _)));
    assert!(matches!(exprs[1], SheafValue::Float(_, _)));
    assert!(matches!(exprs[2], SheafValue::Integer(-17, _)));
    assert!(matches!(exprs[3], SheafValue::Float(_, _)));
}

#[test]
fn test_parse_booleans() {
    let source = "true false";
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 2);

    assert!(matches!(exprs[0], SheafValue::Boolean(true, _)));
    assert!(matches!(exprs[1], SheafValue::Boolean(false, _)));
}

#[test]
fn test_parse_nil() {
    let source = "nil";
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 1);

    assert!(matches!(exprs[0], SheafValue::Nil(_)));
}

#[test]
fn test_parse_multiple_exprs() {
    let source = r#"
(defn foo [x] x)
(defn bar [y] (* y 2))
(bar (foo 5))
"#;
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(exprs.len(), 3);
}

#[test]
fn test_parse_error_unclosed_list() {
    let source = "(+ 1 2";
    let result = parse(source, "<test>");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Unclosed"));
}

#[test]
fn test_parse_error_unexpected_closing() {
    let source = ")";
    let result = parse(source, "<test>");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Unexpected closing"));
}

#[test]
fn test_parse_error_unterminated_string() {
    let source = r#""hello world"#;
    let result = parse(source, "<test>");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Unterminated string"));
}

#[test]
fn test_display_round_trip() {
    let source = "(+ 1 2)";
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    assert_eq!(format!("{}", exprs[0]), "(+ 1 2)");
}

#[test]
fn test_parse_commas_as_whitespace() {
    let source = "[1, 2, 3]";
    let result = parse(source, "<test>");
    assert!(result.is_ok());
    let exprs = result.unwrap();
    match &exprs[0] {
        SheafValue::Vector(elems, _) => {
            assert_eq!(elems.len(), 3);
        }
        _ => panic!("Expected vector"),
    }
}
