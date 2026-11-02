// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Deterministic serialization for the build-time prelude image.

use crate::core::ast::SheafValue;
use crate::core::expr::{CompilerContext, FunctionDef, ParamLayout};
use crate::core::macro_engine::{MacroDef, MacroEngine};
use bincode::Options;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const MAGIC: &[u8; 8] = b"SHFPREL\0";
pub const SCHEMA_VERSION: u32 = 3;

#[cfg(not(any(sheaf_frontend, cargo_source_prelude)))]
mod embedded {
    include!(concat!(env!("PRELUDE_OUT_DIR"), "/metadata.rs"));

    pub const IMAGE: &[u8] = include_bytes!(concat!(env!("PRELUDE_OUT_DIR"), "/prelude.bin"));

    pub fn frontend_identity() -> &'static str {
        PRELUDE_FRONTEND_IDENTITY
    }

    pub fn source_hash() -> &'static str {
        PRELUDE_SOURCE_HASH
    }
}
const MAX_FIELD_LENGTH: usize = 16 * 1024;
const MAX_PAYLOAD_LENGTH: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreludeIdentity {
    pub compiler_version: String,
    pub frontend_identity: String,
    pub source_hash: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PreludePayload {
    env: BTreeMap<String, SheafValue>,
    registry: BTreeMap<String, FunctionDef>,
    local_vars: BTreeMap<String, SheafValue>,
    global_names: BTreeSet<String>,
    param_types: BTreeMap<String, ParamLayout>,
    param_scope: BTreeMap<String, (String, Vec<usize>)>,
    macros: BTreeMap<String, MacroDef>,
    prelude_modules: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreludeError(String);

impl PreludeError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for PreludeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for PreludeError {}

pub fn content_identity<'a>(entries: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> String {
    let mut first = 0xcbf29ce484222325_u64;
    let mut second = 0x84222325cbf29ce4_u64;
    for (name, bytes) in entries {
        update_hash(&mut first, name.as_bytes());
        update_hash(&mut first, &[0]);
        update_hash(&mut first, bytes);
        update_hash(&mut second, bytes);
        update_hash(&mut second, &[0]);
        update_hash(&mut second, name.as_bytes());
    }
    format!("{first:016x}{second:016x}")
}

pub fn encode(
    context: &CompilerContext,
    identity: &PreludeIdentity,
) -> Result<Vec<u8>, PreludeError> {
    let payload = PreludePayload {
        env: context.env.clone().into_iter().collect(),
        registry: context.registry.clone().into_iter().collect(),
        local_vars: context.local_vars.clone().into_iter().collect(),
        global_names: context.global_names.clone().into_iter().collect(),
        param_types: context.param_types.clone().into_iter().collect(),
        param_scope: context.param_scope.clone().into_iter().collect(),
        macros: context.macro_engine.macros.clone().into_iter().collect(),
        prelude_modules: context.prelude_modules.clone().into_iter().collect(),
    };
    let payload = codec().serialize(&payload)
        .map_err(|error| PreludeError::new(format!("cannot encode prelude payload: {error}")))?;
    if payload.len() > MAX_PAYLOAD_LENGTH {
        return Err(PreludeError::new("prelude payload exceeds size limit"));
    }

    let mut output = Vec::with_capacity(payload.len() + 256);
    output.extend_from_slice(MAGIC);
    write_u32(&mut output, SCHEMA_VERSION);
    write_string(&mut output, &identity.compiler_version)?;
    write_string(&mut output, &identity.frontend_identity)?;
    write_string(&mut output, &identity.source_hash)?;
    write_u64(&mut output, payload.len() as u64);
    write_u64(&mut output, checksum(&payload));
    output.extend_from_slice(&payload);
    Ok(output)
}

#[cfg(not(any(sheaf_frontend, cargo_source_prelude)))]
pub fn install_embedded(
    context: &mut CompilerContext,
    compiler_version: &str,
    source_hash: &str,
) -> Result<(), PreludeError> {
    if source_hash != embedded::source_hash() {
        return Err(PreludeError::new(
            "embedded standard-library sources do not match the prelude image",
        ));
    }
    let expected = PreludeIdentity {
        compiler_version: compiler_version.to_string(),
        frontend_identity: embedded::frontend_identity().to_string(),
        source_hash: source_hash.to_string(),
    };
    decode_and_install(context, embedded::IMAGE, &expected)
}

pub fn decode_and_install(
    context: &mut CompilerContext,
    bytes: &[u8],
    expected: &PreludeIdentity,
) -> Result<(), PreludeError> {
    let payload = decode(bytes, expected)?;
    context.env = payload.env.into_iter().collect();
    context.registry = payload.registry.into_iter().collect();
    context.local_vars = payload.local_vars.into_iter().collect();
    context.global_names = payload.global_names.into_iter().collect();
    context.param_types = payload.param_types.into_iter().collect();
    context.param_scope = payload.param_scope.into_iter().collect();
    context.macro_engine = MacroEngine::new();
    context.macro_engine.macros = payload.macros.into_iter().collect();
    context.prelude_modules = payload.prelude_modules.into_iter().collect();
    Ok(())
}

fn decode(bytes: &[u8], expected: &PreludeIdentity) -> Result<PreludePayload, PreludeError> {
    let mut reader = Reader::new(bytes);
    if reader.take(MAGIC.len())? != MAGIC {
        return Err(PreludeError::new("invalid prelude magic"));
    }
    let schema = reader.read_u32()?;
    if schema != SCHEMA_VERSION {
        return Err(PreludeError::new(format!(
            "unsupported prelude schema {schema}, expected {SCHEMA_VERSION}"
        )));
    }
    compare_identity("compiler version", &reader.read_string()?, &expected.compiler_version)?;
    compare_identity("frontend identity", &reader.read_string()?, &expected.frontend_identity)?;
    compare_identity("standard-library source hash", &reader.read_string()?, &expected.source_hash)?;

    let payload_length = reader.read_u64()? as usize;
    if payload_length > MAX_PAYLOAD_LENGTH {
        return Err(PreludeError::new("prelude payload exceeds size limit"));
    }
    let expected_checksum = reader.read_u64()?;
    let payload = reader.take(payload_length)?;
    if !reader.is_empty() {
        return Err(PreludeError::new("trailing bytes after prelude payload"));
    }
    if checksum(payload) != expected_checksum {
        return Err(PreludeError::new("prelude payload checksum mismatch"));
    }
    codec().deserialize(payload)
        .map_err(|error| PreludeError::new(format!("cannot decode prelude payload: {error}")))
}

fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_PAYLOAD_LENGTH as u64)
}

fn compare_identity(field: &str, actual: &str, expected: &str) -> Result<(), PreludeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PreludeError::new(format!(
            "prelude {field} mismatch: image has '{actual}', expected '{expected}'"
        )))
    }
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<(), PreludeError> {
    if value.len() > MAX_FIELD_LENGTH {
        return Err(PreludeError::new("prelude identity field exceeds size limit"));
    }
    write_u32(output, value.len() as u32);
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    update_hash(&mut hash, bytes);
    hash
}

fn update_hash(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], PreludeError> {
        let end = self.position.checked_add(length)
            .ok_or_else(|| PreludeError::new("prelude length overflow"))?;
        let value = self.bytes.get(self.position..end)
            .ok_or_else(|| PreludeError::new("truncated prelude image"))?;
        self.position = end;
        Ok(value)
    }

    fn read_u32(&mut self) -> Result<u32, PreludeError> {
        let bytes: [u8; 4] = self.take(4)?.try_into()
            .map_err(|_| PreludeError::new("truncated prelude u32"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, PreludeError> {
        let bytes: [u8; 8] = self.take(8)?.try_into()
            .map_err(|_| PreludeError::new("truncated prelude u64"))?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_string(&mut self) -> Result<String, PreludeError> {
        let length = self.read_u32()? as usize;
        if length > MAX_FIELD_LENGTH {
            return Err(PreludeError::new("prelude identity field exceeds size limit"));
        }
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| PreludeError::new("prelude identity is not UTF-8"))
    }

    fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(cargo_source_prelude))]
    use crate::core::parser::{parser_invocations, reset_parser_invocations};

    fn identity() -> PreludeIdentity {
        PreludeIdentity {
            compiler_version: env!("SHEAF_COMPILER_VERSION").to_string(),
            frontend_identity: "frontend-test".to_string(),
            source_hash: "source-test".to_string(),
        }
    }

    fn source_context() -> CompilerContext {
        let mut context = CompilerContext::new_without_prelude();
        context.load_prelude_from_source().unwrap();
        context
    }

    #[test]
    fn encoding_is_deterministic_and_round_trips() {
        let source = source_context();
        let first = encode(&source, &identity()).unwrap();
        let second = encode(&source, &identity()).unwrap();
        assert_eq!(first, second);

        let mut decoded = CompilerContext::new_without_prelude();
        decode_and_install(&mut decoded, &first, &identity()).unwrap();
        assert_eq!(encode(&decoded, &identity()).unwrap(), first);
        assert!(decoded.registry.contains_key("layer-norm"));
        assert!(decoded.registry.contains_key("adamw-step"));
        assert!(decoded.macro_engine.macros.contains_key("defmodel"));
    }

    #[test]
    fn source_and_image_install_equivalent_registries() {
        let source = source_context();
        let image = CompilerContext::new();
        assert_eq!(
            encode(&source, &identity()).unwrap(),
            encode(&image, &identity()).unwrap(),
        );
    }

    #[cfg(not(cargo_source_prelude))]
    #[test]
    fn production_install_does_not_invoke_the_parser() {
        reset_parser_invocations();
        let context = CompilerContext::new();
        assert_eq!(parser_invocations(), 0);
        assert!(context.registry.contains_key("softmax"));
    }

    #[test]
    fn image_install_registers_sources_for_diagnostics() {
        use crate::core::error::{SheafError, SourceLocation};
        use std::rc::Rc;

        let source = include_str!("../../lib/nn.shf");
        let line = source.lines().position(|line| line.starts_with("(defn relu"))
            .unwrap() + 1;
        let _context = CompilerContext::new();
        let error = SheafError::Compile {
            message: "diagnostic probe".to_string(),
            location: SourceLocation::new(line, 1, Rc::from("nn.shf")),
        };
        let formatted = crate::core::error_format::format_error(&error);
        assert!(formatted.contains("(defn relu"));
    }

    #[test]
    fn source_content_changes_its_identity() {
        assert_ne!(
            content_identity([("nn.shf", b"first".as_slice())]),
            content_identity([("nn.shf", b"second".as_slice())]),
        );
    }

    #[test]
    fn identity_changes_invalidate_the_image() {
        let bytes = encode(&source_context(), &identity()).unwrap();
        let mut context = CompilerContext::new_without_prelude();

        let mut changed = identity();
        changed.frontend_identity.push_str("-changed");
        assert!(decode_and_install(&mut context, &bytes, &changed).is_err());

        let mut changed = identity();
        changed.source_hash.push_str("-changed");
        assert_ne!(encode(&source_context(), &changed).unwrap(), bytes);
        assert!(decode_and_install(&mut context, &bytes, &changed).is_err());

        let mut changed = identity();
        changed.compiler_version.push_str("-changed");
        assert!(decode_and_install(&mut context, &bytes, &changed).is_err());
    }

    #[test]
    fn rejects_corrupt_truncated_and_trailing_images() {
        let bytes = encode(&source_context(), &identity()).unwrap();
        for length in [0, 1, MAGIC.len(), bytes.len() - 1] {
            let mut context = CompilerContext::new_without_prelude();
            assert!(decode_and_install(&mut context, &bytes[..length], &identity()).is_err());
        }

        let mut wrong_schema = bytes.clone();
        wrong_schema[MAGIC.len()] = (SCHEMA_VERSION + 1) as u8;
        let mut context = CompilerContext::new_without_prelude();
        assert!(decode_and_install(&mut context, &wrong_schema, &identity()).is_err());

        let mut corrupt = bytes.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xff;
        let mut context = CompilerContext::new_without_prelude();
        assert!(decode_and_install(&mut context, &corrupt, &identity()).is_err());

        let mut trailing = bytes;
        trailing.push(0);
        let mut context = CompilerContext::new_without_prelude();
        assert!(decode_and_install(&mut context, &trailing, &identity()).is_err());
    }
}
