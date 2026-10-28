// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Constants.

use crate::core::dtype::ElementType;

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    pub fn emit_typed_scalar_constant(
        &mut self,
        value: f64,
        dtype: ElementType,
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::tensor(Vec::new(), dtype);
        let value = format_float_constant(value, dtype);
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value,
            ty.to_mlir(),
        ));
        (reg, ty)
    }

    pub fn emit_typed_splat_constant(
        &mut self,
        value: f64,
        shape: &[i64],
        dtype: ElementType,
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::tensor(shape.to_vec(), dtype);
        let value_str = format_float_constant(value, dtype);
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value_str,
            ty.to_mlir(),
        ));
        if shape.is_empty() {
            self.known_scalars.insert(reg, value);
        }
        (reg, ty)
    }

    pub fn emit_constant_f32(&mut self, value: f64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::scalar_f32();
        let value_str = if value.fract() == 0.0 && value.is_finite() {
            format!("{:.1}", value)
        } else {
            format!("{}", value)
        };
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value_str,
            ty.to_mlir()
        ));
        self.known_scalars.insert(reg, value);
        reg
    }

    pub fn emit_constant_i32(&mut self, value: i64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::typed_tensor(vec![], "i32");
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value,
            ty.to_mlir()
        ));
        self.known_scalars.insert(reg, value as f64);
        reg
    }

    pub fn emit_constant_i64(&mut self, value: i64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::ScalarI64;
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value,
            ty.to_mlir()
        ));
        reg
    }

    pub fn emit_tensor_constant(&mut self, values: &[Vec<f64>]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        let rows = values.len();
        let cols = if rows > 0 { values[0].len() } else { 0 };
        let shape = vec![rows as i64, cols as i64];
        let ty = StableHLOType::f32_tensor(shape);

        let rows_str: Vec<String> = values
            .iter()
            .map(|row| {
                let row_values: Vec<String> = row
                    .iter()
                    .map(|&v| {
                        if v.fract() == 0.0 && v.is_finite() {
                            format!("{:.1}", v)
                        } else {
                            format!("{}", v)
                        }
                    })
                    .collect();
                format!("[{}]", row_values.join(", "))
            })
            .collect();

        let values_str = rows_str.join(", ");

        self.body.push(format!(
            "    {} = stablehlo.constant dense<[{}]> : {}",
            reg.to_mlir(),
            values_str,
            ty.to_mlir()
        ));

        (reg, ty)
    }

    pub fn emit_integer_vector_constant(&mut self, values: &[i64]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::i32_tensor(vec![values.len() as i64]);
        let values = values.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        self.body.push(format!(
            "    {} = stablehlo.constant dense<[{}]> : {}", reg.to_mlir(), values, ty.to_mlir(),
        ));
        (reg, ty)
    }

    pub fn emit_nd_tensor_constant(
        &mut self,
        data: &[f64],
        shape: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        let values_str = if shape.is_empty() {
            format_f64(data[0])
        } else {
            format_dense_attr(data, shape, 0)
        };

        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            values_str,
            ty.to_mlir()
        ));

        (reg, ty)
    }
}

fn format_float_constant(value: f64, dtype: ElementType) -> String {
    if dtype.is_integer() || dtype == ElementType::Bool {
        return (value as i64).to_string();
    }
    if value.is_infinite() {
        return match (value.is_sign_negative(), dtype) {
            (false, ElementType::F16) => "0x7C00".to_string(),
            (true, ElementType::F16) => "0xFC00".to_string(),
            (false, ElementType::BF16) => "0x7F80".to_string(),
            (true, ElementType::BF16) => "0xFF80".to_string(),
            (false, ElementType::F32) => "0x7F800000".to_string(),
            (true, ElementType::F32) => "0xFF800000".to_string(),
            _ => value.to_string(),
        };
    }
    format_f64(value)
}

pub(super) fn format_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

pub(super) fn format_dense_attr(data: &[f64], shape: &[i64], dim: usize) -> String {
    if dim == shape.len() - 1 {
        let n = shape[dim] as usize;
        let vals: Vec<String> = data[..n].iter().map(|&v| format_f64(v)).collect();
        format!("[{}]", vals.join(", "))
    } else {
        let stride: usize = shape[dim + 1..].iter().map(|&d| d as usize).product();
        let n = shape[dim] as usize;
        let subs: Vec<String> = (0..n)
            .map(|i| format_dense_attr(&data[i * stride..], shape, dim + 1))
            .collect();
        format!("[{}]", subs.join(", "))
    }
}
