// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Indexing operations.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    pub fn emit_index_axis0(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        index: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        assert!(!shape.is_empty(), "Cannot index a scalar");

        let ndim = shape.len();

        let mut start = vec![0i64; ndim];
        let mut limit = shape.to_vec();
        let strides = vec![1i64; ndim];
        start[0] = index;
        limit[0] = index + 1;

        let dims_str = format_slice_dims(&start, &limit, &strides);

        let slice_reg = self.fresh_register();
        let mut slice_shape = shape.to_vec();
        slice_shape[0] = 1;
        let slice_ty = indexing_type(input_ty, slice_shape.clone());
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            slice_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            slice_ty.to_mlir(),
        ));

        let result_shape: Vec<i64> = shape[1..].to_vec();
        if result_shape.is_empty() {
            let result_ty = indexing_type(input_ty, Vec::new());
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                result_reg.to_mlir(),
                slice_reg.to_mlir(),
                slice_ty.to_mlir(),
                result_ty.to_mlir(),
            ));
            (result_reg, result_ty)
        } else {
            let result_ty = indexing_type(input_ty, result_shape);
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                result_reg.to_mlir(),
                slice_reg.to_mlir(),
                slice_ty.to_mlir(),
                result_ty.to_mlir(),
            ));
            (result_reg, result_ty)
        }
    }

    pub fn emit_slice_last_axis(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut start_indices = vec![0i64; ndim];
        let mut limit_indices = shape.to_vec();
        let strides = vec![1i64; ndim];
        start_indices[ndim - 1] = start;
        limit_indices[ndim - 1] = end;

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut slice_shape = shape.to_vec();
        slice_shape[ndim - 1] = end - start;
        let slice_ty = indexing_type(input_ty, slice_shape.clone());

        let slice_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            slice_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            slice_ty.to_mlir(),
        ));

        if end - start == 1 {
            let result_shape: Vec<i64> = shape[..ndim - 1].to_vec();
            if result_shape.is_empty() {
                let result_ty = indexing_type(input_ty, Vec::new());
                let result_reg = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.reshape {} : ({}) -> {}",
                    result_reg.to_mlir(),
                    slice_reg.to_mlir(),
                    slice_ty.to_mlir(),
                    result_ty.to_mlir(),
                ));
                (result_reg, result_ty)
            } else {
                let result_ty = indexing_type(input_ty, result_shape);
                let result_reg = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.reshape {} : ({}) -> {}",
                    result_reg.to_mlir(),
                    slice_reg.to_mlir(),
                    slice_ty.to_mlir(),
                    result_ty.to_mlir(),
                ));
                (result_reg, result_ty)
            }
        } else {
            (slice_reg, slice_ty)
        }
    }

    pub fn emit_slice_range(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut start_indices = vec![0i64; ndim];
        let mut limit_indices = shape.to_vec();
        let strides = vec![1i64; ndim];
        start_indices[0] = start;
        limit_indices[0] = end + 1; // inclusive end

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut result_shape = shape.to_vec();
        result_shape[0] = end + 1 - start;
        let result_ty = indexing_type(input_ty, result_shape);

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        (result_reg, result_ty)
    }

    pub fn emit_roll(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        shift: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let n = shape[0];
        let shift = ((shift % n) + n) % n; // normalize to [0, n)

        if shift == 0 {
            return (*input, input_ty.clone());
        }

        let split = n - shift;

        let (tail, tail_ty) = self.emit_slice_range(input, input_ty, split, n - 1);
        let (head, _head_ty) = self.emit_slice_range(input, input_ty, 0, split - 1);

        self.emit_concatenate(&[tail, head], &[tail_ty.clone(), input_ty.clone()], 0)
    }

    pub fn emit_reverse(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len() as i64;
        let ax = if axis < 0 { (ndim + axis) as usize } else { axis as usize };

        let reg = self.fresh_register();
        let dims_str = format!("{}", ax);
        self.body.push(format!(
            " {} = stablehlo.reverse {}, dims = [{}] : ({}) -> {}",
            reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            input_ty.to_mlir(),
        ));
        (reg, input_ty.clone())
    }

    pub fn emit_index_update(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        index: i64,
        value: &Register,
        value_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut update_shape = vec![1i64];
        update_shape.extend_from_slice(value_ty.shape());
        if value_ty.shape().is_empty() && ndim == 1 {
            let update_ty = indexing_type(input_ty, vec![1]);
            let update_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                update_reg.to_mlir(),
                value.to_mlir(),
                value_ty.to_mlir(),
                update_ty.to_mlir(),
            ));

            let idx_reg = self.emit_constant_i32(index);
            let scalar_i32 = StableHLOType::typed_tensor(vec![], "i32");
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.dynamic_update_slice {}, {}, {} : ({}, {}, {}) -> {}",
                result_reg.to_mlir(),
                input.to_mlir(),
                update_reg.to_mlir(),
                idx_reg.to_mlir(),
                input_ty.to_mlir(),
                update_ty.to_mlir(),
                scalar_i32.to_mlir(),
                input_ty.to_mlir(),
            ));
            (result_reg, input_ty.clone())
        } else {
            let update_ty = indexing_type(input_ty, update_shape);
            let update_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                update_reg.to_mlir(),
                value.to_mlir(),
                value_ty.to_mlir(),
                update_ty.to_mlir(),
            ));

            let idx_reg = self.emit_constant_i32(index);
            let zero_idx = self.emit_constant_i32(0);
            let scalar_i32 = StableHLOType::typed_tensor(vec![], "i32");
            let mut start_regs = vec![idx_reg.to_mlir()];
            for _ in 1..ndim {
                start_regs.push(zero_idx.to_mlir());
            }
            let start_str = start_regs.join(", ");

            let mut start_types = vec![scalar_i32.to_mlir()];
            for _ in 1..ndim {
                start_types.push(scalar_i32.to_mlir());
            }
            let start_types_str = start_types.join(", ");

            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.dynamic_update_slice {}, {}, {} : ({}, {}, {}) -> {}",
                result_reg.to_mlir(),
                input.to_mlir(),
                update_reg.to_mlir(),
                start_str,
                input_ty.to_mlir(),
                update_ty.to_mlir(),
                start_types_str,
                input_ty.to_mlir(),
            ));
            (result_reg, input_ty.clone())
        }
    }

    pub fn emit_dynamic_slice(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        starts: &[Register],
        slice_sizes: &[i64],
    ) -> (Register, StableHLOType) {
        let result_ty = indexing_type(input_ty, slice_sizes.to_vec());
        let result_reg = self.fresh_register();
        let operands = std::iter::once(input.to_mlir().to_string())
            .chain(starts.iter().map(Register::to_mlir))
            .collect::<Vec<_>>();
        let operand_types = std::iter::once(input_ty.to_mlir())
            .chain(starts.iter().map(|_| "tensor<i32>".to_string()))
            .collect::<Vec<_>>();
        let sizes = slice_sizes
            .iter()
            .map(|size| size.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        self.body.push(format!(
            "    {} = \"stablehlo.dynamic_slice\"({}) {{ slice_sizes = array<i64: {}> }} : ({}) -> {}",
            result_reg.to_mlir(),
            operands.join(", "),
            sizes,
            operand_types.join(", "),
            result_ty.to_mlir(),
        ));
        (result_reg, result_ty)
    }

    pub fn emit_dynamic_update_slice(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        update: &Register,
        update_ty: &StableHLOType,
        starts: &[Register],
    ) -> (Register, StableHLOType) {
        let result_reg = self.fresh_register();
        let operands = std::iter::once(input.to_mlir().to_string())
            .chain(std::iter::once(update.to_mlir().to_string()))
            .chain(starts.iter().map(Register::to_mlir))
            .collect::<Vec<_>>();
        let operand_types = std::iter::once(input_ty.to_mlir())
            .chain(std::iter::once(update_ty.to_mlir()))
            .chain(starts.iter().map(|_| "tensor<i32>".to_string()))
            .collect::<Vec<_>>();
        self.body.push(format!(
            "    {} = \"stablehlo.dynamic_update_slice\"({}) : ({}) -> {}",
            result_reg.to_mlir(),
            operands.join(", "),
            operand_types.join(", "),
            input_ty.to_mlir(),
        ));
        (result_reg, input_ty.clone())
    }

    pub fn emit_slice_exclusive(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
    ) -> (Register, StableHLOType) {
        self.emit_slice_axis(input, input_ty, start, end, 0)
    }

    pub fn emit_slice_axis(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
        axis: usize,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut start_indices = vec![0i64; ndim];
        let mut limit_indices = shape.to_vec();
        let strides = vec![1i64; ndim];
        start_indices[axis] = start;
        limit_indices[axis] = end;

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut result_shape = shape.to_vec();
        result_shape[axis] = end - start;
        let result_ty = StableHLOType::typed_tensor(result_shape, input_ty.dtype());

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        (result_reg, result_ty)
    }

    pub fn emit_tensor_split(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        num_sections: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let total = shape[0];
        let section_size = total / num_sections;

        let mut section_regs = Vec::new();
        let mut section_types = Vec::new();

        for i in 0..num_sections {
            let start = i * section_size;
            let end = start + section_size;
            let (reg, ty) = self.emit_slice_exclusive(input, input_ty, start, end);
            section_regs.push(reg);
            section_types.push(ty);
        }

        self.emit_tuple(&section_regs, &section_types)
    }

    pub fn emit_gather_axis0(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        indices: &Register,
        indices_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let operand_shape = operand_ty.shape();
        let indices_shape = indices_ty.shape();

        let indices_int_ty = StableHLOType::i32_tensor(indices_shape.to_vec());
        let indices_int_reg = self.emit_convert(indices, indices_ty, &indices_int_ty);
        // Keep index generation outside the gather's element loops.
        let materialized_indices = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.optimization_barrier {} : {}",
            materialized_indices.to_mlir(), indices_int_reg.to_mlir(), indices_int_ty.to_mlir(),
        ));

        let mut reshaped_shape: Vec<i64> = indices_shape.to_vec();
        reshaped_shape.push(1);
        let indices_3d_reg = self.fresh_register();
        let indices_3d_ty = StableHLOType::i32_tensor(reshaped_shape.clone());
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            indices_3d_reg.to_mlir(),
            materialized_indices.to_mlir(),
            indices_int_ty.to_mlir(),
            indices_3d_ty.to_mlir(),
        ));

        let row_shape = &operand_shape[1..];
        let mut result_shape: Vec<i64> = indices_shape.to_vec();
        result_shape.extend_from_slice(row_shape);
        let result_ty = indexing_type(operand_ty, result_shape);

        let idx_rank = indices_shape.len();
        let offset_dims: Vec<i64> = (idx_rank..idx_rank + row_shape.len())
            .map(|d| d as i64)
            .collect();

        let mut slice_sizes: Vec<i64> = vec![1];
        slice_sizes.extend_from_slice(row_shape);

        let index_vector_dim = reshaped_shape.len() - 1;

        let result_reg = self.fresh_register();
        let offset_dims_str = offset_dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
        let slice_sizes_str = slice_sizes.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");

        self.body.push(format!(
            "    {} = \"stablehlo.gather\"({}, {}) {{\n\
             \x20     dimension_numbers = #stablehlo.gather<\n\
             \x20       offset_dims = [{}],\n\
             \x20       collapsed_slice_dims = [0],\n\
             \x20       operand_batching_dims = [],\n\
             \x20       start_indices_batching_dims = [],\n\
             \x20       start_index_map = [0],\n\
             \x20       index_vector_dim = {}>,\n\
             \x20     slice_sizes = array<i64: {}>,\n\
             \x20     indices_are_sorted = false\n\
             \x20   }} : ({}, {}) -> {}",
            result_reg.to_mlir(),
            operand.to_mlir(),
            indices_3d_reg.to_mlir(),
            offset_dims_str,
            index_vector_dim,
            slice_sizes_str,
            operand_ty.to_mlir(),
            indices_3d_ty.to_mlir(),
            result_ty.to_mlir(),
        ));

        (result_reg, result_ty)
    }

    pub fn emit_top_k(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        k: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        assert!(!shape.is_empty(), "top_k requires at least 1D input");
        let last_axis = shape.len() - 1;

        let iota_reg = self.fresh_register();
        let iota_ty = StableHLOType::i32_tensor(shape.to_vec());
        self.body.push(format!(
            "    {} = stablehlo.iota dim = {} : {}",
            iota_reg.to_mlir(),
            last_axis,
            iota_ty.to_mlir()
        ));

        let sorted_vals = self.fresh_register();
        let sorted_idxs = self.fresh_register();

        let cmp_lhs = self.fresh_register();
        let cmp_rhs = self.fresh_register();
        let cmp_lhs_idx = self.fresh_register();
        let cmp_rhs_idx = self.fresh_register();
        let cmp_pred = self.fresh_register();

        self.body.push(format!(
            "    {}, {} = \"stablehlo.sort\"({}, {}) ({{\n\
             \x20   ^bb0({}: tensor<f32>, {}: tensor<f32>, {}: tensor<i32>, {}: tensor<i32>):\n\
             \x20     {} = \"stablehlo.compare\"({}, {}) {{comparison_direction = #stablehlo<comparison_direction GT>}} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n\
             \x20     \"stablehlo.return\"({}) : (tensor<i1>) -> ()\n\
             \x20   }}) {{dimension = {} : i64, is_stable = true}} : ({}, {}) -> ({}, {})",
            sorted_vals.to_mlir(),
            sorted_idxs.to_mlir(),
            input.to_mlir(),
            iota_reg.to_mlir(),
            cmp_lhs.to_mlir(),
            cmp_rhs.to_mlir(),
            cmp_lhs_idx.to_mlir(),
            cmp_rhs_idx.to_mlir(),
            cmp_pred.to_mlir(),
            cmp_lhs.to_mlir(),
            cmp_rhs.to_mlir(),
            cmp_pred.to_mlir(),
            last_axis,
            input_ty.to_mlir(),
            iota_ty.to_mlir(),
            input_ty.to_mlir(),
            iota_ty.to_mlir(),
        ));

        let (top_vals, top_vals_ty) = self.emit_slice_axis(
            &sorted_vals, input_ty, 0, k, last_axis,
        );
        let (top_idxs_i32, top_idxs_i32_ty) = self.emit_slice_axis(
            &sorted_idxs, &iota_ty, 0, k, last_axis,
        );

        let top_idxs_f32_ty = StableHLOType::f32_tensor(top_idxs_i32_ty.shape().to_vec());
        let top_idxs = self.emit_convert(&top_idxs_i32, &top_idxs_i32_ty, &top_idxs_f32_ty);

        self.emit_tuple(
            &[top_vals, top_idxs],
            &[top_vals_ty, top_idxs_f32_ty],
        )
    }
}

fn indexing_type(input: &StableHLOType, shape: Vec<i64>) -> StableHLOType {
    StableHLOType::tensor(shape, input.element_type().unwrap())
}

pub(super) fn format_slice_dims(starts: &[i64], limits: &[i64], strides: &[i64]) -> String {
    starts
        .iter()
        .zip(limits.iter())
        .zip(strides.iter())
        .map(|((s, l), st)| format!("{}:{}:{}", s, l, st))
        .collect::<Vec<_>>()
        .join(", ")
}
