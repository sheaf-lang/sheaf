// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! StableHLO primitives for the shared PRNG.

use crate::core::prng::{self, FloatOp, Primitives, UnaryOp, WordOp};
use super::{Register, StableHLOEmitter, StableHLOType};

struct HloPrng<'a> {
    emitter: &'a mut StableHLOEmitter,
    word_ty: StableHLOType,
    float_ty: StableHLOType,
}

impl<'a> HloPrng<'a> {
    fn new(emitter: &'a mut StableHLOEmitter, shape: &[i64]) -> Self {
        Self {
            emitter,
            word_ty: StableHLOType::i32_tensor(shape.to_vec()),
            float_ty: StableHLOType::f32_tensor(shape.to_vec()),
        }
    }

    fn binary(&mut self, op: &str, a: Register, b: Register, integer: bool) -> Register {
        let ty = if integer { &self.word_ty } else { &self.float_ty };
        let result = self.emitter.fresh_register();
        self.emitter.body.push(format!(
            "    {} = stablehlo.{} {}, {} : {}",
            result.to_mlir(), op, a.to_mlir(), b.to_mlir(), ty.to_mlir(),
        ));
        result
    }
}

impl Primitives for HloPrng<'_> {
    type Word = Register;
    type Float = Register;
    fn word(&mut self, value: u32) -> Register {
        let result = self.emitter.fresh_register();
        self.emitter.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            result.to_mlir(), value as i32, self.word_ty.to_mlir(),
        ));
        result
    }
    fn float(&mut self, value: f32) -> Register {
        self.emitter.emit_typed_splat_constant(
            value as f64, self.float_ty.shape(), crate::core::dtype::ElementType::F32,
        ).0
    }
    fn word_op(&mut self, op: WordOp, a: Register, b: Register) -> Register {
        let op = match op {
            WordOp::Add => "add", WordOp::Xor => "xor", WordOp::And => "and",
            WordOp::Or => "or", WordOp::ShiftLeft => "shift_left",
            WordOp::ShiftRight => "shift_right_logical",
        };
        self.binary(op, a, b, true)
    }
    fn float_op(&mut self, op: FloatOp, a: Register, b: Register) -> Register {
        let op = match op {
            FloatOp::Add => "add", FloatOp::Multiply => "multiply",
            FloatOp::Minimum => "minimum", FloatOp::Maximum => "maximum",
        };
        self.binary(op, a, b, false)
    }
    fn unary(&mut self, op: UnaryOp, value: Register) -> Register {
        let op = match op {
            UnaryOp::Log => "log", UnaryOp::Sqrt => "sqrt",
            UnaryOp::Cos => "cosine", UnaryOp::Floor => "floor",
        };
        let result = self.emitter.fresh_register();
        self.emitter.body.push(format!(
            "    {} = stablehlo.{} {} : {}",
            result.to_mlir(), op, value.to_mlir(), self.float_ty.to_mlir(),
        ));
        result
    }
    fn to_float(&mut self, value: Register) -> Register {
        self.emitter.emit_convert(&value, &self.word_ty, &self.float_ty)
    }
    fn to_word(&mut self, value: Register) -> Register {
        self.emitter.emit_convert(&value, &self.float_ty, &self.word_ty)
    }
}

impl StableHLOEmitter {
    pub fn emit_random_key(&mut self, seed: i64) -> (Register, StableHLOType) {
        let limbs = prng::encode(&mut prng::Host, prng::seed(seed));
        self.emit_nd_tensor_constant(
            &limbs.map(|x| x as f64), &[prng::KEY_SIZE as i64],
        )
    }

    fn decode_random_key(&mut self, key: &Register, key_ty: &StableHLOType) -> [Register; 2] {
        let limbs = std::array::from_fn(|i| self.emit_index_axis0(key, key_ty, i as i64).0);
        prng::decode(&mut HloPrng::new(self, &[]), limbs)
    }

    fn encode_random_key(&mut self, words: [Register; 2]) -> (Register, StableHLOType) {
        let limbs = prng::encode(&mut HloPrng::new(self, &[]), words);
        let scalar = StableHLOType::scalar_f32();
        let limbs = limbs.map(|x| self.emit_reshape(&x, &scalar, &[1]).0);
        self.emit_concatenate(&limbs, &vec![StableHLOType::f32_tensor(vec![1]); prng::KEY_SIZE], 0)
    }

    fn random_bits(
        &mut self, key: &Register, key_ty: &StableHLOType, total: i64,
    ) -> [Register; 2] {
        let words = self.decode_random_key(key, key_ty);
        let word_ty = StableHLOType::i32_tensor(vec![total]);
        let scalar = StableHLOType::i32_tensor(vec![]);
        let words = words.map(|word| self.emit_broadcast(&word, &scalar, &word_ty));
        let index = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 0 : {}", index.to_mlir(), word_ty.to_mlir(),
        ));
        prng::bits(&mut HloPrng::new(self, &[total]), words, index)
    }

    pub fn emit_random_uniform(
        &mut self, key: &Register, key_ty: &StableHLOType, shape: &[i64],
    ) -> (Register, StableHLOType) {
        let total = shape.iter().product();
        let bits = self.random_bits(key, key_ty, total);
        let value = prng::uniform(&mut HloPrng::new(self, &[total]), bits[0]);
        self.emit_reshape(&value, &StableHLOType::f32_tensor(vec![total]), shape)
    }

    pub fn emit_random_normal(
        &mut self, key: &Register, key_ty: &StableHLOType, shape: &[i64],
    ) -> (Register, StableHLOType) {
        let total = shape.iter().product();
        let bits = self.random_bits(key, key_ty, total);
        let value = prng::normal(&mut HloPrng::new(self, &[total]), bits);
        self.emit_reshape(&value, &StableHLOType::f32_tensor(vec![total]), shape)
    }

    pub fn emit_random_randint(
        &mut self, key: &Register, key_ty: &StableHLOType, shape: &[i64], low: i64, high: i64,
    ) -> (Register, StableHLOType) {
        let total = shape.iter().product();
        let bits = self.random_bits(key, key_ty, total);
        let value = prng::randint(&mut HloPrng::new(self, &[total]), bits[0], low, high);
        self.emit_reshape(&value, &StableHLOType::f32_tensor(vec![total]), shape)
    }

    pub fn emit_random_split_n(
        &mut self, key: &Register, key_ty: &StableHLOType, n: usize,
    ) -> (Register, StableHLOType) {
        let words = self.decode_random_key(key, key_ty);
        let mut keys = Vec::with_capacity(n);
        let mut types = Vec::with_capacity(n);
        for i in 0..n {
            let mut primitives = HloPrng::new(self, &[]);
            let index = primitives.word(i as u32);
            let child = prng::split(&mut primitives, words, index);
            let (key, ty) = self.encode_random_key(child);
            keys.push(key);
            types.push(ty);
        }
        self.emit_tuple(&keys, &types)
    }

    pub fn emit_choice(
        &mut self, key: &Register, key_ty: &StableHLOType, probs: &Register, probs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let k = probs_ty.shape()[0];
        let (u, u_ty) = self.emit_random_uniform(key, key_ty, &[]);
        let (ones, ones_ty) = self.emit_ones(&[k, k]);
        let (tril, tril_ty) = self.emit_tril(&ones, &ones_ty);
        let (cumsum, cumsum_ty) = self.emit_matmul(&tril, probs, &tril_ty, probs_ty);
        let (mask, _) = self.emit_compare(
            ">", &cumsum, &u, &cumsum_ty, &u_ty,
        );
        let (count, count_ty) = self.emit_reduce_sum(&mask, &cumsum_ty, 0, false);
        let k_reg = self.emit_constant_f32(k as f64);
        let (index, ty) = self.emit_binop("-", &k_reg, &count, &u_ty, &count_ty);
        let last = self.emit_constant_f32((k - 1) as f64);
        self.emit_binop("min", &index, &last, &ty, &u_ty)
    }
}
