// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Threefry2x32-20 and distributions shared by host evaluation and codegen.

pub const VERSION: &str = "threefry2x32-20/16bit-v1";
pub const KEY_SIZE: usize = 4;

#[derive(Clone, Copy)]
pub enum WordOp { Add, Xor, And, Or, ShiftLeft, ShiftRight }
#[derive(Clone, Copy)]
pub enum FloatOp { Add, Multiply, Minimum, Maximum }
#[derive(Clone, Copy)]
pub enum UnaryOp { Log, Sqrt, Cos, Floor }

pub trait Primitives {
    type Word: Copy;
    type Float: Copy;
    fn word(&mut self, value: u32) -> Self::Word;
    fn float(&mut self, value: f32) -> Self::Float;
    fn word_op(&mut self, op: WordOp, a: Self::Word, b: Self::Word) -> Self::Word;
    fn float_op(&mut self, op: FloatOp, a: Self::Float, b: Self::Float) -> Self::Float;
    fn unary(&mut self, op: UnaryOp, value: Self::Float) -> Self::Float;
    // Conversions only receive nonnegative integers exactly representable in f32.
    fn to_float(&mut self, value: Self::Word) -> Self::Float;
    fn to_word(&mut self, value: Self::Float) -> Self::Word;
}

pub fn seed(seed: i64) -> [u32; 2] { [seed as u32, (seed as u64 >> 32) as u32] }

// Each limb is an exact, finite f32. No integer bits travel as NaN or subnormal payloads.
pub fn encode<P: Primitives>(p: &mut P, key: [P::Word; 2]) -> [P::Float; KEY_SIZE] {
    let mask = p.word(0xffff);
    let shift = p.word(16);
    let limbs = [
        p.word_op(WordOp::And, key[0], mask),
        p.word_op(WordOp::ShiftRight, key[0], shift),
        p.word_op(WordOp::And, key[1], mask),
        p.word_op(WordOp::ShiftRight, key[1], shift),
    ];
    limbs.map(|limb| p.to_float(limb))
}

pub fn valid_key(limbs: &[f32]) -> bool {
    limbs.len() == KEY_SIZE && limbs.iter().all(|&x| {
        x.is_finite() && (0.0..=65535.0).contains(&x) && x.fract() == 0.0
    })
}

pub fn decode<P: Primitives>(p: &mut P, limbs: [P::Float; KEY_SIZE]) -> [P::Word; 2] {
    let limbs = limbs.map(|limb| p.to_word(limb));
    let shift = p.word(16);
    [0, 2].map(|i| {
        let hi = p.word_op(WordOp::ShiftLeft, limbs[i + 1], shift);
        p.word_op(WordOp::Or, limbs[i], hi)
    })
}

pub fn threefry<P: Primitives>(
    p: &mut P, key: [P::Word; 2], counter: [P::Word; 2],
) -> [P::Word; 2] {
    let parity = p.word(0x1bd11bda);
    let xor = p.word_op(WordOp::Xor, key[0], key[1]);
    let ks = [key[0], key[1], p.word_op(WordOp::Xor, parity, xor)];
    let mut x = [
        p.word_op(WordOp::Add, counter[0], ks[0]),
        p.word_op(WordOp::Add, counter[1], ks[1]),
    ];
    let rotations = [13, 15, 26, 6, 17, 29, 16, 24];
    for round in 0..20 {
        x[0] = p.word_op(WordOp::Add, x[0], x[1]);
        let rotation = rotations[round % 8];
        let left = p.word(rotation);
        let right = p.word(32 - rotation);
        let lo = p.word_op(WordOp::ShiftLeft, x[1], left);
        let hi = p.word_op(WordOp::ShiftRight, x[1], right);
        let rotated = p.word_op(WordOp::Or, lo, hi);
        x[1] = p.word_op(WordOp::Xor, x[0], rotated);
        if round % 4 == 3 {
            let injection = (round + 1) / 4;
            x[0] = p.word_op(WordOp::Add, x[0], ks[injection % 3]);
            x[1] = p.word_op(WordOp::Add, x[1], ks[(injection + 1) % 3]);
            let offset = p.word(injection as u32);
            x[1] = p.word_op(WordOp::Add, x[1], offset);
        }
    }
    x
}

// Sampling uses domain 0; child-key derivation uses domain 1.
pub fn split<P: Primitives>(p: &mut P, key: [P::Word; 2], index: P::Word) -> [P::Word; 2] {
    let domain = p.word(1);
    threefry(p, key, [index, domain])
}

pub fn bits<P: Primitives>(p: &mut P, key: [P::Word; 2], index: P::Word) -> [P::Word; 2] {
    let domain = p.word(0);
    threefry(p, key, [index, domain])
}

pub fn uniform<P: Primitives>(p: &mut P, bits: P::Word) -> P::Float {
    let shift = p.word(8);
    let mantissa = p.word_op(WordOp::ShiftRight, bits, shift);
    let value = p.to_float(mantissa);
    let scale = p.float(1.0 / 16777216.0);
    p.float_op(FloatOp::Multiply, value, scale)
}

pub fn normal<P: Primitives>(p: &mut P, bits: [P::Word; 2]) -> P::Float {
    let u1 = uniform(p, bits[0]);
    let u2 = uniform(p, bits[1]);
    let epsilon = p.float(1.0 / 16777216.0);
    let u1 = p.float_op(FloatOp::Maximum, u1, epsilon);
    let log = p.unary(UnaryOp::Log, u1);
    let neg2 = p.float(-2.0);
    let radius = p.float_op(FloatOp::Multiply, neg2, log);
    let radius = p.unary(UnaryOp::Sqrt, radius);
    let tau = p.float(std::f32::consts::TAU);
    let theta = p.float_op(FloatOp::Multiply, tau, u2);
    let cosine = p.unary(UnaryOp::Cos, theta);
    p.float_op(FloatOp::Multiply, radius, cosine)
}

pub fn randint<P: Primitives>(p: &mut P, bits: P::Word, low: i64, high: i64) -> P::Float {
    let u = uniform(p, bits);
    let range = p.float((high as i128 - low as i128) as f32);
    let scaled = p.float_op(FloatOp::Multiply, u, range);
    let index = p.unary(UnaryOp::Floor, scaled);
    let low = p.float(low as f32);
    let value = p.float_op(FloatOp::Add, index, low);
    let high = p.float((high - 1) as f32);
    let value = p.float_op(FloatOp::Minimum, value, high);
    p.float_op(FloatOp::Maximum, value, low)
}

pub struct Host;
impl Primitives for Host {
    type Word = u32;
    type Float = f32;
    fn word(&mut self, value: u32) -> u32 { value }
    fn float(&mut self, value: f32) -> f32 { value }
    fn word_op(&mut self, op: WordOp, a: u32, b: u32) -> u32 {
        match op {
            WordOp::Add => a.wrapping_add(b), WordOp::Xor => a ^ b,
            WordOp::And => a & b, WordOp::Or => a | b,
            WordOp::ShiftLeft => a << b, WordOp::ShiftRight => a >> b,
        }
    }
    fn float_op(&mut self, op: FloatOp, a: f32, b: f32) -> f32 {
        match op {
            FloatOp::Add => a + b, FloatOp::Multiply => a * b,
            FloatOp::Minimum => a.min(b), FloatOp::Maximum => a.max(b),
        }
    }
    fn unary(&mut self, op: UnaryOp, value: f32) -> f32 {
        match op {
            UnaryOp::Log => value.ln(), UnaryOp::Sqrt => value.sqrt(),
            UnaryOp::Cos => value.cos(), UnaryOp::Floor => value.floor(),
        }
    }
    fn to_float(&mut self, value: u32) -> f32 { value as f32 }
    fn to_word(&mut self, value: f32) -> u32 { value as u32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random123_known_answers() {
        // Random123 tests/kat_vectors, Threefry2x32 with 20 rounds.
        for (counter, key, expected) in [
            ([0, 0], [0, 0], [0x6b200159, 0x99ba4efe]),
            ([u32::MAX; 2], [u32::MAX; 2], [0x1cb996fc, 0xbb002be7]),
            ([0x243f6a88, 0x85a308d3], [0x13198a2e, 0x03707344], [0xc4923a9c, 0x483df7a0]),
        ] {
            assert_eq!(threefry(&mut Host, key, counter), expected);
        }
    }

    #[test]
    fn key_roundtrips_are_lossless() {
        for seed_value in [0, 42, -1, i64::MIN, i64::MAX, 16777217, 4294967295] {
            let key = seed(seed_value);
            let encoded = encode(&mut Host, key);
            assert!(valid_key(&encoded));
            assert_eq!(decode(&mut Host, encoded), key);
            for i in [0, 1, 17, u32::MAX] {
                let child = split(&mut Host, key, i);
                assert_eq!(decode(&mut Host, encode(&mut Host, child)), child);
            }
        }
        for invalid in [vec![0.0; 2], vec![65536.0; 4], vec![-1.0; 4], vec![0.5; 4], vec![f32::NAN; 4]] {
            assert!(!valid_key(&invalid));
        }
    }

    #[test]
    fn distribution_endpoints() {
        assert_eq!(uniform(&mut Host, 0), 0.0);
        assert_eq!(uniform(&mut Host, u32::MAX), 1.0 - 1.0 / 16777216.0);
        for words in [[0, 0], [u32::MAX; 2], [0, u32::MAX]] {
            assert!(normal(&mut Host, words).is_finite());
            assert!((-3.0..7.0).contains(&randint(&mut Host, words[0], -3, 7)));
        }
    }
}
