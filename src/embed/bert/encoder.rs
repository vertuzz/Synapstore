//! BERT encoder forward pass (inference only, one sequence at a time).

use super::safetensors::SafeTensors;
use crate::error::{Error, Result};
use crate::math::gemm::{Packed, linear};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Gelu,
    GeluTanh,
    Relu,
}

#[derive(Debug, Clone, Copy)]
pub struct BertConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub max_positions: usize,
    pub type_vocab: usize,
    pub eps: f32,
    pub act: Activation,
}

/// A `Linear` layer with its weight packed for [`linear`].
struct Dense {
    w: Packed,
    b: Vec<f32>,
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

struct Layer {
    qkv: Dense,
    attn_out: Dense,
    attn_norm: Norm,
    up: Dense,
    down: Dense,
    out_norm: Norm,
}

pub struct Encoder {
    pub cfg: BertConfig,
    word: Vec<f32>,
    pos: Vec<f32>,
    token_type: Vec<f32>,
    emb_norm: Norm,
    layers: Vec<Layer>,
}

struct Loader<'a> {
    st: &'a mut SafeTensors,
    prefix: String,
}

impl Loader<'_> {
    fn get(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let full = format!("{}{name}", self.prefix);
        if self.st.shape(&full).is_none() {
            // Old TF-converted checkpoints call LayerNorm parameters gamma/beta.
            let alt = full.replace("LayerNorm.weight", "LayerNorm.gamma").replace("LayerNorm.bias", "LayerNorm.beta");
            if self.st.shape(&alt).is_some() {
                return self.st.tensor(&alt, shape);
            }
        }
        self.st.tensor(&full, shape)
    }

    fn dense(&mut self, name: &str, inp: usize, out: usize) -> Result<Dense> {
        let w = self.get(&format!("{name}.weight"), &[out, inp])?;
        Ok(Dense { w: Packed::from_linear(&w, out, inp), b: self.get(&format!("{name}.bias"), &[out])? })
    }

    fn norm(&mut self, name: &str, h: usize) -> Result<Norm> {
        Ok(Norm { g: self.get(&format!("{name}.weight"), &[h])?, b: self.get(&format!("{name}.bias"), &[h])? })
    }
}

fn layer_norm(x: &mut [f32], n: &Norm, eps: f32) {
    let h = n.g.len();
    for row in x.chunks_exact_mut(h) {
        let mean = row.iter().sum::<f32>() / h as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ((v, g), b) in row.iter_mut().zip(&n.g).zip(&n.b) {
            *v = (*v - mean) * inv * g + b;
        }
    }
}

/// `e^x` to ~2 ulp, branch-free so loops over it vectorize (Cephes expf).
#[inline(always)]
fn exp(x: f32) -> f32 {
    let x = x.clamp(-87.3, 88.7);
    // Round to nearest via the 1.5·2²³ trick (vectorizes, unlike `round`).
    let n = (x * std::f32::consts::LOG2_E + 12_582_912.0) - 12_582_912.0;
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = ((((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 1.666_666_5e-1)
        * r
        + 0.5;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// erf with |error| < 1.5e-7 (Abramowitz & Stegun 7.1.26).
fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = (((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_74) * t + 0.254_829_6;
    s * (1.0 - poly * t * exp(-x * x))
}

fn activate(x: &mut [f32], act: Activation) {
    for v in x {
        *v = match act {
            Activation::Gelu => 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2)),
            Activation::GeluTanh => 0.5 * *v * (1.0 + (0.797_884_6 * (*v + 0.044_715 * *v * *v * *v)).tanh()),
            Activation::Relu => v.max(0.0),
        }
    }
}

impl Encoder {
    pub fn load(st: &mut SafeTensors, cfg: BertConfig) -> Result<Self> {
        let prefix = if st.shape("bert.embeddings.word_embeddings.weight").is_some() { "bert." } else { "" };
        let mut l = Loader { st, prefix: prefix.into() };
        let h = cfg.hidden;
        if cfg.heads == 0 || !h.is_multiple_of(cfg.heads) {
            return Err(Error::InvalidInput(format!("hidden size {h} not divisible by {} heads", cfg.heads)));
        }
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("encoder.layer.{i}");
            // Fuse Q, K and V into one [3h][h] projection.
            let (mut w, mut b) = (Vec::with_capacity(3 * h * h), Vec::with_capacity(3 * h));
            for part in ["query", "key", "value"] {
                w.extend(l.get(&format!("{p}.attention.self.{part}.weight"), &[h, h])?);
                b.extend(l.get(&format!("{p}.attention.self.{part}.bias"), &[h])?);
            }
            layers.push(Layer {
                qkv: Dense { w: Packed::from_linear(&w, 3 * h, h), b },
                attn_out: l.dense(&format!("{p}.attention.output.dense"), h, h)?,
                attn_norm: l.norm(&format!("{p}.attention.output.LayerNorm"), h)?,
                up: l.dense(&format!("{p}.intermediate.dense"), h, cfg.intermediate)?,
                down: l.dense(&format!("{p}.output.dense"), cfg.intermediate, h)?,
                out_norm: l.norm(&format!("{p}.output.LayerNorm"), h)?,
            });
        }
        Ok(Encoder {
            word: l.get("embeddings.word_embeddings.weight", &[cfg.vocab, h])?,
            pos: l.get("embeddings.position_embeddings.weight", &[cfg.max_positions, h])?,
            token_type: l.get("embeddings.token_type_embeddings.weight", &[cfg.type_vocab, h])?,
            emb_norm: l.norm("embeddings.LayerNorm", h)?,
            layers,
            cfg,
        })
    }

    fn dense(x: &[f32], d: &Dense) -> Vec<f32> {
        linear(x, &d.w, &d.b)
    }

    fn attention(&self, qkv: &[f32], n: usize) -> Vec<f32> {
        let h = self.cfg.hidden;
        let dh = h / self.cfg.heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut ctx = vec![0f32; n * h];
        let (zeros_n, zeros_d) = (vec![0f32; n], vec![0f32; dh]);
        let (mut q, mut k, mut v) = (vec![0f32; n * dh], vec![0f32; n * dh], vec![0f32; n * dh]);
        for head in 0..self.cfg.heads {
            let off = head * dh;
            for t in 0..n {
                let row = &qkv[t * 3 * h..(t + 1) * 3 * h];
                for d in 0..dh {
                    q[t * dh + d] = row[off + d] * scale;
                }
                k[t * dh..(t + 1) * dh].copy_from_slice(&row[h + off..h + off + dh]);
                v[t * dh..(t + 1) * dh].copy_from_slice(&row[2 * h + off..2 * h + off + dh]);
            }
            // scores = (Q/√d)·Kᵀ, softmax per row, then P·V.
            let mut p = linear(&q, &Packed::from_linear(&k, n, dh), &zeros_n);
            for row in p.chunks_exact_mut(n) {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in row.iter_mut() {
                    *s = exp(*s - max);
                    sum += *s;
                }
                let inv = 1.0 / sum;
                row.iter_mut().for_each(|s| *s *= inv);
            }
            let out = linear(&p, &Packed::from_rows(&v, n, dh), &zeros_d);
            for t in 0..n {
                ctx[t * h + off..t * h + off + dh].copy_from_slice(&out[t * dh..(t + 1) * dh]);
            }
        }
        ctx
    }

    /// Final hidden states (`[len][hidden]` each) for a batch of sequences.
    /// Sequences share every dense layer, so weights stream from memory
    /// once per batch instead of once per text.
    pub fn forward(&self, batch: &[Vec<u32>]) -> Vec<Vec<f32>> {
        let h = self.cfg.hidden;
        let lens: Vec<usize> = batch.iter().map(|ids| ids.len().min(self.cfg.max_positions)).collect();
        let total: usize = lens.iter().sum();
        let mut x = vec![0f32; total * h];
        let mut row = 0;
        for (ids, &n) in batch.iter().zip(&lens) {
            for (t, &id) in ids.iter().take(n).enumerate() {
                let id = (id as usize).min(self.cfg.vocab - 1);
                let dst = &mut x[row * h..(row + 1) * h];
                let (w, p, ty) =
                    (&self.word[id * h..(id + 1) * h], &self.pos[t * h..(t + 1) * h], &self.token_type[..h]);
                for i in 0..h {
                    dst[i] = w[i] + p[i] + ty[i];
                }
                row += 1;
            }
        }
        layer_norm(&mut x, &self.emb_norm, self.cfg.eps);
        for l in &self.layers {
            let qkv = Self::dense(&x, &l.qkv);
            let mut ctx = Vec::with_capacity(total * h);
            let mut start = 0;
            for &n in &lens {
                ctx.extend(self.attention(&qkv[start * 3 * h..(start + n) * 3 * h], n));
                start += n;
            }
            let a = Self::dense(&ctx, &l.attn_out);
            x.iter_mut().zip(&a).for_each(|(v, d)| *v += d);
            layer_norm(&mut x, &l.attn_norm, self.cfg.eps);
            let mut up = Self::dense(&x, &l.up);
            activate(&mut up, self.cfg.act);
            let down = Self::dense(&up, &l.down);
            x.iter_mut().zip(&down).for_each(|(v, d)| *v += d);
            layer_norm(&mut x, &l.out_norm, self.cfg.eps);
        }
        let mut out = Vec::with_capacity(batch.len());
        let mut start = 0;
        for &n in &lens {
            out.push(x[start * h..(start + n) * h].to_vec());
            start += n;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_is_accurate() {
        for i in -800..800 {
            let x = i as f32 * 0.1;
            let (got, want) = (exp(x), x.exp());
            assert!(((got - want) / want).abs() < 4e-7, "exp({x}) = {got}, want {want}");
        }
        assert!(exp(-200.0) < 1e-37 && exp(-200.0) >= 0.0);
    }

    #[test]
    fn erf_is_accurate() {
        for (x, want) in [(0.0, 0.0), (0.5, 0.520_499_9), (1.0, 0.842_700_8), (-2.0, -0.995_322_3), (4.0, 1.0)] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x})");
        }
    }
}
