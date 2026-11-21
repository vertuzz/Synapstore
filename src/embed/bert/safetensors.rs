//! Minimal reader for the `safetensors` format: an 8-byte little-endian
//! header length, a JSON header, then raw tensor bytes. F32/F16/BF16 tensors
//! are converted to `f32` on load.

use crate::error::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_HEADER: u64 = 100 << 20;

#[derive(Debug, Clone)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

/// An open safetensors file; tensors are read on demand.
pub struct SafeTensors {
    file: File,
    data_start: u64,
    entries: HashMap<String, Entry>,
}

fn bad(msg: impl Into<String>) -> Error {
    Error::InvalidInput(format!("safetensors: {}", msg.into()))
}

/// IEEE half → f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            // Subnormal: renormalize.
            let shift = f.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((f << shift) & 0x3ff) << 13)
        }
        (31, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 112) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let total = file.metadata()?.len();
        let mut len = [0u8; 8];
        file.read_exact(&mut len).map_err(|_| bad("file too short"))?;
        let n = u64::from_le_bytes(len);
        if n > MAX_HEADER || 8 + n > total {
            return Err(bad("header length out of range"));
        }
        let mut header = vec![0u8; n as usize];
        file.read_exact(&mut header)?;
        let json: Value = serde_json::from_slice(&header).map_err(|e| bad(e.to_string()))?;
        let obj = json.as_object().ok_or_else(|| bad("header is not an object"))?;
        let data_len = total - 8 - n;
        let mut entries = HashMap::new();
        for (name, v) in obj.iter().filter(|(k, _)| *k != "__metadata__") {
            let dtype = v["dtype"].as_str().ok_or_else(|| bad(format!("{name}: no dtype")))?.to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| bad(format!("{name}: no shape")))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| bad(format!("{name}: bad shape")))?;
            let off = v["data_offsets"].as_array().ok_or_else(|| bad(format!("{name}: no offsets")))?;
            let (begin, end) = match (off.first().and_then(Value::as_u64), off.get(1).and_then(Value::as_u64)) {
                (Some(b), Some(e)) if b <= e && e <= data_len => (b, e),
                _ => return Err(bad(format!("{name}: offsets out of range"))),
            };
            entries.insert(name.clone(), Entry { dtype, shape, begin, end });
        }
        Ok(SafeTensors { file, data_start: 8 + n, entries })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// Read tensor `name` as f32, checking that its shape is `expect`.
    pub fn tensor(&mut self, name: &str, expect: &[usize]) -> Result<Vec<f32>> {
        let e = self.entries.get(name).ok_or_else(|| bad(format!("missing tensor {name}")))?.clone();
        if e.shape != expect {
            return Err(bad(format!("{name}: shape {:?}, expected {expect:?}", e.shape)));
        }
        let count: usize = e.shape.iter().product();
        let width = match e.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => return Err(bad(format!("{name}: unsupported dtype {other}"))),
        };
        if (e.end - e.begin) as usize != count * width {
            return Err(bad(format!("{name}: byte length does not match shape")));
        }
        let mut raw = vec![0u8; count * width];
        self.file.seek(SeekFrom::Start(self.data_start + e.begin))?;
        self.file.read_exact(&mut raw)?;
        Ok(match e.dtype.as_str() {
            "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            "F16" => raw.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            _ => raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(f16_to_f32(0x7c00).is_infinite() && f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn reads_and_validates() {
        let dir = std::env::temp_dir().join(format!("st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        let header = br#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F16","shape":[1],"data_offsets":[8,10]},"__metadata__":{"format":"pt"}}"#;
        let mut f = (header.len() as u64).to_le_bytes().to_vec();
        f.extend_from_slice(header);
        f.extend_from_slice(&1.5f32.to_le_bytes());
        f.extend_from_slice(&(-2f32).to_le_bytes());
        f.extend_from_slice(&0x3c00u16.to_le_bytes());
        std::fs::write(&p, &f).unwrap();
        let mut st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.tensor("a", &[2]).unwrap(), vec![1.5, -2.0]);
        assert_eq!(st.tensor("b", &[1]).unwrap(), vec![1.0]);
        assert!(st.tensor("a", &[3]).is_err() && st.tensor("zz", &[1]).is_err());
        std::fs::write(&p, &f[..20]).unwrap();
        assert!(SafeTensors::open(&p).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
