//! rkyv glue: encode records, and view them zero-copy.
//!
//! The crate enables rkyv's `unaligned` format, so archived records can be
//! accessed directly inside redb's page buffers, whatever the byte offset of
//! the value inside the page. [`view`] verifies the zero-copy contract: the
//! returned reference must point *into* the input slice.

use crate::error::{Error, Result};
use rkyv::api::high::{HighDeserializer, HighSerializer, HighValidator};
use rkyv::bytecheck::CheckBytes;
use rkyv::rancor::Error as RkyvError;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};

/// Serialize a record to bytes.
pub fn encode<T>(value: &T) -> Result<AlignedVec>
where
    T: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, RkyvError>>,
{
    Ok(rkyv::to_bytes::<RkyvError>(value)?)
}

/// Validate and borrow the archived form of `T` from `bytes` without copying.
pub fn view<T>(bytes: &[u8]) -> Result<&T::Archived>
where
    T: Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, RkyvError>>,
{
    let archived = rkyv::access::<T::Archived, RkyvError>(bytes)?;
    if !points_into(archived, bytes) {
        return Err(Error::Serialization("zero-copy contract violated: archived view escapes its buffer".into()));
    }
    Ok(archived)
}

/// Fully deserialize an owned `T` (used on write paths that mutate records).
pub fn decode<T>(bytes: &[u8]) -> Result<T>
where
    T: Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, RkyvError>> + Deserialize<T, HighDeserializer<RkyvError>>,
{
    let archived = view::<T>(bytes)?;
    Ok(rkyv::deserialize::<T, RkyvError>(archived)?)
}

/// True when `r` lies entirely inside `buf` — i.e. no hidden copy was made.
pub fn points_into<R: ?Sized>(r: &R, buf: &[u8]) -> bool {
    let start = buf.as_ptr() as usize;
    let end = start + buf.len();
    let p = (r as *const R).cast::<u8>() as usize;
    p >= start && p + std::mem::size_of_val(r) <= end
}

/// Encode a vector of f32 as little-endian bytes (the `vectors` table format).
pub fn f32s_to_le(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode little-endian f32 bytes into an owned vector.
pub fn le_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Chunk, HnswNode, Link};

    fn sample() -> Chunk {
        Chunk {
            id: 7,
            text: "hello".into(),
            tags: vec!["a".into()],
            entities: vec![1, 2],
            source: String::new(),
            created_at: 1,
            updated_at: 2,
            hits: 1,
        }
    }

    #[test]
    fn roundtrip_and_view_is_zero_copy() {
        let bytes = encode(&sample()).unwrap();
        let a = view::<Chunk>(&bytes).unwrap();
        assert_eq!(a.text.as_str(), "hello");
        assert!(points_into(a, &bytes));
        assert!(points_into(a.text.as_str().as_bytes(), &bytes));
        assert_eq!(decode::<Chunk>(&bytes).unwrap(), sample());
    }

    #[test]
    fn view_works_at_any_alignment() {
        let mut node = HnswNode::empty(2);
        node.layers[1].push(Link { id: 9, sim: 0.5 });
        let bytes = encode(&node).unwrap();
        for off in 0..8 {
            let mut buf = vec![0u8; off];
            buf.extend_from_slice(&bytes);
            let a = view::<HnswNode>(&buf[off..]).unwrap();
            assert_eq!(a.layers[1][0].id.to_native(), 9);
        }
    }

    #[test]
    fn garbage_is_rejected() {
        assert!(view::<Chunk>(&[1, 2, 3]).is_err());
        assert!(decode::<HnswNode>(&[0xff; 64]).is_err());
    }

    #[test]
    fn f32_le_roundtrip() {
        let v = vec![1.0f32, -2.5, 3.25];
        assert_eq!(le_to_f32s(&f32s_to_le(&v)), v);
    }
}
