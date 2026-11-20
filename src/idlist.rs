//! Packed, sorted, de-duplicated lists of u64 ids stored as little-endian
//! bytes. Used as values of the SPO / POS / OPS indices and entity→chunk
//! links. Lookups are binary searches directly on the borrowed page bytes.

/// Number of ids in a packed list.
pub fn len(bytes: &[u8]) -> usize {
    bytes.len() / 8
}

#[inline]
fn at(bytes: &[u8], i: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
    u64::from_le_bytes(b)
}

/// Iterate ids in ascending order.
pub fn iter(bytes: &[u8]) -> impl Iterator<Item = u64> + '_ {
    (0..len(bytes)).map(move |i| at(bytes, i))
}

/// Binary search for `id`. `Ok(index)` if present, `Err(insert_pos)` if not.
pub fn search(bytes: &[u8], id: u64) -> Result<usize, usize> {
    let (mut lo, mut hi) = (0usize, len(bytes));
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match at(bytes, mid).cmp(&id) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Ok(mid),
        }
    }
    Err(lo)
}

pub fn contains(bytes: &[u8], id: u64) -> bool {
    search(bytes, id).is_ok()
}

/// Returns a new list with `id` inserted, or `None` if it was already present.
pub fn with(bytes: &[u8], id: u64) -> Option<Vec<u8>> {
    let pos = search(bytes, id).err()?;
    let mut out = Vec::with_capacity(bytes.len() + 8);
    out.extend_from_slice(&bytes[..pos * 8]);
    out.extend_from_slice(&id.to_le_bytes());
    out.extend_from_slice(&bytes[pos * 8..]);
    Some(out)
}

/// Returns a new list without `id`, or `None` if it was absent.
pub fn without(bytes: &[u8], id: u64) -> Option<Vec<u8>> {
    let pos = search(bytes, id).ok()?;
    let mut out = Vec::with_capacity(bytes.len().saturating_sub(8));
    out.extend_from_slice(&bytes[..pos * 8]);
    out.extend_from_slice(&bytes[pos * 8 + 8..]);
    Some(out)
}

/// Build a packed list from arbitrary ids.
pub fn from_ids(ids: impl IntoIterator<Item = u64>) -> Vec<u8> {
    let mut v: Vec<u64> = ids.into_iter().collect();
    v.sort_unstable();
    v.dedup();
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_remove_keep_sorted_unique() {
        let mut list = Vec::new();
        for id in [5u64, 1, 9, 5, 3, u64::MAX, 0] {
            if let Some(next) = with(&list, id) {
                list = next;
            }
        }
        assert_eq!(iter(&list).collect::<Vec<_>>(), vec![0, 1, 3, 5, 9, u64::MAX]);
        assert!(with(&list, 9).is_none());
        list = without(&list, 5).unwrap();
        assert!(!contains(&list, 5));
        assert!(without(&list, 5).is_none());
        assert_eq!(list, from_ids([9, 0, 3, 1, u64::MAX]));
    }
}
