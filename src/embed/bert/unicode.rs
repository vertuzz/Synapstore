//! Unicode properties for BERT normalization, parsed once from the
//! generated `unicode.dat` (see `scripts/gen_unicode_tables.py`).

use std::sync::OnceLock;

struct Tables {
    punct: Vec<(u32, u32)>,
    mark: Vec<(u32, u32)>,
    other: Vec<(u32, u32)>,
    space: Vec<(u32, u32)>,
    /// Sorted `(code point, accent-stripped replacement)`.
    strip: Vec<(u32, Box<[char]>)>,
}

fn parse_ranges(fields: &str) -> Vec<(u32, u32)> {
    fields
        .split_whitespace()
        .filter_map(|r| {
            let (a, b) = r.split_once('-').unwrap_or((r, r));
            Some((u32::from_str_radix(a, 16).ok()?, u32::from_str_radix(b, 16).ok()?))
        })
        .collect()
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables { punct: vec![], mark: vec![], other: vec![], space: vec![], strip: vec![] };
        for line in include_str!("unicode.dat").lines() {
            let (tag, rest) = line.split_once(' ').unwrap_or((line, ""));
            match tag {
                "P" => t.punct = parse_ranges(rest),
                "Mn" => t.mark = parse_ranges(rest),
                "C" => t.other = parse_ranges(rest),
                "Zs" => t.space = parse_ranges(rest),
                "D" => {
                    for entry in rest.split_whitespace() {
                        let Some((cp, seq)) = entry.split_once(':') else { continue };
                        let Ok(cp) = u32::from_str_radix(cp, 16) else { continue };
                        let chars: Box<[char]> = seq
                            .split('.')
                            .filter(|s| !s.is_empty())
                            .filter_map(|h| char::from_u32(u32::from_str_radix(h, 16).ok()?))
                            .collect();
                        t.strip.push((cp, chars));
                    }
                }
                _ => {}
            }
        }
        t.strip.sort_by_key(|e| e.0);
        t
    })
}

fn in_ranges(r: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    r.binary_search_by(|&(a, b)| {
        if b < c {
            std::cmp::Ordering::Less
        } else if a > c {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

/// BERT's notion of punctuation: all non-alphanumeric ASCII symbols plus
/// Unicode category P*.
pub fn is_punctuation(c: char) -> bool {
    let n = c as u32;
    (33..=47).contains(&n)
        || (58..=64).contains(&n)
        || (91..=96).contains(&n)
        || (123..=126).contains(&n)
        || in_ranges(&tables().punct, c)
}

/// Whitespace as the reference (Rust) BERT tokenizer defines it.
pub fn is_whitespace(c: char) -> bool {
    c.is_whitespace() || in_ranges(&tables().space, c)
}

/// Categories Cc/Cf/Co/Cn (except tab/newline), dropped by `clean_text`.
pub fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && in_ranges(&tables().other, c)
}

pub fn is_mark(c: char) -> bool {
    in_ranges(&tables().mark, c)
}

/// CJK ideographs that BERT surrounds with spaces.
pub fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

/// Append `c` with accents stripped (NFD, then drop nonspacing marks).
pub fn push_stripped(c: char, out: &mut String) {
    let n = c as u32;
    if (0xAC00..=0xD7A3).contains(&n) {
        // Hangul syllable → conjoining jamo (Unicode §3.12).
        let s = n - 0xAC00;
        let (l, v, t) = (0x1100 + s / 588, 0x1161 + (s % 588) / 28, s % 28);
        for x in [l, v] {
            out.extend(char::from_u32(x));
        }
        if t != 0 {
            out.extend(char::from_u32(0x11A7 + t));
        }
        return;
    }
    if is_mark(c) {
        return;
    }
    let t = &tables().strip;
    match t.binary_search_by_key(&n, |e| e.0) {
        Ok(i) => out.extend(t[i].1.iter()),
        Err(_) => out.push(c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(s: &str) -> String {
        let mut o = String::new();
        s.chars().for_each(|c| push_stripped(c, &mut o));
        o
    }

    #[test]
    fn properties() {
        assert!(is_punctuation('!') && is_punctuation('$') && is_punctuation('—') && is_punctuation('«'));
        assert!(!is_punctuation('a') && !is_punctuation('ж'));
        assert!(is_whitespace('\u{a0}') && is_whitespace('\u{3000}') && !is_whitespace('\u{200b}'));
        assert!(is_control('\u{200b}') && is_control('\u{0}') && !is_control('\n'));
        assert!(is_cjk('数') && !is_cjk('a'));
    }

    #[test]
    fn strips_accents() {
        assert_eq!(strip("Café naïve Ångström"), "Cafe naive Angstrom");
        assert_eq!(strip("ёжик й"), "ежик и");
        assert_eq!(strip("e\u{301}"), "e");
        assert_eq!(strip("한"), "\u{1112}\u{1161}\u{11ab}");
    }
}
