//! Bespoke text chunker: packs whole sentences into chunks of at most
//! `max_chars` characters; over-long sentences are split on word boundaries.

/// Split `text` into sentences (terminators `.`, `!`, `?` followed by
/// whitespace, or a newline).
pub fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let next_ws = chars.peek().is_none_or(|(_, n)| n.is_whitespace());
        if c == '\n' || (matches!(c, '.' | '!' | '?') && next_ws) {
            let end = i + c.len_utf8();
            let s = text[start..end].trim();
            if !s.is_empty() {
                out.push(s);
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn split_long(sentence: &str, max_chars: usize, out: &mut Vec<String>) {
    let mut cur = String::new();
    for word in sentence.split_whitespace() {
        let wlen = word.chars().count();
        if wlen > max_chars {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            let chars: Vec<char> = word.chars().collect();
            out.extend(chars.chunks(max_chars).map(|c| c.iter().collect::<String>()));
            continue;
        }
        let need = if cur.is_empty() { wlen } else { wlen + 1 };
        if cur.chars().count() + need > max_chars && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
}

/// Chunk `text`. Returns an empty vector for blank input.
pub fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(16);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.chars().count() <= max_chars {
        return vec![trimmed.to_string()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0;
    for s in sentences(trimmed) {
        let slen = s.chars().count();
        if slen > max_chars {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
                cur_len = 0;
            }
            split_long(s, max_chars, &mut out);
            continue;
        }
        let need = if cur.is_empty() { slen } else { slen + 1 };
        if cur_len + need > max_chars {
            out.push(std::mem::take(&mut cur));
            cur_len = 0;
        }
        if !cur.is_empty() {
            cur.push(' ');
            cur_len += 1;
        }
        cur.push_str(s);
        cur_len += slen;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(chunk_text("  hello world ", 100), vec!["hello world"]);
        assert!(chunk_text("   ", 100).is_empty());
    }

    #[test]
    fn chunks_respect_limit_and_keep_all_words() {
        let text = "First sentence here. Second one is a bit longer! Third? \
                    A very very very very very very very very very long sentence without end";
        let chunks = chunk_text(text, 24);
        assert!(chunks.iter().all(|c| c.chars().count() <= 24), "{chunks:?}");
        let joined: Vec<&str> = chunks.iter().flat_map(|c| c.split_whitespace()).collect();
        let orig: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(joined, orig);
    }

    #[test]
    fn giant_word_is_hard_split() {
        let w = "x".repeat(50);
        let chunks = chunk_text(&w, 16);
        assert_eq!(chunks.concat(), w);
        assert!(chunks.iter().all(|c| c.chars().count() <= 16));
    }

    #[test]
    fn sentence_splitting() {
        assert_eq!(sentences("A. B! C?\nD 3.14 e"), vec!["A.", "B!", "C?", "D 3.14 e"]);
    }
}
