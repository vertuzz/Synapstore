//! Heuristic entity extraction used by `memorize` when the caller does not
//! pass explicit entities: runs of capitalized words ("Alice Chen",
//! "Project Falcon", "NASA") become entity names.

const SENTENCE_STARTERS: &[&str] = &[
    "a",
    "after",
    "also",
    "an",
    "and",
    "at",
    "before",
    "but",
    "for",
    "he",
    "her",
    "here",
    "his",
    "how",
    "i",
    "if",
    "in",
    "it",
    "its",
    "my",
    "no",
    "note",
    "of",
    "on",
    "our",
    "please",
    "she",
    "so",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "today",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "yes",
    "yesterday",
    "you",
    "your",
];

fn is_capitalized(word: &str) -> bool {
    word.chars().next().is_some_and(char::is_uppercase)
}

fn is_starter(word: &str) -> bool {
    SENTENCE_STARTERS.contains(&word.to_lowercase().as_str())
}

fn flush(run: &mut Vec<String>, run_at_start: bool, out: &mut Vec<String>) {
    let mut words: &[String] = run;
    if run_at_start && words.first().is_some_and(|w| is_starter(w)) {
        words = &words[1..];
    }
    if words.len() > 1 || words.first().is_some_and(|w| w.chars().count() >= 2) {
        let name = words.join(" ");
        if !out.iter().any(|e| e.eq_ignore_ascii_case(&name)) {
            out.push(name);
        }
    }
    run.clear();
}

/// Extract candidate entity names, in order of first appearance.
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut run: Vec<String> = Vec::new();
    let mut run_at_start = false;
    let mut sentence_start = true;
    for raw in text.split_whitespace() {
        let word = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '+' && c != '#');
        let word = word.trim_start_matches(['+', '#']);
        let ends_clause = raw.ends_with(|c: char| ",.;:!?)\"".contains(c));
        let ends_sentence = raw.ends_with(['.', '!', '?', ':']);
        if !word.is_empty() && is_capitalized(word) {
            if run.is_empty() {
                run_at_start = sentence_start;
            }
            run.push(word.to_string());
        } else {
            flush(&mut run, run_at_start, &mut out);
        }
        if ends_clause {
            flush(&mut run, run_at_start, &mut out);
        }
        sentence_start = ends_sentence || word.is_empty() && sentence_start;
    }
    flush(&mut run, run_at_start, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_multiword_names() {
        let e = extract_entities("Yesterday Alice Chen joined Acme Corp to lead Project Falcon.");
        assert_eq!(e, vec!["Alice Chen", "Acme Corp", "Project Falcon"]);
    }

    #[test]
    fn skips_sentence_starters_and_dedups() {
        let e = extract_entities("The NASA team met Bob. Bob said hi. We like Rust, and C++ too.");
        assert_eq!(e, vec!["NASA", "Bob", "Rust", "C++"]);
    }

    #[test]
    fn empty_and_lowercase() {
        assert!(extract_entities("").is_empty());
        assert!(extract_entities("nothing to see here").is_empty());
    }
}
