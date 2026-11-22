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

/// Words that are never entities on their own, wherever they appear:
/// emphasis and section words, months and weekdays, and common Russian /
/// Ukrainian words that are capitalized at the start of a sentence.
#[rustfmt::skip]
const NOT_NAMES: &[&str] = &[
    "important", "note", "notes", "todo", "warning", "never", "always", "not", "done", "ok", "status", "plan",
    "next", "decision", "update", "summary", "context", "goal", "question", "answer", "result", "rule",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "январь", "января", "февраль", "февраля", "март", "марта",
    "апрель", "апреля", "май", "мая", "июнь", "июня", "июль", "июля",
    "август", "августа", "сентябрь", "сентября", "октябрь",
    "октября", "ноябрь", "ноября", "декабрь", "декабря",
    "понедельник", "вторник", "среда", "четверг", "пятница",
    "суббота", "воскресенье", "я", "мы", "ты", "вы", "он", "она", "оно",
    "они", "это", "этот", "эта", "эти", "тот", "та", "те", "что", "чтобы",
    "как", "когда", "где", "куда", "откуда", "почему", "зачем", "если",
    "но", "и", "а", "или", "в", "во", "на", "по", "для", "после", "до", "при",
    "из", "от", "с", "со", "к", "о", "об", "у", "за", "под", "над", "не", "нет",
    "да", "так", "то", "также", "тоже", "еще", "ещё", "уже", "все", "всё",
    "сегодня", "вчера", "завтра", "сейчас", "здесь", "там", "тут",
    "его", "её", "ее", "их", "мой", "моя", "наш", "наша", "наши", "ваш",
    "свой", "есть", "был", "была", "было", "были", "будет", "можно",
    "нужно", "надо", "важно", "итого", "итог", "примечание", "статус",
    "решение", "план", "вопрос", "ответ", "цель", "задача", "правило",
    "контекст", "дальше", "далее", "сначала", "потом", "итак",
    "кстати", "однако", "пример", "например", "всего", "каждый",
    "каждые", "новый", "новая", "старый", "один", "одна", "два", "три",
    "ні", "так", "це", "що", "як", "коли", "де", "але",
];

/// Russian month names in any grammatical case.
const MONTH_STEMS: &[&str] =
    &["январ", "феврал", "апрел", "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр"];

fn not_a_name(word: &str) -> bool {
    let w = word.to_lowercase();
    NOT_NAMES.contains(&w.as_str())
        || matches!(w.as_str(), "марте" | "мае")
        || w.chars().count() <= 9 && MONTH_STEMS.iter().any(|s| w.starts_with(s))
}

/// True for a name that heuristic extraction should not have produced:
/// only stop/emphasis words, digits or a single character. Used to clean
/// up stores built by older versions (`reindex`).
pub fn is_junk_entity(name: &str) -> bool {
    let words: Vec<&str> = name.split_whitespace().collect();
    words.is_empty()
        || name.chars().filter(|c| c.is_alphanumeric()).count() < 2
        || name.chars().all(|c| !c.is_alphabetic())
        || words.iter().all(|w| {
            let w = w.trim_matches(|c: char| !c.is_alphanumeric());
            w.is_empty() || not_a_name(w) || is_starter(w)
        })
}

fn flush(run: &mut Vec<String>, run_at_start: bool, out: &mut Vec<String>) {
    let mut words: &[String] = run;
    if run_at_start && words.first().is_some_and(|w| is_starter(w)) {
        words = &words[1..];
    }
    while words.first().is_some_and(|w| not_a_name(w)) {
        words = &words[1..];
    }
    while words.last().is_some_and(|w| not_a_name(w)) {
        words = &words[..words.len() - 1];
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
    fn skips_emphasis_months_and_russian_starters() {
        let e =
            extract_entities("ВАЖНО: Вероника и Алексей летят в Турцию. Он рад. В Октябре был Daikin. IMPORTANT note.");
        assert_eq!(e, vec!["Вероника", "Алексей", "Турцию", "Daikin"]);
        assert!(is_junk_entity("Он"));
        assert!(is_junk_entity("ВАЖНО"));
        assert!(is_junk_entity("November"));
        assert!(is_junk_entity("2026"));
        assert!(!is_junk_entity("Walker & Dunlop"));
        assert!(!is_junk_entity("IBKR"));
    }

    #[test]
    fn empty_and_lowercase() {
        assert!(extract_entities("").is_empty());
        assert!(extract_entities("nothing to see here").is_empty());
    }
}
