//! Deterministic synthetic "agent life" (PRD §8): a world of people,
//! companies, projects and cities with tightly linked facts, plus unlinked
//! thoughts and episodic tool logs spread over months.

use super::rng::Rng;
use crate::decay::DAY_MS;
use crate::store::FactInput;

const FIRST: &[&str] = &[
    "Alice", "Bob", "Carmen", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonas", "Kira", "Liam", "Maya",
    "Nikolai", "Olga", "Pavel", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wei", "Ximena", "Yusuf", "Zoe",
    "Arjun", "Beatriz", "Chen", "Dario",
];
const LAST: &[&str] = &[
    "Chen",
    "Novak",
    "Okafor",
    "Silva",
    "Kowalski",
    "Haddad",
    "Tanaka",
    "Ivanova",
    "Moreau",
    "Schmidt",
    "Rossi",
    "Andersen",
    "Petrov",
    "Garcia",
    "Nakamura",
    "Oduya",
    "Lindqvist",
    "Ferreira",
    "Kaur",
    "Dubois",
    "Varga",
    "Yilmaz",
    "Brennan",
    "Sato",
    "Mendes",
];
const CO_A: &[&str] = &["Acme", "Nimbus", "Vertex", "Quanta", "Helix", "Orbital", "Cobalt", "Lumen", "Aurora", "Atlas"];
const CO_B: &[&str] = &["Robotics", "Labs", "Systems", "Analytics", "Dynamics", "Networks", "Foundry", "Bio"];
const CODENAMES: &[&str] =
    &["Falcon", "Nebula", "Granite", "Mistral", "Phoenix", "Tundra", "Oasis", "Kestrel", "Zephyr", "Sequoia"];
const CITIES: &[&str] =
    &["Berlin", "Lisbon", "Tokyo", "Nairobi", "Toronto", "Seoul", "Austin", "Warsaw", "Oslo", "Lima", "Cairo", "Perth"];
const TOOLS: &[&str] = &["grep", "cargo test", "git diff", "curl", "pytest", "kubectl get pods", "ls", "psql"];
const MOODS: &[&str] = &["I suspect", "Remember that", "Noticed that", "Hypothesis:", "TODO:", "Lesson learned:"];
const THINGS: &[&str] = &[
    "the cache layer",
    "flaky integration tests",
    "the deploy script",
    "memory usage",
    "API latency",
    "the onboarding doc",
    "log rotation",
    "the billing job",
];

#[derive(Debug, Clone)]
pub struct Memory {
    pub text: String,
    pub topic: usize,
    pub vector: Vec<f32>,
    pub ts: u64,
    pub entities: Vec<String>,
    pub facts: Vec<FactInput>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub kind: String,
    pub text: String,
    pub ts: u64,
}

/// A query whose expected winner is known (time-decay validation).
#[derive(Debug, Clone)]
pub struct DecayProbe {
    pub query: Vec<f32>,
    pub expect_text: String,
    pub other_text: String,
}

#[derive(Debug, Clone, Default)]
pub struct World {
    pub people: Vec<String>,
    pub companies: Vec<String>,
    pub projects: Vec<String>,
    pub facts: Vec<FactInput>,
    pub memories: Vec<Memory>,
    pub events: Vec<Event>,
    pub probes: Vec<DecayProbe>,
}

/// Per-component Gaussian noise that yields expected cosine `c` in `dim` dims.
pub fn noise_for_cos(c: f32, dim: usize) -> f32 {
    ((1.0 / (c * c) - 1.0) / dim.max(1) as f32).sqrt()
}

fn fact(s: &str, p: &str, o: &str) -> FactInput {
    FactInput { subject: s.into(), predicate: p.into(), object: o.into() }
}

fn unique_name(parts: &[&[&str]], i: usize) -> String {
    let mut n = i;
    let mut words = Vec::new();
    for list in parts {
        words.push(list[n % list.len()].to_string());
        n /= list.len();
    }
    if n > 0 {
        words.push(format!("{}", n + 1));
    }
    words.join(" ")
}

/// Build the world. `chunks`/`events` are the number of semantic memories
/// and episodic events; topics are clusters in embedding space.
pub fn generate(seed: u64, dim: usize, chunks: usize, events: usize, now: u64, span_days: u64) -> World {
    let mut rng = Rng::new(seed);
    let mut w = World::default();
    let n_people = (chunks / 6).clamp(10, 50_000);
    w.people = (0..n_people).map(|i| unique_name(&[FIRST, LAST], i)).collect();
    w.companies = (0..(n_people / 25).max(2)).map(|i| unique_name(&[CO_A, CO_B], i)).collect();
    w.projects = (0..(n_people / 8).max(2)).map(|i| format!("Project {}", unique_name(&[CODENAMES], i))).collect();
    for (ci, c) in w.companies.iter().enumerate() {
        w.facts.push(fact(c, "located_in", CITIES[ci % CITIES.len()]));
    }
    for (pi, p) in w.projects.iter().enumerate() {
        w.facts.push(fact(p, "owned_by", &w.companies[pi % w.companies.len()]));
    }
    for (i, person) in w.people.iter().enumerate() {
        let company = &w.companies[i % w.companies.len()];
        w.facts.push(fact(person, "works_at", company));
        w.facts.push(fact(person, "lives_in", rng.pick(CITIES)));
        w.facts.push(fact(person, "member_of", &w.projects[i % w.projects.len()]));
        let manager = &w.people[i % w.companies.len()];
        if manager != person {
            w.facts.push(fact(person, "reports_to", manager));
        }
        if i % 3 == 1 {
            w.facts.push(fact(person, "sibling_of", &w.people[i - 1]));
        }
    }
    let topics = (chunks / 40).clamp(4, 2000);
    let centers: Vec<Vec<f32>> = (0..topics).map(|_| rng.unit_vector(dim)).collect();
    let noise = noise_for_cos(0.80, dim);
    let span = span_days * DAY_MS;
    for i in 0..chunks {
        let topic = rng.below(topics);
        let ts = now - rng.below(span as usize) as u64;
        let mut m = Memory {
            text: String::new(),
            topic,
            vector: rng.perturb(&centers[topic], noise),
            ts,
            entities: Vec::new(),
            facts: Vec::new(),
            tags: vec![format!("topic-{topic}")],
        };
        if i % 2 == 0 && !w.facts.is_empty() {
            let f = w.facts[rng.below(w.facts.len())].clone();
            m.text = format!("Note #{i}: {} {} {}.", f.subject, f.predicate.replace('_', " "), f.object);
            m.entities = vec![f.subject.clone(), f.object.clone()];
            m.facts.push(f);
            m.tags.push("fact".into());
        } else {
            m.text = format!("Thought #{i}: {} {} around day {}.", rng.pick(MOODS), rng.pick(THINGS), ts / DAY_MS);
            m.entities = Vec::new();
            m.tags.push("thought".into());
        }
        w.memories.push(m);
    }
    for i in 0..events {
        let kind = *rng.pick(&["thought", "action", "tool_result"]);
        let text = match kind {
            "tool_result" => format!("#{i} ran `{}` -> exit {}", rng.pick(TOOLS), rng.below(3)),
            "action" => format!("#{i} edited {} for {}", rng.pick(THINGS), rng.pick(&w.projects)),
            _ => format!("#{i} {} {}", rng.pick(MOODS), rng.pick(THINGS)),
        };
        w.events.push(Event { kind: kind.into(), text, ts: now - rng.below(span as usize) as u64 });
    }
    add_probes(&mut w, &mut rng, dim, now);
    w
}

/// Decay probes. Pairs A: same similarity (~0.85), fresh must beat old.
/// Pairs B: an old exact match (sim 1.0) must beat a fresh related one.
fn add_probes(w: &mut World, rng: &mut Rng, dim: usize, now: u64) {
    let near = noise_for_cos(0.85, dim);
    for i in 0..20 {
        let base = rng.unit_vector(dim);
        let exact_old = i % 2 == 1;
        let (old_vec, fresh_vec) = if exact_old {
            (base.clone(), rng.perturb(&base, near))
        } else {
            (rng.perturb(&base, near), rng.perturb(&base, near))
        };
        let old_text = format!("Probe {i}: archived observation");
        let fresh_text = format!("Probe {i}: recent observation");
        for (text, vector, age_days) in [(&old_text, old_vec, 200), (&fresh_text, fresh_vec, 1)] {
            w.memories.push(Memory {
                text: text.clone(),
                topic: usize::MAX,
                vector,
                ts: now - age_days * DAY_MS,
                entities: Vec::new(),
                facts: Vec::new(),
                tags: vec!["probe".into()],
            });
        }
        let (expect_text, other_text) = if exact_old { (old_text, fresh_text) } else { (fresh_text, old_text) };
        w.probes.push(DecayProbe { query: base, expect_text, other_text });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::dot;

    #[test]
    fn deterministic_and_well_formed() {
        let a = generate(1, 32, 400, 100, 10_000 * DAY_MS, 90);
        let b = generate(1, 32, 400, 100, 10_000 * DAY_MS, 90);
        assert_eq!(a.memories.len(), 440);
        assert_eq!(a.events.len(), 100);
        assert_eq!(a.memories[7].text, b.memories[7].text);
        assert!(a.facts.len() > a.people.len() * 2);
        let names: std::collections::HashSet<_> = a.people.iter().collect();
        assert_eq!(names.len(), a.people.len());
    }

    #[test]
    fn noise_hits_target_cosine() {
        let mut rng = Rng::new(5);
        let base = rng.unit_vector(256);
        let mean: f32 =
            (0..200).map(|_| dot(&base, &rng.perturb(&base, noise_for_cos(0.85, 256)))).sum::<f32>() / 200.0;
        assert!((mean - 0.85).abs() < 0.03, "{mean}");
    }
}
