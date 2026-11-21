//! A tiny, deterministic BERT model written in Hugging Face layout, so the
//! neural-embedder plumbing can be tested without downloading weights.

use std::path::Path;

pub const WORDS: &[&str] = &[
    "alice", "bob", "coffee", "tea", "likes", "drinks", "rust", "python", "database", "memory", "agent", "the", "a",
    "is", "and", "dark", "roast", "green", "fast", "safe", "##s", "##ing", ".", ",",
];

pub fn files(hidden: usize) -> Vec<(String, Vec<u8>)> {
    let mut vocab = vec!["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
    vocab.extend_from_slice(WORDS);
    let (v, h, inter, pos) = (vocab.len(), hidden, hidden * 2, 64);
    let config = format!(
        r#"{{"model_type":"bert","vocab_size":{v},"hidden_size":{h},"num_hidden_layers":1,"num_attention_heads":2,
"intermediate_size":{inter},"max_position_embeddings":{pos},"type_vocab_size":2,"layer_norm_eps":1e-12,"hidden_act":"gelu"}}"#
    );
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        ("embeddings.word_embeddings.weight".into(), vec![v, h]),
        ("embeddings.position_embeddings.weight".into(), vec![pos, h]),
        ("embeddings.token_type_embeddings.weight".into(), vec![2, h]),
        ("embeddings.LayerNorm.weight".into(), vec![h]),
        ("embeddings.LayerNorm.bias".into(), vec![h]),
    ];
    let p = "encoder.layer.0";
    for (name, out, inp) in [
        ("attention.self.query", h, h),
        ("attention.self.key", h, h),
        ("attention.self.value", h, h),
        ("attention.output.dense", h, h),
        ("intermediate.dense", inter, h),
        ("output.dense", h, inter),
    ] {
        tensors.push((format!("{p}.{name}.weight"), vec![out, inp]));
        tensors.push((format!("{p}.{name}.bias"), vec![out]));
    }
    for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
        tensors.push((format!("{p}.{name}.weight"), vec![h]));
        tensors.push((format!("{p}.{name}.bias"), vec![h]));
    }
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for (i, (name, shape)) in tensors.iter().enumerate() {
        let n: usize = shape.iter().product();
        let start = data.len();
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let x = if name.ends_with("LayerNorm.weight") {
                1.0
            } else {
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8
            };
            data.extend_from_slice(&x.to_le_bytes());
        }
        let shape_s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            r#"{}"{name}":{{"dtype":"F32","shape":[{}],"data_offsets":[{start},{}]}}"#,
            if i > 0 { "," } else { "" },
            shape_s.join(","),
            data.len()
        ));
    }
    header.push('}');
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header.as_bytes());
    st.extend_from_slice(&data);
    vec![
        ("config.json".into(), config.into_bytes()),
        ("vocab.txt".into(), (vocab.join("\n") + "\n").into_bytes()),
        ("model.safetensors".into(), st),
    ]
}

pub fn write(dir: &Path, hidden: usize) {
    std::fs::create_dir_all(dir).unwrap();
    for (name, bytes) in files(hidden) {
        std::fs::write(dir.join(name), bytes).unwrap();
    }
}
