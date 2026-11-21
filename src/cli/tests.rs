use super::*;

fn p(s: &str) -> Result<Cli, String> {
    parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
}

#[test]
fn parses_mcp_with_defaults() {
    let c = p("--mcp mem.redb --ef-search 99").unwrap();
    assert_eq!(c.command, Command::Mcp);
    assert_eq!(c.db, Some(PathBuf::from("mem.redb")));
    assert_eq!(c.config.hnsw.ef_search, 99);
    assert_eq!(c.config.log.unwrap().path, PathBuf::from("mem.redb.log"));
    assert_eq!(p("serve mem.redb").unwrap().command, Command::Mcp);
}

#[test]
fn parses_subcommands() {
    assert_eq!(p("stats x --detailed").unwrap().command, Command::Stats { detailed: true });
    assert_eq!(p("dump x").unwrap().command, Command::Dump { vectors: false });
    assert_eq!(p("synth x --records 5").unwrap().command, Command::Synth { records: 5, seed: 42 });
    assert!(p("stats x").unwrap().config.log.is_none());
    assert!(p("--mcp x --no-log").unwrap().config.log.is_none());
    assert_eq!(p("").unwrap().command, Command::Help);
    assert_eq!(p("--version").unwrap().command, Command::Version);
    assert_eq!(p("models").unwrap().command, Command::Models);
    assert_eq!(p("pull minilm").unwrap().command, Command::Pull { model: "minilm".into() });
}

#[test]
fn parses_http_and_tokens() {
    let c = p("serve db --listen 0.0.0.0:9000 --allow-origin https://app.example/ --threads 4").unwrap();
    let Command::Http(o) = c.command else { panic!("not http") };
    assert_eq!((o.listen.as_str(), o.threads), ("0.0.0.0:9000", 4));
    assert_eq!(o.allowed_origins, vec!["https://app.example"]);
    assert!(c.config.log.is_some());
    assert_eq!(
        p("token create db --name ci --scope read --expires-days 30").unwrap().command,
        Command::Token(TokenCmd::Create { name: "ci".into(), scope: "read".into(), expires_days: Some(30) })
    );
    assert_eq!(p("token list db").unwrap().command, Command::Token(TokenCmd::List));
    assert_eq!(p("token revoke db ci").unwrap().command, Command::Token(TokenCmd::Revoke { target: "ci".into() }));
    assert!(p("token create db").is_err() && p("token db").is_err() && p("token zap db").is_err());
}

#[test]
fn parses_embedder() {
    let c = p("--mcp x --embedder minilm").unwrap();
    assert_eq!(c.config.embedder.label().as_deref(), Some("minilm"));
    assert!(p("--mcp x --embedder openai:").is_err());
}

#[test]
fn rejects_garbage() {
    assert!(p("--mcp").is_err());
    assert_eq!(p("--mcp --no-log x").unwrap().db, Some(PathBuf::from("x")));
    assert!(p("--mcp x --bogus").is_err());
    assert!(p("--mcp x --ef-search nope").is_err());
    assert!(p("--mcp x --half-life-days -3").is_err());
    assert!(p("stats a b").is_err());
    assert!(p("frobnicate x").is_err());
}
