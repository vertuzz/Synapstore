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
    assert_eq!(p("--mcp").unwrap().db, None, "the default store");
    assert_eq!(p("--mcp --no-log x").unwrap().db, Some(PathBuf::from("x")));
    assert!(p("--mcp x --bogus").is_err());
    assert!(p("--mcp x --ef-search nope").is_err());
    assert!(p("--mcp x --half-life-days -3").is_err());
    assert!(p("stats a b").is_err());
    assert!(p("frobnicate x").is_err());
}

#[test]
fn optional_store_paths() {
    assert_eq!(p("stats").unwrap().db, None);
    assert_eq!(p("stats --db a").unwrap().db, Some(PathBuf::from("a")));
    assert!(p("stats --db a").unwrap().db_explicit);
    assert_eq!(p("token list").unwrap().db, None);
    let c = p("token revoke ci").unwrap();
    assert_eq!((c.db, c.command), (None, Command::Token(TokenCmd::Revoke { target: "ci".into() })));
    assert_eq!(p("mcp mem.redb").unwrap().db, Some(PathBuf::from("mem.redb")));
}

fn op(args: &[&str]) -> agent::Op {
    let c = parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap();
    let Command::Agent(op) = c.command else { panic!("not an agent command") };
    op
}

#[test]
fn parses_memory_commands() {
    let o = op(&["remember", "User likes tea", "--fact", "user | likes | tea", "--tag", "pref", "--json"]);
    assert_eq!(o.tool, "memorize");
    assert_eq!(
        o.args,
        serde_json::json!({"text": "User likes tea", "facts": [["user", "likes", "tea"]], "tags": ["pref"]})
    );
    assert!(o.json && !o.stdin_text);
    assert!(op(&["remember", "-"]).stdin_text);
    let r = op(&["recall", "how", "to", "deploy", "-k", "3", "--max-age", "7d"]);
    assert_eq!(r.args, serde_json::json!({"query": "how to deploy", "top_k": 3, "max_age": "7d"}));
    assert_eq!(op(&["graph", "Postgres", "--depth", "1"]).args, serde_json::json!({"subject": "Postgres", "depth": 1}));
    assert_eq!(op(&["fact", "rm", "a", "b", "c"]).tool, "retract_fact");
    assert_eq!(op(&["forget", "ab", "cd"]).args["ids"], serde_json::json!(["ab", "cd"]));
    assert_eq!(op(&["call", "stats", "{\"detailed\":true}"]).args, serde_json::json!({"detailed": true}));
    assert_eq!(op(&["events", "--url", "https://m.example"]).url.as_deref(), Some("https://m.example"));
    for bad in [
        &["remember"][..],
        &["recall"],
        &["remember", "x", "--fact", "a|b"],
        &["call", "stats", "[1]"],
        &["fact", "zap", "a", "b", "c"],
    ] {
        assert!(parse(&bad.iter().map(|s| s.to_string()).collect::<Vec<_>>()).is_err(), "{bad:?}");
    }
    assert_eq!(agent::endpoint("https://m.example/"), "https://m.example/mcp");
    assert_eq!(agent::endpoint("http://h:1/x/mcp"), "http://h:1/x/mcp");
}

#[test]
fn parses_install() {
    let Command::Install(o) = p("install --client codex,pi --only skill --dry-run").unwrap().command else { panic!() };
    assert_eq!(o.clients, vec!["codex", "pi"]);
    assert_eq!((o.only, o.dry_run, o.uninstall), (Some(crate::install::Part::Skill), true, false));
    let Command::Install(u) = p("uninstall --url https://m").unwrap().command else { panic!() };
    assert!(u.uninstall && u.url.as_deref() == Some("https://m"));
    assert!(p("install --only both").is_err());
    assert_eq!(p("skill").unwrap().command, Command::Skill);
}
