//! Black-box tests of the memory CLI and `synapstore install` in a fake home.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Home(PathBuf);

impl Home {
    fn new(tag: &str) -> Home {
        let d = std::env::temp_dir().join(format!("synapstore-cli-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Home(d)
    }

    fn run(&self, args: &[&str]) -> Output {
        self.run_with(args, &[])
    }

    fn run_with(&self, args: &[&str], env: &[(&str, &str)]) -> Output {
        let mut c = Command::new(env!("CARGO_BIN_EXE_synapstore"));
        c.args(args)
            .env("HOME", &self.0)
            .env("USERPROFILE", &self.0)
            .env("APPDATA", self.0.join("AppData"))
            .env("XDG_CONFIG_HOME", "")
            .env("CODEX_HOME", "")
            .env_remove("SYNAPSTORE_DB")
            .env_remove("SYNAPSTORE_URL")
            .env_remove("SYNAPSTORE_TOKEN");
        for (k, v) in env {
            c.env(k, v);
        }
        c.output().unwrap()
    }

    /// Stdout of a successful run, with `/` as the path separator.
    fn ok(&self, args: &[&str]) -> String {
        let o = self.run(args);
        assert!(o.status.success(), "{args:?} failed: {}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8(o.stdout).unwrap().replace('\\', "/")
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_default()
}

#[test]
fn memory_commands_use_the_default_store() {
    let h = Home::new("mem");
    let out = h.ok(&["remember", "The user prefers dark roast coffee.", "--fact", "user|prefers|dark roast coffee"]);
    assert!(out.starts_with("remembered ") && out.contains("(new, 1 fact)"), "{out}");
    assert!(h.path(".synapstore/memory.redb").is_file(), "the default store is created");
    h.ok(&["log", "ran the tests", "--kind", "action"]);
    let recall = h.ok(&["recall", "coffee", "-k", "1"]);
    assert!(recall.starts_with("1. [") && recall.contains("dark roast coffee"), "{recall}");
    let graph = h.ok(&["graph", "user"]);
    assert!(graph.contains("user -[prefers]-> dark roast coffee"), "{graph}");
    assert!(h.ok(&["events", "--limit", "5"]).contains("action   ran the tests"));
    let v: serde_json::Value = serde_json::from_str(&h.ok(&["recall", "coffee", "--json"])).unwrap();
    let id = v["hits"][0]["id"].as_str().unwrap().to_string();
    assert_eq!(h.ok(&["forget", &id]).trim(), "forgot 1 memory");
    assert_eq!(h.ok(&["facts", "--predicate", "prefers"]).trim(), "user -[prefers]-> dark roast coffee");
    // An explicit store is separate from the default one.
    let other = h.path("other.redb");
    assert_eq!(h.ok(&["recall", "coffee", "--db", other.to_str().unwrap()]).trim(), "no matching memories");
    let bad = h.run(&["recall"]);
    assert_eq!(bad.status.code(), Some(2), "usage errors exit with 2");
    let skill = h.ok(&["skill"]).replace("\r\n", "\n");
    assert!(skill.starts_with("---\nname: synapstore\n"));
}

#[test]
fn install_wires_agents_and_uninstall_reverts() {
    let h = Home::new("install");
    std::fs::create_dir_all(h.path(".codex")).unwrap();
    let codex_cfg = "model = \"o3\"\n";
    std::fs::write(h.path(".codex/config.toml"), codex_cfg).unwrap();
    std::fs::create_dir_all(h.path(".gemini")).unwrap();
    let gemini_cfg = "{\n  // keep\n  \"theme\": \"dark\"\n}\n";
    std::fs::write(h.path(".gemini/settings.json"), gemini_cfg).unwrap();
    let clients = "codex,gemini,opencode,cursor,pi";
    let dry = h.ok(&["install", "--client", clients, "--dry-run"]);
    assert!(dry.contains("~/.cursor/mcp.json"));
    assert!(!h.path(".cursor").exists() && read(&h.path(".codex/config.toml")) == codex_cfg, "dry run writes nothing");

    let out = h.ok(&["install", "--client", clients]);
    assert!(out.contains("pi              skill  ~/.agents/skills/synapstore (shared"), "{out}");
    let toml = read(&h.path(".codex/config.toml"));
    assert!(toml.starts_with(codex_cfg) && toml.contains("[mcp_servers.synapstore]\ncommand = "), "{toml}");
    let gemini = read(&h.path(".gemini/settings.json"));
    assert!(gemini.contains("// keep") && gemini.contains("\"synapstore\": {"), "{gemini}");
    let cursor: serde_json::Value = serde_json::from_str(&read(&h.path(".cursor/mcp.json"))).unwrap();
    assert_eq!(cursor["mcpServers"]["synapstore"]["args"], serde_json::json!(["mcp"]));
    let exe = cursor["mcpServers"]["synapstore"]["command"].as_str().unwrap().to_string();
    assert!(Path::new(&exe).is_file(), "the MCP command is this binary: {exe}");
    let skill = read(&h.path(".agents/skills/synapstore/SKILL.md"));
    assert!(skill.contains("name: synapstore"));
    assert!(h.path(".config/opencode/skills/synapstore/SKILL.md").is_file());
    assert!(h.ok(&["install", "--client", clients]).contains("already up to date"), "idempotent");

    h.ok(&["uninstall", "--client", clients]);
    assert_eq!(read(&h.path(".codex/config.toml")), codex_cfg);
    assert!(!read(&h.path(".gemini/settings.json")).contains("synapstore"));
    assert!(!h.path(".agents/skills/synapstore").exists());
    assert!(h.path(".synapstore/memory.redb").is_file(), "uninstall keeps the memory");
}

#[test]
fn install_for_a_remote_server_saves_cli_settings() {
    let h = Home::new("remote");
    let refused = h.run(&["install", "--client", "cursor", "--url", "https://mem.example.com"]);
    assert!(!refused.status.success(), "a public server needs a token");
    let env = [("SYNAPSTORE_TOKEN", "syn_test")];
    let o = h.run_with(&["install", "--client", "cursor,windsurf", "--url", "https://mem.example.com"], &env);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let cursor: serde_json::Value = serde_json::from_str(&read(&h.path(".cursor/mcp.json"))).unwrap();
    let entry = &cursor["mcpServers"]["synapstore"];
    assert_eq!(entry["url"], "https://mem.example.com/mcp");
    assert_eq!(entry["headers"]["Authorization"], "Bearer syn_test");
    assert!(read(&h.path(".codeium/windsurf/mcp_config.json")).contains("\"serverUrl\""));
    let saved: serde_json::Value = serde_json::from_str(&read(&h.path(".synapstore/remote.json"))).unwrap();
    assert_eq!(saved["url"], "https://mem.example.com/mcp");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(h.path(".synapstore/remote.json")).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0, "the saved token is private");
    }
    h.ok(&["uninstall", "--client", "cursor,windsurf"]);
    assert!(!h.path(".synapstore/remote.json").exists());
}

#[cfg(unix)]
#[test]
fn claude_code_falls_back_to_its_state_file() {
    let h = Home::new("claude");
    std::fs::write(h.path(".claude.json"), "{\"numStartups\": 3}").unwrap();
    let o = h.run_with(&["install", "--client", "claude-code"], &[("PATH", "/nonexistent")]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let v: serde_json::Value = serde_json::from_str(&read(&h.path(".claude.json"))).unwrap();
    assert_eq!(v["numStartups"], 3);
    assert_eq!(v["mcpServers"]["synapstore"]["args"], serde_json::json!(["mcp"]));
    assert!(h.path(".claude/skills/synapstore/SKILL.md").is_file());
    assert!(h.path(".claude.json.synapstore.bak").is_file(), "the original is backed up");
}
