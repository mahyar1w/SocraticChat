use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ureq::Agent;

pub const DEFAULT_GGUF_NAME: &str = "Qwen3.5-0.8B-Socratic-Q8_0.gguf";

pub fn find_llama_server(preferred: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if !preferred.trim().is_empty() {
        candidates.push(PathBuf::from(preferred.trim()));
    }
    if let Ok(env_bin) = std::env::var("LLAMA_SERVER_BIN")
        && !env_bin.trim().is_empty()
    {
        candidates.push(PathBuf::from(env_bin));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("llama-server"));
    }
    candidates.push(PathBuf::from("llama-server"));
    candidates.push(PathBuf::from("../llama-server"));
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(home).join(".unsloth/llama.cpp/llama-server"));
    }
    for path in candidates {
        if path.is_file() {
            return Some(path);
        }
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let p = dir.join("llama-server");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

pub fn find_default_gguf(preferred: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if !preferred.trim().is_empty() {
        candidates.push(PathBuf::from(preferred.trim()));
    }
    if let Ok(env_model) = std::env::var("SOCRATIC_GGUF")
        && !env_model.trim().is_empty()
    {
        candidates.push(PathBuf::from(env_model));
    }
    if let Ok(xdg_data) = std::env::var("XDG_DATA_HOME")
        && !xdg_data.trim().is_empty()
    {
        candidates.push(PathBuf::from(xdg_data).join("socratic-chat").join(DEFAULT_GGUF_NAME));
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        candidates.push(
            PathBuf::from(home)
                .join(".local/share/socratic-chat")
                .join(DEFAULT_GGUF_NAME),
        );
    }
    candidates.push(PathBuf::from(DEFAULT_GGUF_NAME));
    candidates.push(PathBuf::from("..").join(DEFAULT_GGUF_NAME));
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join(DEFAULT_GGUF_NAME));
        if let Some(parent) = dir.parent() {
            candidates.push(parent.join(DEFAULT_GGUF_NAME));
        }
    }
    candidates.into_iter().find(|p| p.is_file())
}

pub struct LlamaEngine {
    child: Child,
    base_url: String,
}

impl LlamaEngine {
    pub fn start(server_bin: &Path, gguf_path: &Path) -> Result<Self, String> {
        if !server_bin.is_file() {
            return Err(format!("llama-server not found at {}", server_bin.display()));
        }
        if !gguf_path.is_file() {
            return Err(format!("GGUF model not found at {}", gguf_path.display()));
        }

        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .map(|a| a.port())
            .map_err(|e| format!("Failed to allocate local port: {e}"))?;

        let mut cmd = Command::new(server_bin);
        cmd.arg("-m")
            .arg(gguf_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg("4096")
            .arg("--no-webui")
            .arg("--reasoning")
            .arg("off")
            .arg("--reasoning-budget")
            .arg("0")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        if let Some(bin_dir) = server_bin.parent().filter(|p| !p.as_os_str().is_empty()) {
            let mut ld = bin_dir.as_os_str().to_os_string();
            if let Ok(existing) = std::env::var("LD_LIBRARY_PATH")
                && !existing.is_empty()
            {
                ld.push(":");
                ld.push(existing);
            }
            cmd.env("LD_LIBRARY_PATH", ld);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to launch {}: {e}", server_bin.display()))?;

        // Drain stderr continuously so the 64 KB OS pipe buffer never fills and stalls llama-server.
        let stderr_tail = Arc::new(Mutex::new(VecDeque::<String>::with_capacity(16)));
        if let Some(stderr) = child.stderr.take() {
            let tail_bg = Arc::clone(&stderr_tail);
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    if let Ok(mut buf) = tail_bg.lock() {
                        if buf.len() == 12 {
                            buf.pop_front();
                        }
                        buf.push_back(line);
                    }
                }
            });
        }

        let base_url = format!("http://127.0.0.1:{port}/v1");
        let health_url = format!("http://127.0.0.1:{port}/health");
        let agent: Agent = Agent::config_builder()
            .proxy(None)
            .timeout_global(Some(Duration::from_secs(2)))
            .build()
            .into();

        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            if let Ok(Some(status)) = child.try_wait() {
                thread::sleep(Duration::from_millis(25));
                let tail = stderr_tail
                    .lock()
                    .map(|b| b.iter().cloned().collect::<Vec<_>>().join("\n"))
                    .unwrap_or_default();
                return Err(format!("llama-server exited ({status}):\n{tail}"));
            }
            if let Ok(resp) = agent.get(&health_url).call()
                && resp.status().as_u16() == 200
            {
                return Ok(Self { child, base_url });
            }
            thread::sleep(Duration::from_millis(150));
        }

        let _ = child.kill();
        let _ = child.wait();
        Err("Timed out waiting for local llama-server to become healthy".to_string())
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Message, RequestConfig, generate_reply};

    #[test]
    fn test_live_llama_cpp_roundtrip() {
        let Some(bin) = find_llama_server("") else {
            return;
        };
        let Some(gguf) = find_default_gguf("") else {
            return;
        };
        let eng = LlamaEngine::start(&bin, &gguf).expect("llama-server should start cleanly");
        let cfg = RequestConfig {
            use_teacher: false,
            teacher_url: String::new(),
            teacher_model: String::new(),
            teacher_key: String::new(),
            student_url: eng.base_url().to_string(),
            student_model: "local-socratic".to_string(),
            student_key: String::new(),
        };
        let reply = generate_reply(
            &[Message {
                role: "user".into(),
                content: "Hey! How's it going?".into(),
            }],
            &cfg,
        )
        .expect("local llama.cpp should reply");
        assert!(!reply.trim().is_empty(), "reply must not be empty");
    }
}
