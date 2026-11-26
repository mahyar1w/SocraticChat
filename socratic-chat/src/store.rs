use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::llama::{find_default_gguf, find_llama_server};
use crate::llm::Message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub gguf_path: String,
    pub llama_server_path: String,
    pub use_teacher: bool,
    pub teacher_url: String,
    pub teacher_model: String,
    #[serde(skip)]
    pub teacher_key: String,
    pub dark_mode: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        let teacher_url = std::env::var("TEACHER_URL").unwrap_or_default();
        let teacher_model = std::env::var("TEACHER_MODEL").unwrap_or_default();
        let use_teacher = !teacher_url.trim().is_empty() && !teacher_model.trim().is_empty();
        Self {
            gguf_path: find_default_gguf("")
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "Qwen3.5-0.8B-Socratic-Q8_0.gguf".to_string()),
            llama_server_path: find_llama_server("")
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "llama-server".to_string()),
            use_teacher,
            teacher_url,
            teacher_model,
            teacher_key: std::env::var("TEACHER_API_KEY")
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .unwrap_or_default(),
            dark_mode: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: u64,
    pub title: String,
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedState {
    pub config: AppConfig,
    pub conversations: Vec<Conversation>,
    pub active_id: u64,
    pub next_id: u64,
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            config: AppConfig::default(),
            conversations: vec![Conversation {
                id: 1,
                title: "New Chat".to_string(),
                messages: Vec::new(),
            }],
            active_id: 1,
            next_id: 2,
        }
    }
}

fn state_path() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME")
        && !xdg.trim().is_empty()
    {
        return PathBuf::from(xdg).join("socratic-chat/state.json");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home).join(".config/socratic-chat/state.json");
    }
    PathBuf::from("socratic-chat-state.json")
}

fn repair_state(mut state: PersistedState) -> PersistedState {
    if state.config.teacher_key.is_empty() {
        state.config.teacher_key = std::env::var("TEACHER_API_KEY")
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .unwrap_or_default();
    }
    if !PathBuf::from(&state.config.gguf_path).is_file()
        && let Some(p) = find_default_gguf("")
    {
        state.config.gguf_path = p.display().to_string();
    }
    if !PathBuf::from(&state.config.llama_server_path).is_file()
        && let Some(p) = find_llama_server("")
    {
        state.config.llama_server_path = p.display().to_string();
    }
    if state.conversations.is_empty() {
        state.conversations.push(Conversation {
            id: state.next_id,
            title: "New Chat".to_string(),
            messages: Vec::new(),
        });
        state.active_id = state.next_id;
        state.next_id += 1;
    } else if !state.conversations.iter().any(|c| c.id == state.active_id) {
        state.active_id = state.conversations[0].id;
    }
    state
}

fn load_state_from(path: &Path) -> PersistedState {
    let state: PersistedState = fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default();
    repair_state(state)
}

fn save_state_to(path: &Path, state: &PersistedState) {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = fs::create_dir_all(parent);
    }
    let Ok(raw) = serde_json::to_string_pretty(state) else {
        return;
    };
    // Write to a sibling temp file and rename atomically so a crash never truncates state.json.
    let tmp = path.with_extension("json.tmp");
    if fs::write(&tmp, raw).is_ok() {
        let _ = fs::rename(&tmp, path);
    }
}

pub fn load_state() -> PersistedState {
    load_state_from(&state_path())
}

pub fn save_state(state: &PersistedState) {
    save_state_to(&state_path(), state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_save_omits_teacher_key_and_recovers_state() {
        let dir = std::env::temp_dir().join(format!("socratic-store-test-{}", std::process::id()));
        let path = dir.join("state.json");
        let _ = fs::remove_dir_all(&dir);

        let mut state = PersistedState::default();
        state.config.teacher_key = "sk-super-secret-key".to_string();
        state.conversations[0].title = "Binary search".to_string();
        state.conversations[0].messages.push(Message {
            role: "user".into(),
            content: "Why O(log n)?".into(),
        });

        save_state_to(&path, &state);
        let raw = fs::read_to_string(&path).expect("state.json should exist");
        assert!(!raw.contains("sk-super-secret-key"), "teacher_key must never be written to disk");
        assert!(!path.with_extension("json.tmp").exists(), "temp file should be renamed away");

        let loaded = load_state_from(&path);
        assert_eq!(loaded.conversations[0].title, "Binary search");
        assert_eq!(loaded.conversations[0].messages.len(), 1);

        // Empty conversations + corrupt JSON recovery.
        fs::write(&path, "{broken json").unwrap();
        let recovered = load_state_from(&path);
        assert_eq!(recovered.conversations.len(), 1);
        assert_eq!(recovered.active_id, recovered.conversations[0].id);

        let _ = fs::remove_dir_all(&dir);
    }
}
