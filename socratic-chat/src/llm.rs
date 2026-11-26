use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use ureq::Agent;

pub const SOCRATIC_SYSTEM: &str = "You are a warm, witty conversationalist and Socratic mentor. \
In casual chat, greetings, venting, humor, or everyday recommendations, reply naturally \
and directly with your own personable twist. When helping the user learn, debug, or solve \
a problem, guide them step by step with one focused question or hint without spoiling the answer.";

pub const CHAT_TWIST_SYSTEM: &str = "You are a warm, witty conversational companion. Take the reference reply and put your own \
natural, lively, human spin on it—keep all concrete ideas, recommendations, and punchlines \
intact, and never turn casual chat or jokes into a quiz question.";

pub const TUTOR_TWIST_SYSTEM: &str = "You are a warm Socratic mentor. Use the reference answer for factual accuracy, then put \
your own Socratic twist on it: guide the user step by step with one focused question or \
small hint without giving away the final answer.";

pub const TEACHER_SYSTEM: &str = "Prefix your reply with [CHAT] if the user is casually chatting (greetings, smalltalk, venting, \
jokes, opinions, brainstorming, or everyday recommendations), or [TUTOR] if they are asking a \
math, coding, debugging, science, or conceptual 'how/why' question (or answering a tutoring step). \
Then answer directly, accurately, and concisely in 1-4 sentences.";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub use_teacher: bool,
    pub teacher_url: String,
    pub teacher_model: String,
    pub teacher_key: String,
    pub student_url: String,
    pub student_model: String,
    pub student_key: String,
}

pub fn strip_think(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + "<think>".len()..];
        if let Some(end) = after_open.find("</think>") {
            rest = &after_open[end + "</think>".len()..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    // Handle templates that pre-fill `<think>\n` in the prompt so output only contains `</think>`.
    if let Some(close) = out.rfind("</think>") {
        return out[close + "</think>".len()..].trim().to_string();
    }
    out.trim().to_string()
}

pub fn split_teacher_mode(raw_ref: &str) -> (&'static str, String) {
    let trimmed = raw_ref.trim_start();
    if trimmed
        .get(..6)
        .is_some_and(|p| p.eq_ignore_ascii_case("[chat]"))
    {
        return ("CHAT", trimmed[6..].trim().to_string());
    }
    if trimmed
        .get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case("[tutor]"))
    {
        return ("TUTOR", trimmed[7..].trim().to_string());
    }
    ("AUTO", trimmed.trim().to_string())
}

pub fn format_user_with_reference(user_text: &str, reference: Option<&str>, mode: &str) -> String {
    if let Some(r) = reference.map(str::trim).filter(|s| !s.is_empty()) {
        let header = match mode {
            "CHAT" => "[Reference — rewrite naturally in your own warm, witty voice; keep key details/punchlines, no quiz]",
            "TUTOR" => "[Reference — use for accuracy; guide with one Socratic question or hint without spoiling the answer]",
            _ => "[Reference — use these facts/ideas and put your own natural chat or Socratic twist on them]",
        };
        return format!("{header}\n{r}\n\n[User]\n{}", user_text.trim());
    }
    user_text.trim().to_string()
}

pub fn inject_reference(history: &[Message], reference: Option<&str>, mode: &str) -> Vec<Message> {
    let mut body: Vec<Message> = history
        .iter()
        .filter(|m| m.role != "system")
        .cloned()
        .collect();
    if let Some(last) = body.last_mut()
        && last.role == "user"
    {
        last.content = format_user_with_reference(&last.content, reference, mode);
    }
    let sys = match mode {
        "CHAT" => CHAT_TWIST_SYSTEM,
        "TUTOR" => TUTOR_TWIST_SYSTEM,
        _ => SOCRATIC_SYSTEM,
    };
    let mut out = Vec::with_capacity(body.len() + 1);
    out.push(Message {
        role: "system".to_string(),
        content: sys.to_string(),
    });
    out.extend(body);
    out
}

pub fn resolve_endpoint(base_url: &str) -> Result<(String, bool), String> {
    let mut url = base_url.trim().trim_end_matches('/').to_string();
    if url.is_empty() {
        return Err("Endpoint URL must not be empty".to_string());
    }
    if !url.contains("://") {
        url = format!("http://{url}");
    }
    let (scheme, after_scheme) = url
        .split_once("://")
        .ok_or_else(|| "Invalid endpoint URL".to_string())?;
    if !matches!(scheme.to_ascii_lowercase().as_str(), "http" | "https") {
        return Err(format!("Unsupported URL scheme {scheme:?}; use http or https"));
    }
    let path = after_scheme
        .find('/')
        .map(|i| after_scheme[i..].trim_end_matches('/'))
        .unwrap_or("");
    if path.ends_with("/api/chat") {
        return Ok((url, true));
    }
    if path.is_empty() {
        return Ok((format!("{url}/v1/chat/completions"), false));
    }
    if !path.ends_with("/chat/completions") {
        return Ok((format!("{url}/chat/completions"), false));
    }
    Ok((url, false))
}

fn is_loopback(url: &str) -> bool {
    let after = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = after.split('/').next().unwrap_or("");
    let host = if authority.starts_with('[') {
        authority.split(']').next().unwrap_or("").trim_start_matches('[')
    } else {
        authority.split(':').next().unwrap_or("")
    };
    matches!(host.to_ascii_lowercase().as_str(), "localhost" | "127.0.0.1" | "::1")
}

fn extract_content(val: Option<&Value>) -> String {
    match val {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| {
                item.as_object()
                    .and_then(|o| o.get("text"))
                    .and_then(Value::as_str)
                    .or_else(|| item.as_str())
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

pub fn parse_sse(raw_body: &str) -> Option<String> {
    let mut chunks = String::new();
    let mut saw_event = false;
    for line in raw_body.lines() {
        let trimmed = line.trim();
        let Some(payload) = trimmed.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            saw_event = true;
            continue;
        }
        let Ok(obj) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        saw_event = true;
        if let Some(msg) = obj.get("message") {
            chunks.push_str(&extract_content(msg.get("content")));
        } else if let Some(c0) = obj.get("choices").and_then(|c| c.get(0)) {
            let holder = c0.get("delta").or_else(|| c0.get("message"));
            chunks.push_str(&extract_content(holder.and_then(|h| h.get("content"))));
        }
    }
    saw_event.then_some(chunks)
}

pub fn chat_completion(
    base_url: &str,
    model: &str,
    messages: &[Message],
    api_key: &str,
    temperature: f32,
    max_tokens: u32,
) -> Result<String, String> {
    if model.trim().is_empty() {
        return Err("Model name must not be empty".to_string());
    }
    let (url, is_ollama_native) = resolve_endpoint(base_url)?;
    let payload = if is_ollama_native {
        json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "options": { "temperature": temperature, "num_predict": max_tokens }
        })
    } else {
        json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
    };

    let mut cfg = Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(120)))
        .http_status_as_error(false);
    if is_loopback(&url) {
        cfg = cfg.proxy(None);
    }
    let agent: Agent = cfg.build().into();

    let mut req = agent
        .post(&url)
        .header("Accept", "application/json")
        .header("User-Agent", "socratic-chat/0.1");
    if !api_key.trim().is_empty() {
        req = req.header("Authorization", &format!("Bearer {}", api_key.trim()));
    }

    let mut resp = req
        .send_json(&payload)
        .map_err(|e| format!("Cannot reach {url} (model={model:?}): {e}"))?;

    let status = resp.status().as_u16();
    let raw_body = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed reading response from {url}: {e}"))?;
    let trimmed = raw_body.trim();

    if status >= 400 {
        let preview: String = trimmed.chars().take(260).collect();
        return Err(format!("HTTP {status} from {url} (model={model:?}): {preview}"));
    }
    if trimmed.is_empty() {
        return Err(format!("Empty response from {url} (model={model:?})"));
    }

    let data: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(_) => {
            if let Some(sse) = parse_sse(trimmed) {
                return Ok(strip_think(&sse));
            }
            let preview: String = trimmed.chars().take(180).collect();
            return Err(format!(
                "Non-JSON response from {url} (model={model:?}): {preview:?}"
            ));
        }
    };

    if let Some(err) = data.get("error").filter(|v| !v.is_null()) {
        return Err(format!("API error from {url} (model={model:?}): {err}"));
    }

    let raw = if let Some(msg) = data.get("message") {
        extract_content(msg.get("content"))
    } else if let Some(c0) = data.get("choices").and_then(|c| c.get(0)) {
        extract_content(c0.get("message").and_then(|m| m.get("content")))
    } else {
        String::new()
    };
    Ok(strip_think(&raw))
}

pub fn generate_reply_with_ref(
    history: &[Message],
    cfg: &RequestConfig,
) -> Result<(Option<String>, String), String> {
    let mut mode = "AUTO";
    let mut reference = String::new();

    if cfg.use_teacher && !cfg.teacher_url.trim().is_empty() && !cfg.teacher_model.trim().is_empty() {
        let mut teacher_msgs = Vec::with_capacity(history.len() + 1);
        teacher_msgs.push(Message {
            role: "system".to_string(),
            content: TEACHER_SYSTEM.to_string(),
        });
        teacher_msgs.extend(history.iter().cloned());

        let raw_ref = chat_completion(
            &cfg.teacher_url,
            &cfg.teacher_model,
            &teacher_msgs,
            &cfg.teacher_key,
            0.3,
            1536,
        )
        .map_err(|e| format!("Teacher step failed — {e}"))?;
        let (m, r) = split_teacher_mode(&raw_ref);
        mode = m;
        reference = r;
    }

    let ref_opt = (!reference.is_empty()).then_some(reference.as_str());
    let student_msgs =
        inject_reference(history, ref_opt, if ref_opt.is_some() { mode } else { "AUTO" });

    let mut twist = chat_completion(
        &cfg.student_url,
        &cfg.student_model,
        &student_msgs,
        &cfg.student_key,
        0.6,
        768,
    )
    .map_err(|e| format!("Student step failed — {e}"))?;

    if twist.is_empty() {
        twist = chat_completion(
            &cfg.student_url,
            &cfg.student_model,
            &student_msgs,
            &cfg.student_key,
            0.4,
            1024,
        )
        .map_err(|e| format!("Student retry failed — {e}"))?;
    }

    if twist.is_empty() {
        if mode == "CHAT" && !reference.is_empty() {
            return Ok((Some(reference.clone()), reference));
        }
        return Err("Student model returned an empty reply".to_string());
    }
    let out_ref = (!reference.is_empty()).then_some(reference);
    Ok((out_ref, twist))
}

pub fn generate_reply(history: &[Message], cfg: &RequestConfig) -> Result<String, String> {
    generate_reply_with_ref(history, cfg).map(|(_, twist)| twist)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn test_resolve_endpoint() {
        assert_eq!(
            resolve_endpoint("http://localhost:20128").unwrap(),
            ("http://localhost:20128/v1/chat/completions".into(), false)
        );
        assert_eq!(
            resolve_endpoint("127.0.0.1:8080/v1").unwrap(),
            ("http://127.0.0.1:8080/v1/chat/completions".into(), false)
        );
        assert_eq!(
            resolve_endpoint("http://localhost:11434/api/chat").unwrap(),
            ("http://localhost:11434/api/chat".into(), true)
        );
        assert!(resolve_endpoint("file:///etc/passwd").is_err());
    }

    #[test]
    fn test_strip_think_and_mode() {
        assert_eq!(strip_think("<think>abc</think>\nHello"), "Hello");
        assert_eq!(strip_think("<think>cut off mid thought"), "");
        assert_eq!(strip_think("prefilled reasoning\n</think>\n[Chat] Hi!"), "[Chat] Hi!");
        assert_eq!(split_teacher_mode("[CHAT] Hey there!"), ("CHAT", "Hey there!".into()));
        assert_eq!(split_teacher_mode("[Chat] Hey there!"), ("CHAT", "Hey there!".into()));
        assert_eq!(split_teacher_mode("[Tutor] Speed = d/t"), ("TUTOR", "Speed = d/t".into()));
        assert_eq!(split_teacher_mode("Plain reply"), ("AUTO", "Plain reply".into()));
    }

    #[test]
    fn test_two_step_sse_and_retry() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let student_attempts = Arc::new(Mutex::new(0usize));
        let attempts_bg = Arc::clone(&student_attempts);

        thread::spawn(move || {
            for stream in listener.incoming().take(3) {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut content_len = 0usize;
                let mut line = String::new();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        content_len = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0u8; content_len];
                reader.read_exact(&mut body).unwrap();
                let req: Value = serde_json::from_slice(&body).unwrap();
                let model = req["model"].as_str().unwrap_or("");

                let (ctype, payload) = if model == "big-teacher" {
                    (
                        "text/event-stream",
                        "data: {\"choices\":[{\"delta\":{\"content\":\"[Tutor] Light is much faster \"}}]}\n\n\
                         data: {\"choices\":[{\"delta\":{\"content\":\"than sound.\"}}]}\n\n\
                         data: [DONE]\n\n"
                            .to_string(),
                    )
                } else {
                    let mut n = attempts_bg.lock().unwrap();
                    *n += 1;
                    let content = if *n == 1 {
                        "<think>unfinished"
                    } else {
                        "Which travels faster through air: light or sound?"
                    };
                    (
                        "application/json",
                        json!({"choices":[{"message":{"content": content}}]}).to_string(),
                    )
                };

                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                stream.write_all(resp.as_bytes()).unwrap();
            }
        });

        let cfg = RequestConfig {
            use_teacher: true,
            teacher_url: format!("http://127.0.0.1:{port}"),
            teacher_model: "big-teacher".into(),
            teacher_key: String::new(),
            student_url: format!("http://127.0.0.1:{port}/v1"),
            student_model: "socratic".into(),
            student_key: String::new(),
        };
        let history = vec![Message {
            role: "user".into(),
            content: "Why do we see lightning before thunder?".into(),
        }];
        let (reference, reply) = generate_reply_with_ref(&history, &cfg).unwrap();
        assert_eq!(reference.as_deref(), Some("Light is much faster than sound."));
        assert_eq!(reply, "Which travels faster through air: light or sound?");
        assert_eq!(*student_attempts.lock().unwrap(), 2);
    }
}
