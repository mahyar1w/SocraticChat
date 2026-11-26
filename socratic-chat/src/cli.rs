use std::env;
use std::io::{self, Write};

use crate::llm::{Message, RequestConfig, generate_reply_with_ref};

fn env_or(key: &str, default: &str) -> String {
    env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

pub fn run_ask(args: &[String]) -> i32 {
    let mut prompt: Option<String> = None;
    let mut teacher_url = env_or("TEACHER_URL", "http://localhost:11434/v1");
    let mut teacher_model = env_or("TEACHER_MODEL", "qwen3:8b");
    let mut teacher_key = env::var("TEACHER_API_KEY")
        .or_else(|_| env::var("OPENAI_API_KEY"))
        .unwrap_or_default();
    let mut student_url = env_or("STUDENT_URL", "http://localhost:11434/v1");
    let mut student_model = env_or("STUDENT_MODEL", "socratic");
    let mut student_key = env::var("STUDENT_API_KEY").unwrap_or_default();
    let mut use_teacher = true;
    let mut show_reference = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_ask_help();
                return 0;
            }
            "--teacher-url" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    teacher_url = v.clone();
                }
            }
            "--teacher-model" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    teacher_model = v.clone();
                }
            }
            "--teacher-key" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    teacher_key = v.clone();
                }
            }
            "--student-url" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    student_url = v.clone();
                }
            }
            "--student-model" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    student_model = v.clone();
                }
            }
            "--student-key" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    student_key = v.clone();
                }
            }
            "--no-teacher" => use_teacher = false,
            "--show-reference" => show_reference = true,
            other if !other.starts_with('-') && prompt.is_none() => {
                prompt = Some(other.to_string());
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_ask_help();
                return 2;
            }
        }
        i += 1;
    }

    let cfg = RequestConfig {
        use_teacher,
        teacher_url,
        teacher_model,
        teacher_key,
        student_url,
        student_model,
        student_key,
    };

    if let Some(p) = prompt {
        let history = vec![Message {
            role: "user".into(),
            content: p,
        }];
        match generate_reply_with_ref(&history, &cfg) {
            Ok((reference, twist)) => {
                if show_reference
                    && let Some(r) = reference
                {
                    println!("[Teacher ({})]\n{r}\n", cfg.teacher_model);
                }
                println!("{twist}");
                return 0;
            }
            Err(e) => {
                eprintln!("Error: {e}");
                return 1;
            }
        }
    }

    if cfg.use_teacher {
        println!("Teacher: {} @ {}", cfg.teacher_model, cfg.teacher_url);
    } else {
        println!("Teacher: disabled (standalone 0.8B)");
    }
    println!(
        "Student: {} @ {} (Ctrl-D or 'exit' to quit)\n",
        cfg.student_model, cfg.student_url
    );

    let mut history: Vec<Message> = Vec::new();
    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        print!("You: ");
        let _ = io::stdout().flush();
        line.clear();
        match stdin.read_line(&mut line) {
            Ok(0) => {
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Input error: {e}");
                break;
            }
        }
        let user_msg = line.trim();
        if user_msg.is_empty() {
            continue;
        }
        if matches!(user_msg.to_ascii_lowercase().as_str(), "exit" | "quit") {
            break;
        }
        history.push(Message {
            role: "user".into(),
            content: user_msg.to_string(),
        });
        match generate_reply_with_ref(&history, &cfg) {
            Ok((reference, twist)) => {
                history.push(Message {
                    role: "assistant".into(),
                    content: twist.clone(),
                });
                if show_reference
                    && let Some(r) = reference
                {
                    println!("\n[Reference]\n{r}");
                }
                println!("\nSocratic: {twist}\n");
            }
            Err(e) => {
                history.pop();
                eprintln!("\nError: {e}\n");
            }
        }
    }
    0
}

fn print_ask_help() {
    println!(
        "Usage: socratic-chat ask [PROMPT] [OPTIONS]\n\n\
         Options:\n  \
           --teacher-url URL      Teacher OpenAI/Ollama URL (env: TEACHER_URL)\n  \
           --teacher-model MODEL  Teacher model name (env: TEACHER_MODEL)\n  \
           --teacher-key KEY      Teacher API key (env: TEACHER_API_KEY / OPENAI_API_KEY)\n  \
           --no-teacher           Skip teacher step and call student directly\n  \
           --student-url URL      Student OpenAI/Ollama URL (env: STUDENT_URL)\n  \
           --student-model MODEL  Student model name (env: STUDENT_MODEL)\n  \
           --student-key KEY      Student API key (env: STUDENT_API_KEY)\n  \
           --show-reference       Print the teacher's reference answer before the twist"
    );
}
