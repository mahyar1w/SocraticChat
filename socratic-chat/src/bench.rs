use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};

const TERMUX_FILES: &str = "/sdcard/Android/data/com.termux/files";
const DEFAULT_MODEL_NAME: &str = "Qwen3.5-0.8B-Socratic-Q8_0.gguf";

fn ensure_exec_bin(src: &Path, dst_dir: &Path, name: &str) -> Result<PathBuf, String> {
    if !src.is_file() {
        return Err(format!("Binary not found: {}", src.display()));
    }
    fs::create_dir_all(dst_dir).map_err(|e| format!("Cannot create {}: {e}", dst_dir.display()))?;
    let dst = dst_dir.join(name);
    let needs_copy = match (fs::metadata(src), fs::metadata(&dst)) {
        (Ok(s), Ok(d)) => s.len() != d.len(),
        _ => true,
    };
    if needs_copy {
        fs::copy(src, &dst)
            .map_err(|e| format!("Failed copying {} -> {}: {e}", src.display(), dst.display()))?;
    }
    let _ = fs::set_permissions(&dst, fs::Permissions::from_mode(0o755));
    Ok(dst)
}

fn run_single_bench(
    bin_path: &Path,
    model: &Path,
    threads: u32,
    ngl: u32,
    n_prompt: u32,
    n_gen: u32,
    reps: u32,
) -> Result<Vec<Value>, String> {
    let out = Command::new(bin_path)
        .args([
            "-m",
            &model.display().to_string(),
            "-p",
            &n_prompt.to_string(),
            "-n",
            &n_gen.to_string(),
            "-r",
            &reps.to_string(),
            "-t",
            &threads.to_string(),
            "-ngl",
            &ngl.to_string(),
            "-o",
            "json",
        ])
        .output()
        .map_err(|e| format!("Failed running {}: {e}", bin_path.display()))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!("exit status {}", out.status)
        } else {
            stderr
        });
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let Some(start) = stdout.find('[') else {
        let preview: String = stdout.chars().take(200).collect();
        return Err(format!("No JSON output from {}: {preview}", bin_path.display()));
    };
    serde_json::from_str::<Vec<Value>>(&stdout[start..])
        .map_err(|e| format!("Invalid JSON from {}: {e}", bin_path.display()))
}

fn extract_speeds(rows: &[Value]) -> (String, String, String) {
    let pp = rows
        .iter()
        .find(|r| r.get("n_prompt").and_then(Value::as_u64).unwrap_or(0) > 0)
        .and_then(|r| Some((r.get("avg_ts")?.as_f64()?, r.get("stddev_ts")?.as_f64()?)))
        .map(|(avg, sd)| format!("{avg:.2} ± {sd:.2}"))
        .unwrap_or_else(|| "-".to_string());
    let tg = rows
        .iter()
        .find(|r| r.get("n_gen").and_then(Value::as_u64).unwrap_or(0) > 0)
        .and_then(|r| Some((r.get("avg_ts")?.as_f64()?, r.get("stddev_ts")?.as_f64()?)))
        .map(|(avg, sd)| format!("{avg:.2} ± {sd:.2}"))
        .unwrap_or_else(|| "-".to_string());
    let info = rows
        .first()
        .and_then(|r| {
            r.get("gpu_info")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .or_else(|| r.get("backends").and_then(Value::as_str).filter(|s| !s.is_empty()))
        })
        .unwrap_or("-")
        .to_string();
    (pp, tg, info)
}

pub fn run_bench(args: &[String]) -> i32 {
    let mut model = PathBuf::from(TERMUX_FILES).join(DEFAULT_MODEL_NAME);
    let mut cpu_bin_arg: Option<PathBuf> = None;
    let mut vk_bin_arg: Option<PathBuf> = None;
    let mut n_prompt = 128u32;
    let mut n_gen = 64u32;
    let mut reps = 3u32;
    let mut threads: Vec<u32> = vec![2, 4];
    let mut out_path = PathBuf::from("bench_results.json");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: socratic-chat bench [OPTIONS]\n\n\
                     Options:\n  \
                       -m, --model PATH       Path to GGUF model\n  \
                       --cpu-bin PATH         Explicit path to CPU llama-bench\n  \
                       --vulkan-bin PATH      Explicit path to Vulkan llama-bench\n  \
                       -p, --n-prompt N       Prompt tokens (default: 128)\n  \
                       -n, --n-gen N          Generated tokens (default: 64)\n  \
                       -r, --reps N           Repetitions (default: 3)\n  \
                       --threads N[,N...]     Comma-separated thread counts (default: 2,4)\n  \
                       --out PATH             Output JSON file (default: bench_results.json)"
                );
                return 0;
            }
            "-m" | "--model" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    model = PathBuf::from(v);
                }
            }
            "--cpu-bin" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cpu_bin_arg = Some(PathBuf::from(v));
                }
            }
            "--vulkan-bin" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    vk_bin_arg = Some(PathBuf::from(v));
                }
            }
            "-p" | "--n-prompt" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    n_prompt = v;
                }
            }
            "-n" | "--n-gen" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    n_gen = v;
                }
            }
            "-r" | "--reps" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    reps = v;
                }
            }
            "--threads" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    let parsed: Vec<u32> = v
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if !parsed.is_empty() {
                        threads = parsed;
                    }
                }
            }
            "--out" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    out_path = PathBuf::from(v);
                }
            }
            other => {
                eprintln!("Unknown argument: {other}");
                return 2;
            }
        }
        i += 1;
    }

    if !model.is_file() {
        eprintln!("Model not found: {}", model.display());
        return 1;
    }

    let work_dir = env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("llama-bench-bin");

    let cpu_bin = match cpu_bin_arg {
        Some(p) => p,
        None => match ensure_exec_bin(
            &PathBuf::from(TERMUX_FILES).join("llama-cpp-android-cpu/llama-bench"),
            &work_dir.join("cpu"),
            "llama-bench",
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        },
    };
    let vk_bin = match vk_bin_arg {
        Some(p) => p,
        None => match ensure_exec_bin(
            &PathBuf::from(TERMUX_FILES).join("llama-cpp-android-vulkan/llama-bench"),
            &work_dir.join("vulkan"),
            "llama-bench",
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        },
    };

    if let Ok(dev) = Command::new(&vk_bin).arg("--list-devices").output() {
        println!("=== Vulkan Devices ===");
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&dev.stdout),
            String::from_utf8_lossy(&dev.stderr)
        );
        let text = combined.trim();
        println!("{}\n", if text.is_empty() { "(no output)" } else { text });
    }

    let mut cases: Vec<(String, PathBuf, u32, u32)> = threads
        .iter()
        .map(|&t| (format!("cpu-t{t}"), cpu_bin.clone(), t, 0))
        .collect();
    cases.push((format!("vulkan-t{}-ngl99", threads[0]), vk_bin, threads[0], 99));

    let mut all_results: Vec<Value> = Vec::new();
    let mut summary: Vec<(String, u32, u32, String, String, String)> = Vec::new();

    for (label, bin_path, t, ngl) in cases {
        println!("Running {label}...");
        match run_single_bench(&bin_path, &model, t, ngl, n_prompt, n_gen, reps) {
            Ok(rows) => {
                let (pp, tg, info) = extract_speeds(&rows);
                all_results.push(json!({ "case": label, "status": "ok", "rows": rows }));
                summary.push((label, t, ngl, pp, tg, info));
            }
            Err(e) => {
                let tail: String = e
                    .lines()
                    .last()
                    .unwrap_or(&e)
                    .chars()
                    .take(60)
                    .collect();
                all_results.push(json!({ "case": label, "status": "error", "error": e }));
                summary.push((label, t, ngl, "ERR".into(), "ERR".into(), tail));
            }
        }
    }

    if let Ok(raw) = serde_json::to_string_pretty(&all_results) {
        let _ = fs::write(&out_path, raw);
    }

    let model_name = model
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| model.display().to_string());
    println!("\nModel: {model_name} (pp={n_prompt}, tg={n_gen}, reps={reps})");
    let header = format!(
        "{:<18} {:>2} {:>3}  {:<18} {:<18} Backend/GPU",
        "Case", "T", "ngl", "pp (tok/s)", "tg (tok/s)"
    );
    println!("{header}");
    println!("{}", "-".repeat(header.len()));
    for (label, t, ngl, pp, tg, info) in summary {
        println!("{label:<18} {t:>2} {ngl:>3}  {pp:<18} {tg:<18} {info}");
    }
    println!("\nSaved raw results to {}", out_path.display());
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_speeds_handles_valid_and_empty() {
        let rows = vec![
            json!({"n_prompt": 128, "n_gen": 0, "avg_ts": 420.5, "stddev_ts": 3.25, "gpu_info": "Adreno"}),
            json!({"n_prompt": 0, "n_gen": 64, "avg_ts": 38.12, "stddev_ts": 0.44, "gpu_info": "Adreno"}),
        ];
        let (pp, tg, info) = extract_speeds(&rows);
        assert_eq!(pp, "420.50 ± 3.25");
        assert_eq!(tg, "38.12 ± 0.44");
        assert_eq!(info, "Adreno");

        assert_eq!(extract_speeds(&[]), ("-".into(), "-".into(), "-".into()));
    }
}
