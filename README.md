# SocraticChat

A lightweight local desktop & CLI chat app built in Rust around a fine-tuned **Qwen3.5-0.8B** Socratic tutor and conversationalist, backed by a bundled `llama.cpp` (`llama-server`) engine.

- **Casual chat** (greetings, venting, jokes, recommendations): replies naturally and directly with a warm, personable voice.
- **Learning & debugging** (math, code, science, concepts): guides you step by step with one focused Socratic question or hint without spoiling the answer.
- **Optional two-step Teacher -> 0.8B mode**: query a larger OpenAI- or Ollama-compatible teacher model for factual grounding, then let the local 0.8B model put its own conversational or Socratic twist on the reply — or uncheck it to run **100% offline**.

Model weights are **not** stored in this repository. Follow the training & export guide below to train your own `Qwen3.5-0.8B-Socratic-Q8_0.gguf` and store it locally on your PC.

---

## Quick Start (All-in-One Linux AppImage)

The GitHub Release AppImage bundles **both** `socratic-chat` and `llama-server` (compiled with **CPU + Vulkan** GPU acceleration and automatic CPU fallback) in one portable executable.

1. Download `SocraticChat-x86_64.AppImage` from [Releases](https://github.com/mahyar1w/SocraticChat/releases) (or trigger the **Build & Release AppImage** GitHub Action manually).
2. Place your trained GGUF model in the default local data directory on your PC:
   ```bash
   mkdir -p ~/.local/share/socratic-chat
   cp Qwen3.5-0.8B-Socratic-Q8_0.gguf ~/.local/share/socratic-chat/
   ```
   *(You can also set `SOCRATIC_GGUF=/path/to/model.gguf` or pick any path in the app's **Settings** panel.)*
3. Make the AppImage executable and launch it:
   ```bash
   chmod +x SocraticChat-x86_64.AppImage
   ./SocraticChat-x86_64.AppImage
   ```

The AppImage also supports CLI and benchmark subcommands:
```bash
./SocraticChat-x86_64.AppImage ask "Why is binary search O(log n)?"
./SocraticChat-x86_64.AppImage ask --show-reference          # interactive terminal chat
./SocraticChat-x86_64.AppImage bench --help                  # llama-bench suite
```

---

## Where the Model & App State Are Stored on Your PC

| Item | Default Location |
|---|---|
| Local GGUF model (auto-discovered) | `~/.local/share/socratic-chat/Qwen3.5-0.8B-Socratic-Q8_0.gguf` (or `$XDG_DATA_HOME/socratic-chat/...`, `$SOCRATIC_GGUF`, or current directory) |
| Saved chats & settings | `~/.config/socratic-chat/state.json` (or `$XDG_CONFIG_HOME/socratic-chat/state.json`) |
| Teacher API key | Memory / `TEACHER_API_KEY` / `OPENAI_API_KEY` env var only — **never written to disk** |

---

## Training, Distilling & Exporting the 0.8B Model

This repo includes the full Python pipeline to curate the dataset, fine-tune `Qwen3.5-0.8B` with 16-bit LoRA via Unsloth, and export a quantized `Q8_0` GGUF model.

### Prerequisites
- Linux with an NVIDIA GPU (CUDA) and Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

### Step 1: Install training dependencies
```bash
uv sync
```

### Step 2: Prepare the dataset (`prepare_dataset.py`)
Curates `data/train.jsonl` and `data/val.jsonl` from:
- Everyday casual conversation (`HuggingFaceTB/smoltalk` + synthetic chat-twist dialogues with `[CHAT]` reference formatting)
- Multi-step technical & conceptual Socratic dialogues (synthetic + `openai/gsm8k` + `meric533/socrateach-sft` + `rubanikov/socratic-only-sft` with `[TUTOR]` reference formatting)

```bash
uv run python prepare_dataset.py
```

### Step 3: Fine-tune with LoRA (`train.py`)
Fine-tunes `Qwen3.5-0.8B` (uses local `./Qwen3.5-0.8B` if present, otherwise downloads `unsloth/Qwen3.5-0.8B`) with response-only loss masking, saves the adapter to `outputs/lora_adapter/`, and writes a before/after evaluation report to `outputs/eval_comparison.json`:

```bash
uv run python train.py
```

### Step 4: Merge & export to GGUF (`export_gguf.py`)
Merges the LoRA adapter into 16-bit weights (`Qwen3.5-0.8B-Socratic/`), converts to `Qwen3.5-0.8B-Socratic-Q8_0.gguf`, and generates `Modelfile.socratic`:

```bash
uv run python export_gguf.py
```

### Step 5: Store the model on your PC
Move or copy the exported GGUF to SocraticChat's local data directory so both the AppImage and source builds find it automatically:

```bash
mkdir -p ~/.local/share/socratic-chat
mv Qwen3.5-0.8B-Socratic-Q8_0.gguf ~/.local/share/socratic-chat/
```

*(Optional)* If you also want to serve it through Ollama:
```bash
ollama create socratic -f Modelfile.socratic
```

---

## Building From Source

Clone with the `llama.cpp` submodule:
```bash
git clone --recursive git@github.com:mahyar1w/SocraticChat.git
cd SocraticChat
```

Build `llama-server` (CPU + Vulkan, or omit `-DGGML_VULKAN=ON` for CPU-only):
```bash
cmake -S llama.cpp -B llama.cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build llama.cpp/build --target llama-server -j"$(nproc)"
```

Build and run `socratic-chat`:
```bash
cd socratic-chat
cargo build --release
LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server ./target/release/socratic-chat
```

---

## GitHub Actions Workflows

- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): runs on every push/PR to `main` — executes Rust unit tests, `cargo check --release`, and Python syntax + prompt helper checks.
- **AppImage Release** ([`.github/workflows/release-appimage.yml`](.github/workflows/release-appimage.yml)): builds `llama-server` (CPU + Vulkan) from the `llama.cpp` submodule and `socratic-chat` in release mode, packages `SocraticChat-x86_64.AppImage`, uploads it as an artifact, and publishes a GitHub Release whenever you push a version tag:
  ```bash
  git tag v0.1.0
  git push origin v0.1.0
  ```
  You can also trigger a build any time from the **Actions -> Build & Release AppImage -> Run workflow** button.

---

## Project Layout

```text
.
├── .github/workflows/   # CI and all-in-one AppImage release workflows
├── llama.cpp/           # Git submodule for local llama-server (CPU + Vulkan)
├── packaging/           # AppImage desktop entry & vector icon
├── socratic-chat/       # Rust desktop GUI (eframe/egui), CLI (`ask`), and `bench`
├── prompts.py           # Shared prompt templates & <think> stripping for training
├── prepare_dataset.py   # Dataset curation (casual chat + Socratic tutoring)
├── train.py             # Unsloth 16-bit LoRA fine-tuning
├── export_gguf.py       # LoRA merge & Q8_0 GGUF export
└── Modelfile.socratic   # Optional Ollama Modelfile
```
