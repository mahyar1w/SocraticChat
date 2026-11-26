"""Merge trained Socratic LoRA adapter and export Ollama-compatible Q8_0 GGUF."""

import unsloth  # noqa: F401
import sys
from pathlib import Path
from unsloth import FastLanguageModel

from prompts import SOCRATIC_SYSTEM

ADAPTER_DIR = Path("outputs/lora_adapter")
MERGED_DIR = Path("Qwen3.5-0.8B-Socratic")
LLAMA_CPP_DIR = Path.home() / ".unsloth/llama.cpp"


def main():
    assert ADAPTER_DIR.exists(), f"Missing trained adapter at {ADAPTER_DIR}"

    model, processor = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(MERGED_DIR), processor, save_method="merged_16bit")

    # Export with unfused attn_qkv/attn_gate and ssm_beta/ssm_alpha for Ollama compatibility.
    sys.path.insert(0, str(LLAMA_CPP_DIR))
    sys.path.insert(1, str(LLAMA_CPP_DIR / "gguf-py"))
    import conversion.qwen as cq
    import convert_hf_to_gguf as conv

    cq.Qwen3_5TextModel._SHARED_INPUT_MERGES = ()
    target_gguf = Path("Qwen3.5-0.8B-Socratic-Q8_0.gguf").resolve()
    sys.argv = ["convert_hf_to_gguf.py", str(MERGED_DIR), "--outtype", "q8_0", "--outfile", str(target_gguf)]
    conv.main()
    assert target_gguf.exists(), f"Failed to write {target_gguf}"
    print(f"Exported GGUF: {target_gguf} ({target_gguf.stat().st_size / 1e9:.2f} GB)")

    modelfile = Path("Modelfile.socratic")
    modelfile.write_text(
        f"FROM ./{target_gguf.name}\n"
        "PARAMETER temperature 0.6\n"
        "PARAMETER top_p 0.9\n"
        'PARAMETER stop "<|im_end|>"\n'
        f'SYSTEM "{SOCRATIC_SYSTEM}"\n',
        encoding="utf-8",
    )
    print(f"Wrote {modelfile}")


if __name__ == "__main__":
    main()
