"""Fine-tune Qwen3.5-0.8B with 16-bit LoRA to rewrite a larger model's answer into a Socratic twist."""

import unsloth  # noqa: F401  # must be imported before trl/transformers
import json
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from prompts import inject_reference, strip_think

LOCAL_BASE = Path("Qwen3.5-0.8B")
MODEL_NAME = str(LOCAL_BASE) if LOCAL_BASE.exists() else "unsloth/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = Path("outputs")
ADAPTER_DIR = OUTPUT_DIR / "lora_adapter"

# Held-out evaluation cases covering BOTH casual chatting and Socratic tutoring.
EVAL_PROMPTS = [
    (
        "chat_greeting",
        [{"role": "user", "content": "hey! how's your day going?"}],
        "Hey! My day's going well, thanks for asking—ready to chat about whatever's on your mind. How is your day going?",
    ),
    (
        "chat_vent",
        [{"role": "user", "content": "ugh i've been fighting a broken CSS layout for three hours and my brain is cooked"}],
        "Three hours fighting CSS is exhausting! Step away for 5 minutes to rest your eyes—and if you want, we can vent about it or inspect the flexbox/grid container when you're back.",
    ),
    (
        "chat_recommend",
        [{"role": "user", "content": "got any good rainy-night movie recommendations? something atmospheric, not loud action"}],
        "Blade Runner 2049, Arrival, or Lost in Translation are atmospheric, moody rainy-night films without frantic Marvel-style action.",
    ),
    (
        "chat_banter",
        [{"role": "user", "content": "tell me a quick nerdy joke"}],
        "There are 10 types of people in the world: those who understand binary, and those who don't.",
    ),
    (
        "math",
        [{"role": "user", "content": "If a train travels 180 km in 2.5 hours, what is its average speed?"}],
        "Average speed equals total distance divided by total time: 180 km / 2.5 hours = 72 km/h.",
    ),
    (
        "python_bug",
        [{"role": "user", "content": "Why does `nums = [1, 2, 3]; print(nums[3])` crash with IndexError?"}],
        "Python uses 0-based indexing, so a 3-element list has indices 0, 1, and 2. Index 3 is out of bounds; use `nums[2]` or `nums[-1]` for the third item.",
    ),
    (
        "postgres_planner",
        [{"role": "user", "content": "Why did PostgreSQL ignore my B-tree index on `is_active` for `SELECT * FROM users WHERE is_active = true` when 85% of users are active?"}],
        "When 85% of rows match, jumping via a B-tree index triggers heavy random heap page I/O across nearly every page anyway, so PostgreSQL's planner picks a faster sequential table scan.",
    ),
    (
        "pushback",
        [{"role": "user", "content": "Don't ask me questions, just give me the formula and final answer for the area of a circle with radius 5."}],
        "The area of a circle is A = pi * r^2. Substituting r = 5 gives A = 25 * pi (approx 78.54).",
    ),
]


def generate_samples(model, tok, label: str, use_reference: bool, use_unsloth_infer: bool = False) -> dict[str, str]:
    if use_unsloth_infer:
        FastLanguageModel.for_inference(model)
    else:
        model.eval()
    results = {}
    for name, user_msgs, ref in EVAL_PROMPTS:
        mode = "CHAT" if name.startswith("chat_") else "TUTOR"
        msgs = inject_reference(user_msgs, ref, mode=mode) if use_reference else user_msgs
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=160,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                eos_token_id=[tok.eos_token_id, 248044],
                pad_token_id=tok.eos_token_id,
            )
        gen = strip_think(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        results[name] = gen
        print(f"\n[{label} | {name}]\n{gen}")
    if not use_unsloth_infer:
        model.train()
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, processor = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
    )
    tok = getattr(processor, "tokenizer", processor)

    base_samples = generate_samples(model, tok, "BASE_WITH_REF", use_reference=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    def format_batch(examples):
        return {
            "text": [
                tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in examples["messages"]
            ]
        }

    train_ds = load_dataset("json", data_files="data/train.jsonl", split="train").map(
        format_batch, batched=True, remove_columns=["source", "messages"]
    )
    val_ds = load_dataset("json", data_files="data/val.jsonl", split="train").map(
        format_batch, batched=True, remove_columns=["source", "messages"]
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=2,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            weight_decay=0.01,
            optim="adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=30,
            save_strategy="steps",
            save_steps=30,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            seed=42,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    stats = trainer.train()
    eval_metrics = trainer.evaluate()
    print("Training summary:", stats.metrics)
    print("Best eval metrics:", eval_metrics)

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")

    twist_samples = generate_samples(model, tok, "SOCRATIC_TWIST", use_reference=True, use_unsloth_infer=True)
    standalone_samples = generate_samples(model, tok, "SOCRATIC_NO_REF", use_reference=False, use_unsloth_infer=True)
    report = {
        "train_metrics": stats.metrics,
        "eval_metrics": eval_metrics,
        "comparisons": {
            name: {
                "prompt": msgs[-1]["content"],
                "teacher_reference": ref,
                "base_with_ref": base_samples[name],
                "socratic_twist": twist_samples[name],
                "socratic_standalone": standalone_samples[name],
            }
            for name, msgs, ref in EVAL_PROMPTS
        },
    }
    (OUTPUT_DIR / "eval_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
