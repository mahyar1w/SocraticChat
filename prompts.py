"""Shared prompt templates and text helpers for dataset prep, training, and GGUF export."""

import re

SOCRATIC_SYSTEM = (
    "You are a warm, witty conversationalist and Socratic mentor. "
    "In casual chat, greetings, venting, humor, or everyday recommendations, reply naturally "
    "and directly with your own personable twist. When helping the user learn, debug, or solve "
    "a problem, guide them step by step with one focused question or hint without spoiling the answer."
)

CHAT_TWIST_SYSTEM = (
    "You are a warm, witty conversational companion. Take the reference reply and put your own "
    "natural, lively, human spin on it—keep all concrete ideas, recommendations, and punchlines "
    "intact, and never turn casual chat or jokes into a quiz question."
)

TUTOR_TWIST_SYSTEM = (
    "You are a warm Socratic mentor. Use the reference answer for factual accuracy, then put "
    "your own Socratic twist on it: guide the user step by step with one focused question or "
    "small hint without giving away the final answer."
)

THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def format_user_with_reference(user_text: str, reference: str | None = None, mode: str = "AUTO") -> str:
    if reference and reference.strip():
        if mode == "CHAT":
            header = "[Reference — rewrite naturally in your own warm, witty voice; keep key details/punchlines, no quiz]"
        elif mode == "TUTOR":
            header = "[Reference — use for accuracy; guide with one Socratic question or hint without spoiling the answer]"
        else:
            header = "[Reference — use these facts/ideas and put your own natural chat or Socratic twist on them]"
        return f"{header}\n{reference.strip()}\n\n[User]\n{user_text.strip()}"
    return user_text.strip()


def system_for_mode(mode: str = "AUTO") -> str:
    if mode == "CHAT":
        return CHAT_TWIST_SYSTEM
    if mode == "TUTOR":
        return TUTOR_TWIST_SYSTEM
    return SOCRATIC_SYSTEM


def inject_reference(
    messages: list[dict[str, str]], reference: str | None = None, mode: str = "AUTO"
) -> list[dict[str, str]]:
    body = [dict(m) for m in messages if m["role"] != "system"]
    if body and body[-1]["role"] == "user" and reference and reference.strip():
        body[-1]["content"] = format_user_with_reference(body[-1]["content"], reference, mode=mode)
    return [{"role": "system", "content": system_for_mode(mode)}, *body]


def strip_think(text: str) -> str:
    cleaned = THINK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned.strip()
