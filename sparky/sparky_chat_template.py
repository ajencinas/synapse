"""Shared Synapse SFT chat template — single source of truth for the chatbot
and the eval harness.

Must match sft/tokenize_sft_data.py EXACTLY: reserved role tokens (ids 9/10/11),
<|im_end|> separators, no <|im_start|>, no newlines. Prompting the SFT model in
any other format runs it out-of-distribution.
"""

import os
import sys

# Canonical tool-call serializer — shared with tokenize_sft_data.py so the
# tool_call JSON renders byte-identically at train and inference.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sft"))
from tools_runtime import dump_tool_call

ROLE_SYSTEM = "<|reserved_0|>"      # id 9
ROLE_USER = "<|reserved_1|>"        # id 10
ROLE_ASSISTANT = "<|reserved_2|>"   # id 11
TOOL_CALL_MARKER = "<|tool_call|>"  # id 7 — emitted INLINE inside an assistant turn
ROLE_TOOL = "<|tool_result|>"       # id 8 — role string "tool" (injected results)
IM_END = "<|im_end|>"
EOT = "<|endoftext|>"

# Empty on purpose: the SFT prose sources (tulu3, dolly, no_robots, alpaca, ...)
# carry NO system prompt, so an empty system is the trained distribution. The
# old "You are a helpful assistant..." string appeared in zero training rows and
# measurably hurt sft_bench. Tool-use turns use CANONICAL_TOOL_SYSTEM instead.
DEFAULT_SYSTEM_PROMPT = ""


def build_sft_prompt(messages, system=None):
    """messages: [{'role':'user'|'assistant','content':str}, ...] ending on the
    latest user turn. Returns the text to encode (with add_special_tokens=False),
    ending on the assistant role marker to cue a response."""
    parts = []
    if system:
        parts.append(ROLE_SYSTEM + system + IM_END)
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            parts.append(ROLE_USER + content + IM_END)
        elif role == "assistant":
            seg = ROLE_ASSISTANT + content
            if m.get("tool_call"):                      # inline call, before im_end
                seg += TOOL_CALL_MARKER + dump_tool_call(m["tool_call"])
            parts.append(seg + IM_END)
        elif role == "tool":                            # injected tool result
            parts.append(ROLE_TOOL + content + IM_END)
        else:
            raise ValueError(f"unknown message role {role!r} (expected user/assistant/tool)")
    parts.append(ROLE_ASSISTANT)
    return "".join(parts)


def sft_stop_token_ids(tokenizer, eot_id):
    """Token ids that should halt generation for an SFT turn."""
    stops = {eot_id}
    ie = tokenizer.token_to_id(IM_END)
    if ie is not None:
        stops.add(ie)
    return stops
