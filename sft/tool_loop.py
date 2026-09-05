#!/usr/bin/env python3
"""Shared inference-time tool loop for SynapseGPT — SFT v2 Phase 4.2.

Written ONCE and imported by both the chatbot (sparky/sparky_chatbot.py) and the
tool eval (sft/eval_tool_use.py), so "what the model is served" and "what the
model is measured on" are structurally the same loop.

Wire format (must match sft/tokenize_sft_data.py + sparky_chat_template.py):

    <|reserved_2|> {content} <|tool_call|> {"code":"...","tool":"python"} <|im_end|>
    <|tool_result|> {result} <|im_end|>
    <|reserved_2|> {final answer} <|im_end|>

The loop works on TOKEN IDS, not decoded text: the tokenizer's decode() drops
special tokens by default, so the only reliable way to see a call is to find
id 7 (<|tool_call|>) in the generated ids. Everything before it is the turn's
prose; everything after it (up to the stop token) is the JSON of the call.

Contract:
  - `generate_ids(prompt_ids) -> iterable[int]` is supplied by the caller. It must
    yield generated token ids one at a time and STOP (not yield) at <|im_end|> /
    <|endoftext|>. The loop does not care how sampling is done.
  - Tool errors are fed back as the tool result (the training data contains
    recover-from-error traces); malformed JSON likewise. A call that repeats an
    earlier call byte-for-byte ends the loop — re-running it cannot change the
    result and the model is stuck.
  - <|tool_call|> is NOT a stop token. sft_stop_token_ids() is unchanged.

Events yielded (dicts), in order, so a UI can stream:
  {"type": "token", "text": str}                        prose as it is generated
  {"type": "tool_call", "tool": str, "call": dict}       a parsed call (or "tool": None if malformed)
  {"type": "tool_result", "text": str}                   what was fed back
  {"type": "final", "messages": [...], "status": str, "rounds": int}
      status: "answered" | "max_rounds" | "repeat" | "budget"
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "sparky"))

import tools_runtime as tr
from sparky_chat_template import build_sft_prompt

MAX_ROUNDS = 5                 # tool calls per user turn (plan §Phase 4.2)
TOOL_CALL_ID = 7               # <|tool_call|>  (tools_runtime.EXPECTED_TOOL_IDS)


def _check_tokenizer(tok):
    tc = tok.token_to_id("<|tool_call|>")
    if tc != TOOL_CALL_ID:
        raise SystemExit(f"tokenizer <|tool_call|> id is {tc}, expected {TOOL_CALL_ID} — "
                         f"wrong tokenizer for this model")


def parse_call(call_text):
    """Parse the JSON after <|tool_call|>. Returns (call_dict, error_or_None)."""
    try:
        call = json.loads(call_text)
    except ValueError as e:
        return None, f"malformed tool call JSON: {e}"
    if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
        return None, "malformed tool call: expected an object with a string 'tool' field"
    return call, None


def execute(call, *, search_api_key=None):
    """Run one parsed call through tools_runtime and return the (truncated)
    result text. Never raises — errors are the result, so the model can recover."""
    tool = call.get("tool")
    if tool == "python":
        raw = tr.run_python(call.get("code", ""))
        return tr.truncate_tokens(raw, tr.PY_RESULT_TOK)
    if tool == "search":
        key = search_api_key or os.environ.get("BRAVE_API_KEY")
        if not key:
            return "[error] search is not available in this session"
        raw = tr.run_search(call.get("query", ""), api_key=key)
        return tr.truncate_tokens(raw, tr.SEARCH_RESULT_TOK)
    return f"[error] unknown tool {tool!r}; available: python"


def run_tool_loop(generate_ids, tokenizer, messages, system, *, max_rounds=MAX_ROUNDS,
                  max_prompt_tokens=None, search_api_key=None):
    """Drive one assistant reply, executing tool calls until the model answers.

    messages: Synapse-schema history ending on a user turn (assistant turns may
              carry 'tool_call'; 'tool' turns carry results) — the same schema the
              training records use, so the prompt is byte-identical to training.
    system:   CANONICAL_TOOL_SYSTEM when tools are on, "" / None when off.

    Generator of events (see module docstring). The final event's "messages" is
    the list of NEW turns to append to the history (assistant/tool/.../assistant).
    """
    _check_tokenizer(tokenizer)
    history = list(messages)
    new_turns = []
    seen_calls = set()

    for rnd in range(max_rounds + 1):
        prompt = build_sft_prompt(history, system=system or None)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if max_prompt_tokens and len(prompt_ids) > max_prompt_tokens:
            # Refuse to silently truncate the LEFT of a tool trace (it would cut
            # the system prompt / question the model is reasoning about).
            yield {"type": "final", "messages": new_turns, "status": "budget", "rounds": rnd}
            return

        prose_ids, call_ids, in_call = [], [], False
        printed = 0
        for tid in generate_ids(prompt_ids):
            if in_call:
                call_ids.append(tid)
                continue
            if tid == TOOL_CALL_ID:
                in_call = True
                continue
            prose_ids.append(tid)
            text = tokenizer.decode(prose_ids)
            if text.endswith("�"):        # incomplete multibyte char — wait
                continue
            if len(text) > printed:
                yield {"type": "token", "text": text[printed:]}
                printed = len(text)
        content = tokenizer.decode(prose_ids)
        if len(content) > printed and not content.endswith("�"):
            yield {"type": "token", "text": content[printed:]}

        if not in_call:                          # plain answer — done
            new_turns.append({"role": "assistant", "content": content})
            history.append(new_turns[-1])
            yield {"type": "final", "messages": new_turns, "status": "answered", "rounds": rnd}
            return

        call_text = tokenizer.decode(call_ids)
        call, err = parse_call(call_text)
        if rnd >= max_rounds:
            # Budget exhausted with the model still wanting a tool: stop here.
            new_turns.append({"role": "assistant", "content": content,
                              **({"tool_call": call} if call else {})})
            yield {"type": "tool_call", "tool": (call or {}).get("tool"), "call": call or {"raw": call_text}}
            yield {"type": "final", "messages": new_turns, "status": "max_rounds", "rounds": rnd}
            return

        if call is None:
            yield {"type": "tool_call", "tool": None, "call": {"raw": call_text}}
            result = f"[error] {err}"
            # A malformed call can't be rendered back through dump_tool_call, so the
            # assistant turn keeps only its prose; the error goes in as the result.
            new_turns.append({"role": "assistant", "content": content})
        else:
            key = tr.dump_tool_call(call)
            yield {"type": "tool_call", "tool": call.get("tool"), "call": call}
            new_turns.append({"role": "assistant", "content": content, "tool_call": call})
            if key in seen_calls:
                history.append(new_turns[-1])
                yield {"type": "final", "messages": new_turns, "status": "repeat", "rounds": rnd + 1}
                return
            seen_calls.add(key)
            result = execute(call, search_api_key=search_api_key)

        history.append(new_turns[-1])
        yield {"type": "tool_result", "text": result}
        new_turns.append({"role": "tool", "content": result})
        history.append(new_turns[-1])

    raise AssertionError("unreachable: loop must return from inside")


def final_text(events):
    """Convenience for non-streaming callers: consume events, return
    (final_assistant_text, final_event)."""
    last = None
    for ev in events:
        if ev["type"] == "final":
            last = ev
    if last is None:
        raise RuntimeError("tool loop ended without a final event")
    turns = [m for m in last["messages"] if m["role"] == "assistant"]
    return (turns[-1]["content"] if turns else ""), last
