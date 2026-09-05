#!/usr/bin/env python3
"""Shared tool runtime for SynapseGPT tool-use SFT.

ONE module used **identically** at data-generation time (`generate_tool_use.py`)
and at inference time (`sparky_chatbot.py` / `sparky_eval.py`). If the tools or the
serialization differed between the two, we'd train on one distribution and run on
another — a silent, hard-to-debug failure. Sharing this module makes "identical"
structural, not aspirational. See sft/TOOL_USE_PLAN.md §7 and TOOL_USE_DATAGEN.md
§0/§1.

Contract decisions (deliberate):
  - `run_python` / `run_search` return **raw** result text. Truncation to a token
    budget is the CALLER's job via `truncate_tokens(...)` — one truncation point,
    and the tools don't depend on the tokenizer. (The agentic loop does
    `truncate_tokens(run_python(code), PY_RESULT_TOK)`.)
  - Tool errors are returned as the result string (so the model learns to recover),
    never raised.
  - `dump_tool_call` is the single canonical serializer — byte-stable across
    generator / tokenizer / template / chatbot.

Token facts (verified against tokenizer fingerprint 7a570a7ba9fc7985):
  <|tool_call|>=7, <|tool_result|>=8 (dedicated named specials — NOT reserved_3/4).

Security: `run_python` executes model-generated code. It is sandboxed (no network,
CPU/address-space rlimits, wall-clock timeout, temp cwd, minimal env that excludes
the process's secrets) but is NOT a hardened jail. Run only in a trusted context.

Deps: tokenizers (token counting); stdlib urllib for Brave (no `requests`).
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Constants (budgets pinned here; see TOOL_USE_DATAGEN.md §1b/§7)
# ---------------------------------------------------------------------------
PY_RESULT_TOK = 200          # caller truncates python results to this
SEARCH_RESULT_TOK = 128      # caller truncates search results to this (small on purpose)
PYTHON_TIMEOUT = 5           # wall-clock seconds for run_python
PY_MEM_BYTES = 4 * 1024 ** 3  # address-space cap (generous so sympy imports cleanly)

SEARCH_UNAVAILABLE = "[search unavailable]"   # transient-failure sentinel (§4 rejects traces with it)
NO_RESULTS = "[no results]"                   # search succeeded but empty (NOT a failure)

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

EXPECTED_TOOL_IDS = {"<|tool_call|>": 7, "<|tool_result|>": 8}

# The ONE system prompt stored in every tool_use / tool_negative record and used
# at inference when tools are on (train == inference). Lives here — not in the
# generator — so the chatbot / eval never import a data-generation script.
# NOTE: it promises `search`, but v2 shipped zero search traces (python only);
# the tool loop still executes a search call if a BRAVE_API_KEY is present.
CANONICAL_TOOL_SYSTEM = (
    "You are an expert problem solver with tools. Use `python` (sympy available) for "
    "any non-trivial calculation, and `search` to recall the method for an unfamiliar "
    "problem. Use tools when they help, then end with a line exactly:\n"
    "The answer is: <answer>"
)


# ---------------------------------------------------------------------------
# Canonical tool-call serializer (byte-stable everywhere) — DATAGEN §1d
# ---------------------------------------------------------------------------
def dump_tool_call(tc):
    """Serialize a tool-call dict identically wherever it is turned into text.

    sort_keys -> stable key order; compact separators -> no spaces; ensure_ascii
    False -> keep unicode literal (digit-per-token already taxes the budget)."""
    return json.dumps(tc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Synapse tokenizer: token counting + truncation (digit-per-token => must use the
# real tokenizer, not char/word estimates). Lazy-loaded; fail loud if missing.
# ---------------------------------------------------------------------------
_TOK = None
_TOK_LOCK = threading.Lock()


def default_synapse_dir():
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/synapse"
    return os.path.abspath("./synapse")


def tokenizer_path():
    """SYNAPSE_TOKENIZER (explicit file) overrides; else SYNAPSE_DIR/tokenizer_out."""
    override = os.environ.get("SYNAPSE_TOKENIZER")
    if override:
        return override
    syn = os.environ.get("SYNAPSE_DIR") or default_synapse_dir()
    return os.path.join(syn, "tokenizer_out", "tokenizer.json")


def _tok():
    global _TOK
    if _TOK is None:
        with _TOK_LOCK:
            if _TOK is None:
                from tokenizers import Tokenizer
                path = tokenizer_path()
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Synapse tokenizer not found at {path!r}. Set SYNAPSE_TOKENIZER "
                        f"or SYNAPSE_DIR. Char/word fallbacks are forbidden (digit-per-token).")
                _TOK = Tokenizer.from_file(path)
    return _TOK


def synapse_token_len(text):
    """Exact token count with the Synapse tokenizer (specials NOT added)."""
    if not text:
        return 0
    return len(_tok().encode(text, add_special_tokens=False).ids)


def truncate_tokens(text, max_tok):
    """Keep the first max_tok tokens; append a marker if truncated."""
    if not text:
        return text
    enc = _tok().encode(text, add_special_tokens=False)
    if len(enc.ids) <= max_tok:
        return text
    return _tok().decode(enc.ids[:max_tok]) + " …[truncated]"


def verify_tool_tokens(tok=None):
    """Phase-0 regression guard: assert the dedicated tool tokens are 7/8 and atomic.
    Fail loud (the project rule) so a swapped tokenizer can't silently break the
    wire format."""
    tok = tok or _tok()
    for name, expected in EXPECTED_TOOL_IDS.items():
        tid = tok.token_to_id(name)
        if tid != expected:
            raise SystemExit(
                f"tool token {name!r} has id {tid}, expected {expected} — wrong/"
                f"renumbered tokenizer; refusing to proceed")
        n = len(tok.encode(name, add_special_tokens=False).ids)
        if n != 1:
            raise SystemExit(f"tool token {name!r} is not atomic (encodes to {n} tokens)")
    return True


# ---------------------------------------------------------------------------
# python tool: sandboxed subprocess
# ---------------------------------------------------------------------------
# Runner executed via `python -c`: blocks network, then execs the user's file.
# Subclass the real socket (keep it a class so urllib/http.client internals that
# reference the type still work) and block outbound connects; also replace the
# create_connection helper. Both raise the same OSError.
_RUNNER = (
    "import sys, socket\n"
    "_Real = socket.socket\n"
    "class _Blocked(_Real):\n"
    "    def connect(self, *a, **k):\n"
    "        raise OSError('network disabled in sandbox')\n"
    "    def connect_ex(self, *a, **k):\n"
    "        raise OSError('network disabled in sandbox')\n"
    "socket.socket = _Blocked\n"
    "def _no_conn(*a, **k):\n"
    "    raise OSError('network disabled in sandbox')\n"
    "socket.create_connection = _no_conn\n"
    "import ast\n"
    "p = sys.argv[1]\n"
    "with open(p) as f:\n"
    "    src = f.read()\n"
    "g = {'__name__': '__main__'}\n"
    "tree = ast.parse(src, '<tool>', 'exec')\n"   # SyntaxError -> surfaced as [error]
    "body = tree.body\n"
    # REPL semantics: if the snippet ends on a bare expression, print its value
    # (like Jupyter) so teachers/models don't need an explicit print() on the
    # final line. print(...)-terminated or assignment-terminated code is unchanged.
    "if body and isinstance(body[-1], ast.Expr):\n"
    "    last = body.pop()\n"
    "    exec(compile(ast.Module(body, []), '<tool>', 'exec'), g)\n"
    "    _v = eval(compile(ast.Expression(last.value), '<tool>', 'eval'), g)\n"
    "    if _v is not None:\n"
    "        print(repr(_v))\n"
    "else:\n"
    "    exec(compile(tree, '<tool>', 'exec'), g)\n"
)


def _set_limits():
    """preexec: cap CPU + address space so runaway code can't hog the box."""
    import resource
    cpu = PYTHON_TIMEOUT + 1
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (PY_MEM_BYTES, PY_MEM_BYTES))
    except (ValueError, OSError):
        pass


def run_python(code, *, timeout=PYTHON_TIMEOUT):
    """Execute `code` in a sandboxed subprocess; return raw stdout (or an error/
    timeout marker). Caller truncates with truncate_tokens(..., PY_RESULT_TOK).

    sympy is importable. Network is blocked; secrets are NOT in the child env."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        codefile = os.path.join(td, "snippet.py")
        with open(codefile, "w") as f:
            f.write(code or "")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", _RUNNER, codefile],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=td,
                env={"PATH": os.environ.get("PATH", "")},   # no secrets leak to the snippet
                preexec_fn=_set_limits, start_new_session=True)
        except Exception as e:  # pragma: no cover - spawn failure is environmental
            return f"[python failed to start: {e}]"
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.communicate(timeout=2)
            except Exception:
                pass
            return f"[python timed out after {timeout}s]"

        out = (out or "").strip()
        err = (err or "").strip()
        if proc.returncode != 0:
            # surface the last error line so the model can self-correct
            tail = err.splitlines()[-1] if err else f"exited with code {proc.returncode}"
            return (out + "\n" if out else "") + f"[error] {tail}"
        if not out and err:
            return err           # warnings printed to stderr but success
        return out if out else "[no output]"


# ---------------------------------------------------------------------------
# Rate limiter (process-wide; lock-guarded next_allowed_time, NOT a semaphore)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Bounds the AGGREGATE request rate across all threads. Each acquire() reserves
    the next time slot under the lock, then sleeps (outside the lock) until that slot
    arrives — so 48 workers issue at most `rate` req/s combined."""

    def __init__(self, rate):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.min_interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.min_interval
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)


_DEFAULT_LIMITER = RateLimiter(1.0)   # Brave free tier ~1 req/s


# ---------------------------------------------------------------------------
# search tool: Brave Search API via stdlib urllib (monkeypatchable for tests)
# ---------------------------------------------------------------------------
def _http_get(url, headers, params, timeout):
    """Return (status_code, body_text). Never raises for HTTP error status — maps
    HTTPError to (code, body). Raises urllib.error.URLError only for transport
    failures (DNS/timeout/connection)."""
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        return e.code, body


def _backoff(attempt):
    time.sleep(min(2 ** attempt, 8))


def run_search(query, *, limiter=None, api_key=None, count=3, retries=4, timeout=10):
    """Query Brave; return raw joined "title: snippet" results (caller truncates with
    SEARCH_RESULT_TOK). Returns SEARCH_UNAVAILABLE on exhausted retries / bad key
    (so the trace is rejected, §4), or NO_RESULTS when the query simply finds nothing.

    Rate-limited (§1c): every attempt, including retries, passes through `limiter`.
    429/5xx and transport errors are transient -> backoff+retry; other 4xx (bad
    key/params) -> give up immediately without burning retries."""
    api_key = api_key or os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return SEARCH_UNAVAILABLE
    limiter = limiter or _DEFAULT_LIMITER
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    params = {"q": query, "count": count}

    for attempt in range(retries):
        limiter.acquire()
        try:
            status, body = _http_get(BRAVE_URL, headers, params, timeout)
        except urllib.error.URLError:
            _backoff(attempt)
            continue
        if status == 429 or status >= 500:
            _backoff(attempt)
            continue
        if status != 200:
            return SEARCH_UNAVAILABLE      # 4xx (auth/bad query): retrying won't help
        try:
            data = json.loads(body)
        except ValueError:
            _backoff(attempt)
            continue
        # scan the body for a Brave error envelope — these can arrive even with an
        # HTTP 200, so don't trust the status code alone. Treat as transient.
        if (not isinstance(data, dict) or data.get("type") == "ErrorResponse"
                or data.get("error") is not None):
            _backoff(attempt)
            continue
        results = ((data.get("web") or {}).get("results")) or []
        snippets = []
        for r in results[:count]:
            title = (r.get("title") or "").strip()
            desc = (r.get("description") or "").strip()
            if title and desc:
                snippets.append(f"{title}: {desc}")
            elif title or desc:
                snippets.append(title or desc)
        text = " | ".join(snippets)
        return text if text else NO_RESULTS
    return SEARCH_UNAVAILABLE


# ---------------------------------------------------------------------------
# CLI self-check (Phase-2 "unit-exercise both tools standalone")
# ---------------------------------------------------------------------------
def _selfcheck():
    print("[tokens]", "verifying <|tool_call|>=7, <|tool_result|>=8 ...", end=" ")
    try:
        verify_tool_tokens()
        print("OK")
    except (FileNotFoundError, SystemExit) as e:
        print(f"SKIP/FAIL: {e}")

    print("[serializer]", dump_tool_call({"tool": "python", "code": "print(1)"}))

    print("[python] 2847*391 ->", run_python("print(2847*391)"))
    print("[python] sympy   ->",
          run_python("import sympy; print(sympy.simplify('(x**2-1)/(x-1)'))"))
    print("[python] error   ->", run_python("1/0"))
    print("[python] timeout ->", run_python("while True: pass", timeout=2))
    print("[python] no-net  ->",
          run_python("import urllib.request as u; u.urlopen('http://example.com')"))

    if os.environ.get("BRAVE_API_KEY"):
        print("[search] live ->", run_search("related rates calculus method")[:200])
    else:
        print("[search] skipped (no BRAVE_API_KEY)")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    except ImportError:
        pass
    _selfcheck()
