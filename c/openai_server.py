#!/usr/bin/env python3
"""Dependency-free OpenAI-compatible HTTP gateway for the colibri engine."""

import argparse
import codecs
import collections
import contextlib
import json
import math
import os
import select
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


HERE = Path(__file__).resolve().parent
END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"
MAX_BODY = 4 << 20
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://tauri.localhost",
    "tauri://localhost",
)


class APIError(Exception):
    def __init__(self, status, message, param=None, code=None, error_type="invalid_request_error",
                 headers=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type
        self.headers = headers or {}


class ClientCancelled(Exception):
    pass


def error_object(error):
    return {"error": {"message": error.message, "type": error.error_type,
                      "param": error.param, "code": error.code}}


class GenerationScheduler:
    """Bounded FIFO admission for the engine's independent KV contexts."""

    def __init__(self, max_queue=8, queue_timeout=300, capacity=1):
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.capacity = capacity
        self.free_slots = set(range(capacity))
        self.condition = threading.Condition()
        self.queue = collections.deque()
        self.active = 0
        self.closed = False
        self.admitted = 0
        self.completed = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0

    @contextlib.contextmanager
    def admit(self, cancelled=None, slot=None):
        ticket = object()
        queued_at = time.monotonic()
        with self.condition:
            if self.closed:
                raise APIError(503, "The inference scheduler is shutting down.", None,
                               "scheduler_closed", "server_error")
            if (self.active >= self.capacity or self.queue) and len(self.queue) >= self.max_queue:
                self.rejected += 1
                raise APIError(429, "The inference queue is full.", None, "queue_full",
                               "rate_limit_error", {"Retry-After": "1"})
            self.queue.append(ticket)
            deadline = queued_at + self.queue_timeout
            while True:
                if self.closed:
                    self.queue.remove(ticket)
                    self.condition.notify_all()
                    raise APIError(503, "The inference scheduler is shutting down.", None,
                                   "scheduler_closed", "server_error")
                available = min(self.free_slots) if slot is None and self.free_slots else slot
                if self.queue[0] is ticket and available in self.free_slots:
                    break
                if cancelled and cancelled():
                    self.queue.remove(ticket)
                    self.cancelled += 1
                    self.condition.notify_all()
                    raise ClientCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.queue.remove(ticket)
                    self.timed_out += 1
                    self.condition.notify_all()
                    raise APIError(429, "Timed out waiting for the inference engine.", None,
                                   "queue_timeout", "rate_limit_error", {"Retry-After": "1"})
                self.condition.wait(min(remaining, 0.25))
            self.queue.popleft()
            self.free_slots.remove(available)
            self.active += 1
            self.admitted += 1
            wait_seconds = time.monotonic() - queued_at
        try:
            yield wait_seconds, available
        finally:
            with self.condition:
                self.active -= 1
                self.free_slots.add(available)
                self.completed += 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "queued": len(self.queue),
                    "capacity": self.capacity,
                    "max_queue": self.max_queue, "queue_timeout_seconds": self.queue_timeout,
                    "admitted": self.admitted, "completed": self.completed,
                    "rejected": self.rejected, "timed_out": self.timed_out,
                    "cancelled": self.cancelled}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def content_text(content, param):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of text parts.", param)
    parts = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            raise APIError(400, "Colibri currently supports text message content only.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(part.get("text"), str):
            raise APIError(400, "Text content parts require a string `text` field.",
                           f"{param}.{index}.text")
        parts.append(part["text"])
    return "".join(parts)


# ---- GLM-5.2 tool calling -----------------------------------------------------------------
# The model expresses tool calls as ordinary text (from chat_template.jinja):
#   <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
# and tool results come back as <|observation|><tool_response>{content}</tool_response>.
# We render those markers into the prompt and parse them back into OpenAI `tool_calls`.

BOX_START, BOX_END = "<tool_call>", "</tool_call>"
TR_OPEN,  TR_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

_BOX_RE  = re.compile(re.escape(BOX_START) + r"(.*?)" + re.escape(BOX_END), re.DOTALL)
_ARG_RE  = re.compile(r"<arg_key>([^<]*)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL)
_NAME_RE = re.compile(r"\s*([A-Za-z0-9_.\-]+)")
_TAG_RE  = re.compile(r"</?arg_key>|</?arg_value>")

# ---- Hy3 tool calling (tencent/Hy3, `:opensource` tokens) ----------------------------------
# <tool_calls:opensource>
#   <tool_call:opensource>{name}<tool_sep:opensource><arg_key:opensource>{k}</arg_key:opensource>
#   <arg_value:opensource>{v}</arg_value:opensource>...</tool_call:opensource>
# </tool_calls:opensource> ... <tool_responses:opensource><tool_response:opensource>{content}
# </tool_response:opensource></tool_responses:opensource>

HY3_TOOLCALLS_OPEN,  HY3_TOOLCALLS_CLOSE  = "<tool_calls:opensource>",  "</tool_calls:opensource>"
HY3_TOOLCALL_OPEN,   HY3_TOOLCALL_CLOSE   = "<tool_call:opensource>",   "</tool_call:opensource>"
HY3_TOOLSEP = "<tool_sep:opensource>"
HY3_ARGKEY_OPEN,   HY3_ARGKEY_CLOSE   = "<arg_key:opensource>",   "</arg_key:opensource>"
HY3_ARGVALUE_OPEN, HY3_ARGVALUE_CLOSE = "<arg_value:opensource>", "</arg_value:opensource>"
HY3_TOOLRESPS_OPEN, HY3_TOOLRESPS_CLOSE = "<tool_responses:opensource>", "</tool_responses:opensource>"
HY3_TOOLRESP_OPEN,  HY3_TOOLRESP_CLOSE  = "<tool_response:opensource>",  "</tool_response:opensource>"
HY3_THINK_OPEN, HY3_THINK_CLOSE = "<think:opensource>", "</think:opensource>"
HY3_REASONING_MODE = "<｜reasoning_mode:opensource｜>"
HY3_BOS = "<｜hy_begin_of_sentence:opensource｜>"
HY3_USER = "<｜hy_User:opensource｜>"
HY3_ASSISTANT = "<｜hy_Assistant:opensource｜>"
HY3_EOS = "<｜hy_eos:opensource｜>"

_HY3_CALL_RE = re.compile(re.escape(HY3_TOOLCALL_OPEN) + r"(.*?)" + re.escape(HY3_TOOLCALL_CLOSE),
                         re.DOTALL)
_HY3_ARG_RE  = re.compile(re.escape(HY3_ARGKEY_OPEN) + r"([^<]*)" + re.escape(HY3_ARGKEY_CLOSE)
                         + r"\s*" + re.escape(HY3_ARGVALUE_OPEN) + r"(.*?)" + re.escape(HY3_ARGVALUE_CLOSE),
                         re.DOTALL)
_HY3_TAG_RE  = re.compile("|".join(re.escape(t) for t in
                         (HY3_ARGKEY_OPEN, HY3_ARGKEY_CLOSE, HY3_ARGVALUE_OPEN, HY3_ARGVALUE_CLOSE)))

# De-mangler: opt-in recovery for heavily-quantized models that drop the
# <arg_key>K</arg_key><arg_value> structure. Default OFF (never rewrites well-formed output).
_SALVAGE = os.environ.get("COLI_TOOL_SALVAGE", "0") == "1"


def _tool_param_order(tools):
    """name -> ordered param names (required first) from the request schema, for de-mangling."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        params = ((fn.get("parameters") or {}).get("properties") or {})
        required = list((fn.get("parameters") or {}).get("required") or [])
        out[name] = required + [p for p in params if p not in required]
    return out


def _tool_param_types(tools):
    """name -> {param: declared JSON-schema type}. The model emits every argument as text;
    without the schema a string-typed value that happens to look numeric ("12345" for an
    order id, an SKU, a phone number) would be json.loads()'d into an int and the tool would
    receive the wrong type."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        types = {}
        for key, spec in props.items():
            if isinstance(spec, dict):
                t = spec.get("type")
                if isinstance(t, list):          # {"type": ["string", "null"]}
                    t = next((x for x in t if x != "null"), None)
                types[key] = t
        out[name] = types
    return out


def _coerce_arg(value, declared):
    """Decode a raw <arg_value> according to the declared schema type.

    A string-typed parameter is kept verbatim -- never parsed as JSON. Everything else keeps
    the previous permissive behaviour (parse if it parses, otherwise leave as text)."""
    if declared == "string":
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    if declared in ("integer", "number") and isinstance(parsed, bool):
        return value                              # `true` is not a number
    if declared and declared not in ("integer", "number", "boolean", "object", "array"):
        return value
    return parsed


def parse_tool_calls(reply, tools=None):
    """Return (content, tool_calls). Strict GLM parse; optional de-mangler (COLI_TOOL_SALVAGE=1)
    rescues malformed int4 output by mapping a lone payload onto the tool's primary parameter."""
    param_order = _tool_param_order(tools)
    param_types = _tool_param_types(tools)
    calls, salvaged = [], []
    for match in _BOX_RE.finditer(reply):
        inner = match.group(1)
        name_match = _NAME_RE.match(inner)
        name = name_match.group(1) if name_match else inner.strip()
        args = {}
        types = param_types.get(name, {})
        for arg in _ARG_RE.finditer(inner):
            key, value = arg.group(1), arg.group(2)
            args[key] = _coerce_arg(value, types.get(key))
        if not args and _SALVAGE:
            rest = inner[name_match.end():] if name_match else ""
            payload = _TAG_RE.sub("", rest).strip()
            if payload.startswith("(") and payload.endswith(")"):
                payload = payload[1:-1].strip()
            if payload:
                key = (param_order.get(name) or ["input"])[0]
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                args = {key: payload}
                salvaged.append(name)
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    text = _BOX_RE.sub("", reply)
    if THINK_CLOSE in text:
        text = text.split(THINK_CLOSE, 1)[1]
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if calls:
        dm = len(salvaged)
        sys.stderr.write("[api] tool-calls: %d total, %d strict, %d de-mangled [%s]%s\n"
                         % (len(calls), len(calls) - dm, dm, "CLEAN" if dm == 0 else "DE-MANGLED",
                            (" -> " + ", ".join(salvaged)) if dm else ""))
        sys.stderr.flush()
    return text.strip(), calls


def parse_tool_calls_hy3(reply, tools=None):
    """Return (content, tool_calls) for the Hy3 `:opensource` tool-call format. Name is terminated
    by <tool_sep:opensource> rather than an arg tag; otherwise shares the schema-aware arg coercion
    and optional COLI_TOOL_SALVAGE de-mangler with the GLM parser above."""
    param_order = _tool_param_order(tools)
    param_types = _tool_param_types(tools)
    calls, salvaged = [], []
    for match in _HY3_CALL_RE.finditer(reply):
        inner = match.group(1)
        name, _, rest = inner.partition(HY3_TOOLSEP)
        name = name.strip()
        if not name:
            continue                              # skip malformed block with no tool name
        args = {}
        types = param_types.get(name, {})
        for arg in _HY3_ARG_RE.finditer(rest):
            key, value = arg.group(1).strip(), arg.group(2)
            args[key] = _coerce_arg(value, types.get(key))
        if not args and _SALVAGE:
            payload = _HY3_TAG_RE.sub("", rest).strip()
            if payload.startswith("(") and payload.endswith(")"):
                payload = payload[1:-1].strip()
            if payload:
                key = (param_order.get(name) or ["input"])[0]
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                args = {key: payload}
                salvaged.append(name)
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    text = _HY3_CALL_RE.sub("", reply)
    text = text.replace(HY3_TOOLCALLS_OPEN, "").replace(HY3_TOOLCALLS_CLOSE, "")
    if HY3_REASONING_MODE in text:
        text = text.split(HY3_REASONING_MODE, 1)[0]     # trailing `reasoning_effort:X` marker
    if HY3_THINK_CLOSE in text:
        text = text.split(HY3_THINK_CLOSE, 1)[1]
    text = text.replace(HY3_THINK_OPEN, "").replace(HY3_THINK_CLOSE, "")
    if calls:
        dm = len(salvaged)
        sys.stderr.write("[api] tool-calls: %d total, %d strict, %d de-mangled [%s]%s\n"
                         % (len(calls), len(calls) - dm, dm, "CLEAN" if dm == 0 else "DE-MANGLED",
                            (" -> " + ", ".join(salvaged)) if dm else ""))
        sys.stderr.flush()
    return text.strip(), calls


def split_reasoning_hy3(text, enable_thinking):
    """Split raw Hy3 model output into (reasoning, rest, truncated).

    When `enable_thinking` is True the prompt ends with an unclosed `<think:opensource>`,
    so the generation starts with reasoning text.  We look for the first
    `</think:opensource>` marker; everything before it is reasoning, everything after
    is the visible answer (possibly containing tool-call markers).

    Returns:
        reasoning  -- the reasoning text (empty if disabled or truncated)
        rest       -- the visible answer text (the whole input if disabled)
        truncated  -- True when thinking was enabled but the close marker was never found
                      (3c: whole text becomes content, flagged but not stored as reasoning)
    """
    if not enable_thinking:
        return "", text, False
    idx = text.find(HY3_THINK_CLOSE)
    if idx < 0:
        # 3c: marker never found — whole text is content, reasoning is empty, flagged
        return "", text, True
    reasoning = text[:idx]
    rest = text[idx + len(HY3_THINK_CLOSE):]
    return reasoning, rest, False


def _resolve_tool_choice(tools, tool_choice):
    """Apply `tool_choice` forcing/filtering shared by every chat-template renderer: a named
    function narrows the offered tools to that one function; "none" withholds tools entirely.
    Returns (tools, forced_name)."""
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None                              # the client forbade tools: do not offer them
    return tools, forced


def render_chat(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                tool_choice=None):
    """Render the text-only subset of the official GLM-5.2 chat template."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        effort = "High" if reasoning_effort == "high" else "Max"
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    tools, forced = _resolve_tool_choice(tools, tool_choice)
    if tools:
        # AUTHORITATIVE GLM-5.2 tool-declaration block (byte-matches chat_template.jinja): the
        # `# Tools` + <tools></tools> XML structure is what the model was trained on. A made-up
        # preamble makes it hallucinate other frameworks' syntax (e.g. `end_action`).
        prompt.append("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the "
                      "user query.\n\nYou are provided with function signatures within <tools></tools> "
                      "XML tags:\n<tools>\n")
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean = {k: v for k, v in fn.items() if k not in ("defer_loading", "strict")}
            prompt.append(json.dumps(clean, ensure_ascii=False) + "\n")
        prompt.append("</tools>\n\nFor each function call, output the function name and arguments "
                      "within the following XML format:\n<tool_call>{function-name}"
                      "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                      "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
        if forced:
            prompt.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
        elif tool_choice == "required":
            prompt.append("\n\nYou must call one of the functions above. Do not answer directly.")
    prev_tool = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role in ("system", "developer"):
            prompt.append(f"<|system|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "user":
            prompt.append(f"<|user|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "assistant":
            # content may be null when the message is purely tool_calls
            raw = message.get("content")
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            prompt.append(f"<|assistant|><think></think>{text.strip()}")
            for tc in (message.get("tool_calls") or []):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                prompt.append(BOX_START + (fn.get("name") or ""))
                for key, value in (args or {}).items():
                    prompt.append(f"<arg_key>{key}</arg_key><arg_value>"
                                  + (value if isinstance(value, str)
                                     else json.dumps(value, ensure_ascii=False)) + "</arg_value>")
                prompt.append(BOX_END)
        elif role == "tool":
            if not prev_tool:                       # one <|observation|> per consecutive tool run
                prompt.append("<|observation|>")
            prompt.append(TR_OPEN + content_text(message.get("content"), f"messages.{index}.content") + TR_CLOSE)
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
        prev_tool = (role == "tool")
    prompt.append("<|assistant|><think>" if enable_thinking else
                  "<|assistant|><think></think>")
    return "".join(prompt)


def hy3_effort(enable_thinking, reasoning_effort):
    if not enable_thinking:
        return "reasoning_effort:no_think"
    if reasoning_effort in ("high", "xhigh"):
        return "reasoning_effort:high"
    if reasoning_effort == "max":
        return "reasoning_effort:max"
    if reasoning_effort in ("low", "minimal"):
        return "reasoning_effort:low"
    if reasoning_effort == "medium":
        return "reasoning_effort:medium"
    return "reasoning_effort:high"


def render_chat_hy3(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                    tool_choice=None):
    """Render the official Hunyuan Hy3 chat template (tencent/Hy3, `:opensource` tokens),
    including the <tool_calls:opensource> tool-declaration/tool-call/tool-response format."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    has_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role == "user":
            has_user = True
        elif role not in ("system", "developer", "assistant", "tool"):
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
    if not has_user:
        raise APIError(400, "Hy3 chat requires at least one user message.", "messages")

    tools, forced = _resolve_tool_choice(tools, tool_choice)
    system_prompt = "\n\n".join(content_text(m.get("content"), f"messages.{i}.content")
                                for i, m in enumerate(messages) if m.get("role") in ("system", "developer"))
    effort = hy3_effort(enable_thinking, reasoning_effort)     # e.g. "reasoning_effort:high"
    if not tools:
        # Insert a delimiter before the reasoning-mode marker so it never fuses with the
        # last system/developer message (and multiple system messages stay separated).
        system_prompt = (system_prompt + "\n\n" if system_prompt else "") + HY3_REASONING_MODE + effort

    parts = [HY3_BOS, system_prompt]

    if tools:
        parts.append(("\n\n" if system_prompt else "") +
                     "# Tools\n\nYou may call one or more functions to assist with the user "
                     "query.\n\nYou are provided with function signatures within <tools></tools> "
                     "XML tags:\n<tools>\n")
        clean_tools = []
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean_tools.append({k: v for k, v in fn.items() if k not in ("defer_loading", "strict")})
        parts.append("\n".join(json.dumps(t, ensure_ascii=False) for t in clean_tools))
        parts.append("\n</tools>\n\n"
                     "For function call returns, you should first print " + HY3_TOOLCALLS_OPEN + "\n"
                     "For each function call, you should return object like:\n"
                     + HY3_TOOLCALL_OPEN + "{function-name}" + HY3_TOOLSEP + "\n"
                     + HY3_ARGKEY_OPEN + "{arg-key-1}" + HY3_ARGKEY_CLOSE + "\n"
                     + HY3_ARGVALUE_OPEN + "{arg-value-1}" + HY3_ARGVALUE_CLOSE + "\n"
                     + HY3_ARGKEY_OPEN + "{arg-key-2}" + HY3_ARGKEY_CLOSE + "\n"
                     + HY3_ARGVALUE_OPEN + "{arg-value-2}" + HY3_ARGVALUE_CLOSE + "\n"
                     "...\n" + HY3_TOOLCALL_CLOSE + "\n"
                     "At the end of function call returns, you should print "
                     + HY3_TOOLCALLS_CLOSE + HY3_REASONING_MODE + effort)
        if forced:
            parts.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
        elif tool_choice == "required":
            parts.append("\n\nYou must call one of the functions above. Do not answer directly.")

    prev_tool = False
    for index, message in enumerate(messages):
        role = message.get("role")
        if role in ("system", "developer"):
            continue                                  # already folded into system_prompt above
        if role == "user":
            text = content_text(message.get("content"), f"messages.{index}.content")
            if prev_tool:
                parts.append(HY3_TOOLRESPS_CLOSE)
            parts.append(HY3_USER + text)
            prev_tool = False
        elif role == "assistant":
            raw = message.get("content")
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            if prev_tool:
                parts.append(HY3_TOOLRESPS_CLOSE)
            parts.append(HY3_ASSISTANT)
            calls = message.get("tool_calls")
            if calls:
                parts.append(HY3_THINK_OPEN + HY3_THINK_CLOSE + text)
                parts.append(HY3_TOOLCALLS_OPEN + "\n")
                for tc in calls:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    parts.append(HY3_TOOLCALL_OPEN + (fn.get("name") or "") + HY3_TOOLSEP + "\n")
                    for key, value in (args or {}).items():
                        value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                        parts.append(HY3_ARGKEY_OPEN + key + HY3_ARGKEY_CLOSE + "\n"
                                     + HY3_ARGVALUE_OPEN + value + HY3_ARGVALUE_CLOSE + "\n")
                    parts.append(HY3_TOOLCALL_CLOSE + "\n")
                # Match the declared tool-call format: the model is told to end its tool calls
                # with </tool_calls:opensource> followed by the reasoning-mode marker, then EOS.
                parts.append(HY3_TOOLCALLS_CLOSE + HY3_REASONING_MODE + effort + HY3_EOS)
            else:
                parts.append(HY3_THINK_OPEN + HY3_THINK_CLOSE + text + HY3_EOS)
            prev_tool = False
        elif role == "tool":
            text = content_text(message.get("content"), f"messages.{index}.content")
            if not prev_tool:
                parts.append(HY3_TOOLRESPS_OPEN + "\n")
            parts.append(HY3_TOOLRESP_OPEN + "\n" + text + "\n" + HY3_TOOLRESP_CLOSE + "\n")
            prev_tool = True
    if prev_tool:
        parts.append(HY3_TOOLRESPS_CLOSE)
    parts.append(HY3_ASSISTANT + (HY3_THINK_OPEN if enable_thinking
                                  else HY3_THINK_OPEN + HY3_THINK_CLOSE))
    return "".join(parts)


def model_family(model_dir):
    try:
        cfg = json.loads(Path(model_dir).joinpath("config.json").read_text(encoding="utf-8"))
        return cfg.get("model_type", "glm")
    except (OSError, json.JSONDecodeError, TypeError):
        return "glm"


def default_model_id(model_dir):
    return "hy3-colibri" if model_family(model_dir) == "hy_v3" else "glm-5.2-colibri"


def generation_options(body, limit):
    if body.get("n", 1) != 1:
        raise APIError(400, "Colibri currently supports `n=1` only.", "n", "unsupported_value")
    # `tools`/`functions` are handled by render_chat (declaration) + parse_tool_calls (output).
    choice = body.get("tool_choice")
    if choice is not None:
        if isinstance(choice, str):
            if choice not in ("auto", "none", "required"):
                raise APIError(400, "`tool_choice` must be one of \"auto\", \"none\", \"required\", "
                                    "or a function object.", "tool_choice", "unsupported_value")
        elif isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            if not name:
                raise APIError(400, "`tool_choice` function object must include a name.",
                               "tool_choice", "invalid_value")
            declared = [(t.get("function", t) if isinstance(t, dict) else {}).get("name")
                        for t in (body.get("tools") or body.get("functions") or [])]
            if name not in declared:
                raise APIError(400, f"`tool_choice` names {name!r}, which is not in `tools`.",
                               "tool_choice", "invalid_value")
        else:
            raise APIError(400, "`tool_choice` must be a string or a function object.",
                           "tool_choice", "invalid_value")
        if choice != "none" and not (body.get("tools") or body.get("functions")):
            raise APIError(400, "`tool_choice` requires `tools`.", "tool_choice", "invalid_value")
    if body.get("stop") is not None:
        raise APIError(400, "Custom stop sequences are not supported yet.", "stop", "unsupported_parameter")
    if body.get("logprobs"):
        raise APIError(400, "Log probabilities are not supported yet.", "logprobs", "unsupported_parameter")
    if body.get("frequency_penalty", 0) or body.get("presence_penalty", 0):
        raise APIError(400, "Token penalties are not supported yet.", None, "unsupported_parameter")
    if body.get("seed") is not None:
        raise APIError(400, "Per-request seeds are not supported yet.", "seed", "unsupported_parameter")
    response_format = body.get("response_format")
    if response_format not in (None, {"type": "text"}):
        raise APIError(400, "Only the default text response format is supported.",
                       "response_format", "unsupported_parameter")

    maximum = body.get("max_completion_tokens")
    maximum_param = "max_completion_tokens"
    if maximum is None:
        maximum = body.get("max_tokens")
        maximum_param = "max_tokens"
    if maximum is None:
        maximum = min(256, limit)
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    temperature = 0.7 if temperature is None else temperature
    top_p = 0.9 if top_p is None else top_p
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= limit:
        raise APIError(400, f"`{maximum_param}` must be an integer between 1 and {limit}.", maximum_param)
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
            not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise APIError(400, "`temperature` must be between 0 and 2.", "temperature")
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or
            not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise APIError(400, "`top_p` must be greater than 0 and at most 1.", "top_p")
    return maximum, float(temperature), float(top_p)


def read_engine_turn(stream, sentinel, on_bytes):
    pending = b""
    while True:
        byte = stream.read(1)
        if byte == b"":
            raise RuntimeError("colibri engine exited unexpectedly")
        pending += byte
        if pending.endswith(sentinel):
            data = pending[:-len(sentinel)]
            if data:
                on_bytes(data)
            break
        if len(pending) > len(sentinel):
            on_bytes(pending[:-len(sentinel)])
            pending = pending[-len(sentinel):]

    fields = stream.readline().decode("utf-8", "replace").strip().split()
    if len(fields) < 5 or fields[0] != "STAT":
        raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
    return {
        "completion_tokens": int(fields[1]),
        "tokens_per_second": float(fields[2]),
        "cache_hit_percent": float(fields[3]),
        "rss_gb": float(fields[4]),
        "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
        "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
        "decode_tokens_per_second": float(fields[7]) if len(fields) > 7 else None,
    }


class Engine:
    def __init__(self, executable, model, cap=8, max_tokens=1024, env=None, kv_slots=1, family=None):
        child_env = dict(env or os.environ, SNAP=str(model), SERVE="1", SERVE_BATCH="1",
                         NGEN=str(max_tokens), KV_SLOTS=str(kv_slots))
        fam = family or model_family(model)
        if fam == "hy_v3":
            child_env.setdefault("IDOT", "0")
            argv = [str(executable), str(cap), "4", "8"]
        else:
            argv = [str(executable), str(cap)]
        # hy3.c has no run_serve_mux/step_decode_batch (those symbols live only in glm.c) --
        # it silently ignores SERVE_BATCH and always runs the old single-sequence run_serve(),
        # which speaks \x02PROMPT/END, not SUBMIT/DATA/DONE. Fall back to that protocol for Hy3
        # so the SUBMIT header text isn't mistaken for a literal chat message.
        self.protocol = "single" if fam == "hy_v3" else "mux"
        self.process = subprocess.Popen(
            argv, env=child_env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, bufsize=0,
        )
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.next_request_id = 1
        self.closed = False
        self.dispatcher_error = None
        self.kv_slots = kv_slots
        # run_serve() handles one line of the wire protocol at a time (no request ids to
        # disambiguate interleaved replies), so single-protocol requests must be fully
        # serialized -- this lock is unused by the mux protocol.
        self.serialize_lock = threading.Lock()
        read_engine_turn(self.process.stdout, READY, lambda _: None)
        self.dispatcher = None
        if self.protocol == "mux":
            self.dispatcher = threading.Thread(target=self._dispatch_stdout,
                                               name="colibri-stdout", daemon=True)
            self.dispatcher.start()

    @staticmethod
    def _stats(fields):
        if len(fields) < 5 or fields[0] != "STAT":
            raise RuntimeError(f"invalid engine status: {' '.join(fields)}")
        return {
            "completion_tokens": int(fields[1]),
            "tokens_per_second": float(fields[2]),
            "cache_hit_percent": float(fields[3]),
            "rss_gb": float(fields[4]),
            "prompt_tokens": int(fields[5]) if len(fields) > 5 else 0,
            "length_limited": bool(int(fields[6])) if len(fields) > 6 else False,
            "decode_tokens_per_second": float(fields[7]) if len(fields) > 7 else None,
        }

    def _fail_pending(self, error):
        with self.pending_lock:
            requests = list(self.pending.values())
            self.pending.clear()
        for events in requests:
            events.put(("error", error))

    def _read_exact(self, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if chunk == b"":
                raise RuntimeError("truncated engine DATA payload")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _dispatch_stdout(self):
        try:
            while True:
                line = self.process.stdout.readline()
                if line == b"":
                    raise RuntimeError("colibri engine exited unexpectedly")
                fields = line.decode("utf-8", "replace").strip().split()
                if not fields:
                    continue
                kind = fields[0]
                if kind == "DATA" and len(fields) == 3:
                    request_id = fields[1]
                    size = int(fields[2])
                    if not 0 <= size <= 65536:
                        raise RuntimeError("invalid engine DATA size")
                    data = self._read_exact(size)
                    if self._read_exact(1) != b"\n":
                        raise RuntimeError("invalid engine DATA terminator")
                    with self.pending_lock:
                        events = self.pending.get(request_id)
                    if events is not None:
                        events.put(("data", data))
                elif kind == "DONE" and len(fields) >= 7:
                    request_id = fields[1]
                    stats = self._stats(fields[2:])
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("done", stats))
                elif kind == "ERROR" and len(fields) >= 2:
                    request_id = fields[1]
                    message = " ".join(fields[2:]) or "engine request failed"
                    with self.pending_lock:
                        events = self.pending.pop(request_id, None)
                    if events is not None:
                        events.put(("error", RuntimeError(message)))
                else:
                    raise RuntimeError(f"invalid engine response: {' '.join(fields)}")
        except Exception as error:
            if not self.closed:
                self.dispatcher_error = error
                self._fail_pending(error)

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None):
        if self.protocol == "single":
            return self._generate_single(prompt, max_tokens, temperature, top_p, on_text,
                                         cache_slot, cancelled)
        return self._generate_mux(prompt, max_tokens, temperature, top_p, on_text,
                                  cache_slot, cancelled)

    def _generate_single(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                         cancelled=None):
        """Hy3 fallback wire protocol (see the comment in __init__): \\x02PROMPT <bytes>
        <max_tokens> <temperature> <top_p> <slot>\\n<payload>\\n in, a raw byte stream
        terminated by \\x01\\x01END\\x01\\x01\\n plus a STAT line out. The engine handles one
        line/turn at a time -- serialize_lock enforces one in-flight request system-wide.
        Unlike the mux protocol there is no mid-stream CANCEL: `cancelled` is only honored
        before submission, not once generation has started."""
        if isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or not 0 <= cache_slot < self.kv_slots:
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def decode(data):
            text = decoder.decode(data)
            if text:
                on_text(text)

        with self.serialize_lock:
            if self.closed:
                raise RuntimeError("colibri engine is shutting down")
            if self.process.poll() is not None:
                raise RuntimeError("colibri engine is not running")
            if cancelled and cancelled():
                raise ClientCancelled()
            header = (f"\x02PROMPT {len(payload)} {max_tokens} {temperature:.8g} "
                     f"{top_p:.8g} {cache_slot}\n").encode()
            self.process.stdin.write(header + payload + b"\n")
            self.process.stdin.flush()
            stats = read_engine_turn(self.process.stdout, END, decode)
            tail = decoder.decode(b"", final=True)
            if tail:
                on_text(tail)
            return stats

    def _generate_mux(self, prompt, max_tokens, temperature, top_p, on_text, cache_slot=0,
                      cancelled=None):
        if isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or not 0 <= cache_slot < self.kv_slots:
            raise APIError(400, "Invalid cache slot.", "cache_slot")
        payload = prompt.encode("utf-8")
        if b"\0" in payload:
            raise APIError(400, "NUL bytes are not supported in prompts.", "messages")
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def decode(data):
            text = decoder.decode(data)
            if text:
                on_text(text)

        events = queue.Queue()
        with self.pending_lock:
            if self.closed:
                raise RuntimeError("colibri engine is shutting down")
            if self.dispatcher_error is not None:
                raise RuntimeError("colibri engine dispatcher stopped") from self.dispatcher_error
            if self.process.poll() is not None:
                raise RuntimeError("colibri engine is not running")
            request_id = str(self.next_request_id)
            self.next_request_id += 1
            self.pending[request_id] = events
        header = (f"SUBMIT {request_id} {cache_slot} {len(payload)} {max_tokens} "
                  f"{temperature:.8g} {top_p:.8g}\n").encode()
        try:
            with self.write_lock:
                if self.process.poll() is not None:
                    raise RuntimeError("colibri engine is not running")
                self.process.stdin.write(header + payload + b"\n")
                self.process.stdin.flush()
        except Exception:
            with self.pending_lock:
                self.pending.pop(request_id, None)
            raise

        cancel_sent = False
        while True:
            kind, value = events.get()
            if kind == "data":
                if not cancel_sent:
                    decode(value)
                    if cancelled and cancelled():
                        cancel_sent = True
                        with self.write_lock:
                            self.process.stdin.write(f"CANCEL {request_id}\n".encode())
                            self.process.stdin.flush()
            elif kind == "done":
                tail = decoder.decode(b"", final=True)
                if tail:
                    on_text(tail)
                return value
            elif cancel_sent and isinstance(value, RuntimeError) and str(value) == "CANCELLED":
                raise ClientCancelled()
            else:
                raise value

    def close(self):
        with self.pending_lock:
            if self.closed:
                return
            self.closed = True
        self._fail_pending(RuntimeError("colibri engine is shutting down"))
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.dispatcher is not None and self.dispatcher is not threading.current_thread():
            self.dispatcher.join(timeout=5)


def model_object(model_id, created):
    return {"id": model_id, "object": "model", "created": created, "owned_by": "colibri"}


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine, model_id, api_key=None, max_tokens=1024,
                 cors_origins=DEFAULT_CORS_ORIGINS, max_queue=8, queue_timeout=300,
                 kv_slots=1, family="glm"):
        super().__init__(address, APIHandler)
        self.engine = engine
        self.model_id = model_id
        self.model_family = family
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.scheduler = GenerationScheduler(max_queue, queue_timeout, kv_slots)
        self.kv_slots = kv_slots
        self.cors_origins = tuple(cors_origins)
        self.created = int(time.time())


class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "colibri"

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status, body, request_id=None, headers=None):
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if request_id:
            self.send_header("x-request-id", request_id)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def send_cors_headers(self):
        origin = self.headers.get("Origin")
        if not origin or ("*" not in self.server.cors_origins and origin not in self.server.cors_origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in self.server.cors_origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Expose-Headers",
                         "x-request-id, x-colibri-queue-wait-ms, Retry-After")
        self.send_header("Access-Control-Max-Age", "600")
        if "*" not in self.server.cors_origins:
            self.send_header("Vary", "Origin")

    def require_auth(self):
        if self.server.api_key and self.headers.get("Authorization") != f"Bearer {self.server.api_key}":
            raise APIError(401, "Invalid or missing API key.", None, "invalid_api_key",
                           "authentication_error")

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "Invalid Content-Length header.")
        if length < 1 or length > MAX_BODY:
            raise APIError(400, f"Request body must be between 1 and {MAX_BODY} bytes.")
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError(400, "Request body must be valid JSON.")
        if not isinstance(body, dict):
            raise APIError(400, "Request body must be a JSON object.")
        return body

    def check_model(self, body):
        model = body.get("model")
        if model != self.server.model_id:
            raise APIError(404, f"The model `{model}` does not exist.", "model", "model_not_found")

    def do_GET(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            path = urlsplit(self.path).path
            if path == "/health":
                self.send_json(200, {"status": "ok", "scheduler": self.server.scheduler.snapshot(),
                                     "kv_slots": self.server.kv_slots}, request_id)
                return
            self.require_auth()
            if path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [model_object(
                    self.server.model_id, self.server.created)]}, request_id)
            elif path.startswith("/v1/models/") and unquote(path[11:]) == self.server.model_id:
                self.send_json(200, model_object(self.server.model_id, self.server.created), request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        request_id = "req_" + uuid.uuid4().hex
        try:
            self.require_auth()
            body = self.read_json()
            self.check_model(body)
            path = urlsplit(self.path).path
            if path == "/v1/chat/completions":
                self.chat_completion(body, request_id)
            elif path == "/v1/completions":
                self.completion(body, request_id)
            else:
                raise APIError(404, "Not found.", None, "not_found")
        except APIError as error:
            self.send_json(error.status, error_object(error), request_id, error.headers)
        except ClientCancelled:
            pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.log_error("request failed: %s", error)
            api_error = APIError(500, "The colibri engine failed to process the request.",
                                 None, "engine_error", "server_error")
            try:
                self.send_json(500, error_object(api_error), request_id)
            except OSError:
                pass

    def generation(self, body, prompt, request_id, chat, enable_thinking=False):
        maximum, temperature, top_p = generation_options(body, self.server.max_tokens)
        tools = (body.get("tools") or body.get("functions") or None) if chat else None
        if body.get("tool_choice") == "none":
            tools = None          # client forbade tools: never surface tool_calls
        hy3 = chat and self.server.model_family == "hy_v3"
        hy3_thinking = hy3 and enable_thinking  # Hy3-only: Stage-1 reasoning splitter engages
        box_start = HY3_TOOLCALLS_OPEN if hy3 else BOX_START
        parse_calls = parse_tool_calls_hy3 if hy3 else parse_tool_calls
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        stream_options = body.get("stream_options") if stream else None
        if stream and stream_options is not None and not isinstance(stream_options, dict):
            raise APIError(400, "`stream_options` must be an object.", "stream_options")
        include_usage = bool((stream_options or {}).get("include_usage"))
        object_name = "chat.completion" if chat else "text_completion"
        id_prefix = "chatcmpl-" if chat else "cmpl-"
        completion_id = id_prefix + uuid.uuid4().hex
        created = int(time.time())

        with self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            queue_headers = {"x-colibri-queue-wait-ms": str(round(queue_wait * 1000))}
            if not stream:
                output = []
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, output.append, cache_slot,
                    self.client_disconnected)
                text = "".join(output)
                length_finish = "length" if stats["length_limited"] else "stop"
                if hy3_thinking:
                    reasoning, rest, truncated = split_reasoning_hy3(text, True)
                    if chat and tools:
                        content, calls = parse_calls(rest, tools)
                    else:
                        content, calls = rest, []
                    message = {"role": "assistant", "content": content or None, "refusal": None}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                        message["reasoning"] = reasoning
                    if calls:
                        message["tool_calls"] = calls
                    if truncated:
                        message["colibri_reasoning_truncated"] = True
                    finish = "tool_calls" if calls else length_finish
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                elif chat and tools:
                    content, calls = parse_calls(text, tools)
                    message = {"role": "assistant", "content": content or None, "refusal": None}
                    if calls:
                        message["tool_calls"] = calls
                    finish = "tool_calls" if calls else length_finish
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                else:
                    choice = ({"index": 0, "message": {"role": "assistant", "content": text,
                               "refusal": None}, "logprobs": None, "finish_reason": length_finish} if chat else
                              {"index": 0, "text": text, "logprobs": None, "finish_reason": length_finish})
                self.send_json(200, {"id": completion_id, "object": object_name, "created": created,
                    "model": self.server.model_id, "choices": [choice], "usage": self.usage(stats)},
                    request_id, queue_headers)
                return

            stream_object = "chat.completion.chunk" if chat else object_name
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("x-request-id", request_id)
            for name, value in queue_headers.items(): self.send_header(name, value)
            self.send_cors_headers()
            self.end_headers()
            connected = True
            # KEEPALIVE: engine.generate() blocks SILENTLY during the (minutes-long) cold
            # prefill, and the client drops the socket after its idle timeout. A background pump
            # emits a reasoning_content "." delta (the channel that reliably resets the client's
            # timer and lands in the thinking panel, so answer content stays clean) whenever no
            # event has been written for KA_GAP seconds. All wfile writes share ka_lock so the
            # pump and event() never interleave; last_write gates the pump so it stays quiet
            # while real tokens are flowing (e.g. during decode).
            ka_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()
            KA_GAP = 10.0
            dbg_echo = os.environ.get("COLI_DEBUG", "0") == "1"   # tee decoded tokens to stderr

            def event(choices, usage_marker=False):
                nonlocal connected
                if not connected:
                    return
                event_body = {"id": completion_id, "object": stream_object, "created": created,
                              "model": self.server.model_id, "choices": choices}
                if include_usage:
                    event_body["usage"] = None if not usage_marker else usage_marker
                data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                with ka_lock:
                    try:
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected = False

            def _keepalive():
                ping = [{"index": 0, "delta": ({"reasoning_content": "."} if chat else {"content": ""}),
                         "logprobs": None, "finish_reason": None}]
                while not ka_stop.wait(1.0):
                    if not connected:
                        return
                    if time.time() - last_write[0] >= KA_GAP:
                        event(ping)

            if chat:
                event([{"index": 0, "delta": {"role": "assistant", "content": ""},
                        "logprobs": None, "finish_reason": None}])

            def emit(text):
                choice = ({"index": 0, "delta": {"content": text}, "logprobs": None,
                           "finish_reason": None} if chat else
                          {"index": 0, "text": text, "logprobs": None, "finish_reason": None})
                event([choice])

            ka_thread = threading.Thread(target=_keepalive, daemon=True)
            ka_thread.start()

            # Stage-1 reasoning splitter (Hy3 only, when thinking enabled).
            # Wraps whatever Stage-2 closure (emit_tools/emit_plain) is chosen below.
            # When hy3_thinking is True, the prompt ends with an unclosed <think:opensource>,
            # so the first bytes from the engine are always reasoning text.  We buffer them
            # and emit as reasoning_content deltas until </think:opensource> is found, then
            # hand off the remainder to Stage 2 (unchanged tool/plain logic).
            rs = None
            reasoning_raw = []
            if hy3_thinking:
                rs = {"buf": "", "done": False}
                THINK_HOLD = len(HY3_THINK_CLOSE) - 1

                def emit_reasoning(text):
                    reasoning_raw.append(text)
                    if dbg_echo:
                        sys.stderr.write(text); sys.stderr.flush()
                    if not text:
                        return
                    choice = [{"index": 0, "delta": {"reasoning_content": text, "reasoning": text},
                               "logprobs": None, "finish_reason": None}]
                    event(choice)

            def wrap_stage1(stage2):
                """If hy3_thinking is on, return a Stage-1 reasoning-split wrapper that feeds
                post-marker text into stage2.  Otherwise return stage2 unchanged."""
                if not rs:
                    return stage2
                def _fn(chunk):
                    if rs["done"]:
                        stage2(chunk)
                        return
                    rs["buf"] += chunk
                    idx = rs["buf"].find(HY3_THINK_CLOSE)
                    if idx >= 0:
                        if idx:
                            emit_reasoning(rs["buf"][:idx])
                        rs["done"] = True
                        remainder = rs["buf"][idx + len(HY3_THINK_CLOSE):]
                        rs["buf"] = ""
                        if remainder:
                            stage2(remainder)
                        return
                    flush = max(0, len(rs["buf"]) - THINK_HOLD)
                    if flush:
                        emit_reasoning(rs["buf"][:flush])
                        rs["buf"] = rs["buf"][flush:]
                return _fn

            if chat and tools:
                # Suppress tool-call markers from the streamed content and parse the authoritative
                # calls from the FULL reply after generation. Hold back a marker-length tail so a
                # <tool_call> split across engine chunks is still caught.
                sp = {"buf": "", "tool": False}
                hold = len(box_start) - 1
                raw = []
                def emit_tools(chunk):
                    raw.append(chunk)
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    if sp["tool"]:
                        return
                    sp["buf"] += chunk
                    cut = sp["buf"].find(box_start)
                    if cut >= 0:
                        if cut:
                            emit(sp["buf"][:cut])
                        sp["buf"] = ""
                        sp["tool"] = True
                        return
                    flush = max(0, len(sp["buf"]) - hold)
                    if flush:
                        emit(sp["buf"][:flush])
                        sp["buf"] = sp["buf"][flush:]
                emit_target = wrap_stage1(emit_tools)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, emit_target, cache_slot,
                    lambda: not connected)
                if not sp["tool"] and sp["buf"]:
                    emit(sp["buf"])                     # no tool call happened: flush held tail
                _content, calls = parse_calls("".join(raw), tools)
                for i, tc in enumerate(calls):
                    event([{"index": 0, "delta": {"tool_calls": [{"index": i, "id": tc["id"],
                             "type": "function", "function": {"name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]}}]},
                            "logprobs": None, "finish_reason": None}])
                finish = "tool_calls" if calls else ("length" if stats["length_limited"] else "stop")
            else:
                def emit_plain(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    emit(chunk)
                emit_target = wrap_stage1(emit_plain)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, emit_target, cache_slot,
                    lambda: not connected)
                finish = "length" if stats["length_limited"] else "stop"
            ka_stop.set()                          # generation done: stop the keepalive pump
            ka_thread.join(timeout=2)
            # Streaming final-fixup chunk: for Hy3 thinking, if Stage 1 never saw the close
            # marker, flag truncation on the final chunk so clients can detect it.
            if hy3_thinking and rs is not None and not rs["done"]:
                # </think:opensource> never arrived (truncation per 3c): Stage-1 already
                # delivered the output as reasoning_content deltas. Clients that ignore
                # reasoning_content would otherwise miss it, so replay the accumulated
                # reasoning text as visible content too (including the tail held in rs["buf"]).
                full = "".join(reasoning_raw)
                if rs["buf"]:
                    full += rs["buf"]
                    rs["buf"] = ""
                if full:
                    emit(full)
                final_choice = {"index": 0, "delta": {}, "logprobs": None,
                                "finish_reason": finish,
                                "colibri_reasoning_truncated": True}
            else:
                final_choice = ({"index": 0, "delta": {}, "logprobs": None,
                                 "finish_reason": finish} if chat else
                                {"index": 0, "text": "", "logprobs": None,
                                 "finish_reason": finish})
            event([final_choice])
            if include_usage:
                event([], self.usage(stats))
            if connected:
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
            self.close_connection = True

    def client_disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            return self.connection.recv(1, flags) == b""
        except (OSError, ValueError):
            return True

    @staticmethod
    def usage(stats):
        prompt = stats["prompt_tokens"]
        completion = stats["completion_tokens"]
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion}

    def chat_completion(self, body, request_id):
        reasoning_effort = body.get("reasoning_effort")
        efforts = (None, "none", "minimal", "low", "medium", "high", "xhigh")
        if reasoning_effort not in efforts:
            raise APIError(400, "`reasoning_effort` must be none, minimal, low, medium, high, or xhigh.",
                           "reasoning_effort")
        # COLI_THINK=1 makes thinking the default when the client sends NEITHER reasoning_effort
        # nor enable_thinking (a global switch, like the old server's --think). An explicit
        # client value always wins. Default off => exact OpenAI-standard behavior.
        if (reasoning_effort is None and "enable_thinking" not in body
                and os.environ.get("COLI_THINK", "0") == "1"):
            reasoning_effort = "high"
        enable_thinking = body.get("enable_thinking", reasoning_effort not in (None, "none"))
        if not isinstance(enable_thinking, bool):
            raise APIError(400, "`enable_thinking` must be a boolean.", "enable_thinking")
        tools = body.get("tools") or body.get("functions") or None
        if self.server.model_family == "hy_v3":
            prompt = render_chat_hy3(body.get("messages"), enable_thinking, reasoning_effort, tools,
                                     body.get("tool_choice"))
        else:
            prompt = render_chat(body.get("messages"), enable_thinking, reasoning_effort, tools,
                                 body.get("tool_choice"))
        self.generation(body, prompt, request_id, True, enable_thinking)

    def completion(self, body, request_id):
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "Colibri currently requires `prompt` to be a string.", "prompt")
        self.generation(body, prompt, request_id, False, enable_thinking=False)


def serve(model, host="127.0.0.1", port=8000, model_id="glm-5.2-colibri", api_key=None,
          cap=8, max_tokens=1024, engine=HERE / "glm", env=None, cors_origins=None,
          max_queue=8, queue_timeout=300, kv_slots=1):
    if not 1 <= max_tokens:
        raise ValueError("max_tokens must be positive")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if max_queue < 0:
        raise ValueError("max_queue cannot be negative")
    if queue_timeout <= 0:
        raise ValueError("queue_timeout must be positive")
    if not 1 <= kv_slots <= 16:
        raise ValueError("kv_slots must be between 1 and 16")
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print("WARNING: API is listening beyond localhost without COLI_API_KEY", file=sys.stderr)
    origins = DEFAULT_CORS_ORIGINS if cors_origins is None else tuple(cors_origins)
    family = model_family(model)
    # Bind before starting the engine. A stale/occupied port must fail in
    # milliseconds rather than loading hundreds of GB and leaking a child.
    server = APIServer((host, port), None, model_id, api_key, max_tokens, origins,
                       max_queue, queue_timeout, kv_slots, family)
    runtime = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        runtime = Engine(engine, model, cap, max_tokens, env, kv_slots, family)
        server.engine = runtime
        print(f"OpenAI-compatible API listening on http://{host}:{port}/v1", file=sys.stderr)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.scheduler.close()
        server.server_close()
        if runtime is not None:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("COLI_MODEL"), required=not os.environ.get("COLI_MODEL"))
    parser.add_argument("--engine", default=str(HERE / "glm"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--api-key", default=os.environ.get("COLI_API_KEY"))
    parser.add_argument("--cors-origin", action="append", default=None,
                        help="allowed browser origin; repeat as needed (use '*' for any origin)")
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-queue", type=int, default=int(os.environ.get("COLI_MAX_QUEUE", "8")))
    parser.add_argument("--queue-timeout", type=float,
                        default=float(os.environ.get("COLI_QUEUE_TIMEOUT", "300")))
    parser.add_argument("--kv-slots", type=int, default=int(os.environ.get("COLI_KV_SLOTS", "1")))
    args = parser.parse_args()
    model_id = args.model_id or os.environ.get("COLI_MODEL_ID") or default_model_id(args.model)
    serve(args.model, args.host, args.port, model_id, args.api_key,
          args.cap,args.max_tokens,args.engine,cors_origins=args.cors_origin,
          max_queue=args.max_queue,queue_timeout=args.queue_timeout,kv_slots=args.kv_slots)


if __name__ == "__main__":
    main()
