#!/usr/bin/env python3
"""Manual end-to-end tool-calling probe against a running colibri server (#401).

Runs a real two-turn loop against the OpenAI-compatible endpoint: declare a tool, let the
model call it, execute it locally, feed the result back, and check the model uses it.
No client library and no dependencies -- stdlib only.

    python3 tools/try_tool_calling.py                        # default http://127.0.0.1:8080
    python3 tools/try_tool_calling.py --url http://host:port
    python3 tools/try_tool_calling.py --raw                  # also dump the raw model text

Exit status is 0 only if every stage passed, so it can be used as a smoke test.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Rome"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

# What the tool "really" returns, so the second turn has a fact the model could not invent.
FAKE_WEATHER = {"city": "Rome", "temp_c": 31, "conditions": "sunny"}


def discover_model(url, timeout):
    """Ask the server which model it is serving, so the caller never has to guess the id."""
    with urllib.request.urlopen(url + "/v1/models", timeout=timeout) as resp:
        data = json.loads(resp.read()).get("data") or []
    if not data:
        raise RuntimeError("server reports no models")
    return data[0]["id"]


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="server base URL")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per request")
    ap.add_argument("--raw", action="store_true", help="print raw message objects")
    ap.add_argument("--tool-choice", default=None, help='e.g. "required" to force a call')
    ap.add_argument("--model", default=None, help="model id (default: ask the server)")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    try:
        model = args.model or discover_model(url, 30)
    except (urllib.error.URLError, OSError) as e:
        print(f"FAIL: cannot reach {url} -- is the server running?  ({e})")
        return 2
    print(f"server: {url}   model: {model}")

    messages = [{"role": "user", "content": "What's the weather in Rome right now? "
                                            "Use the tool, don't guess."}]
    body = {"model": model, "messages": messages, "tools": [WEATHER_TOOL],
            "temperature": 0, "max_tokens": 256}
    if args.tool_choice:
        body["tool_choice"] = (args.tool_choice if args.tool_choice in ("auto", "none", "required")
                               else {"type": "function", "function": {"name": args.tool_choice}})

    print("== turn 1: asking the model to call the tool ==")
    try:
        out = post(url, body, args.timeout)
    except urllib.error.URLError as e:
        print(f"FAIL: cannot reach {url} -- is the server running?  ({e})")
        return 2
    msg = out["choices"][0]["message"]
    if args.raw:
        print(json.dumps(msg, indent=2, ensure_ascii=False))

    calls = msg.get("tool_calls") or []
    if not calls:
        print("FAIL: the model returned no tool_calls.")
        print(f"      content: {(msg.get('content') or '')[:400]!r}")
        print("      If the content shows <tool_call> markers, the parse is the problem -- "
              "check the server's stderr line starting with '[api]'.")
        return 1

    call = calls[0]
    name = call["function"]["name"]
    try:
        parsed = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError:
        print(f"FAIL: arguments are not valid JSON: {call['function']['arguments']!r}")
        return 1
    print(f"  tool_calls: {len(calls)}  name={name}  arguments={parsed}")

    if name != "get_weather":
        print(f"FAIL: model called {name!r}, not the declared tool.")
        return 1
    if "city" not in parsed:
        print(f"FAIL: required parameter 'city' missing from {parsed}.")
        return 1
    if "rom" not in str(parsed["city"]).lower():
        print(f"WARN: city is {parsed['city']!r}, expected Rome -- continuing anyway.")

    print("== turn 2: feeding the tool result back ==")
    messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
    messages.append({"role": "tool", "tool_call_id": call["id"], "name": name,
                     "content": json.dumps(FAKE_WEATHER)})
    body["messages"] = messages
    body.pop("tool_choice", None)
    out2 = post(url, body, args.timeout)
    reply = (out2["choices"][0]["message"].get("content") or "").strip()
    if args.raw:
        print(json.dumps(out2["choices"][0]["message"], indent=2, ensure_ascii=False))
    print(f"  reply: {reply[:400]!r}")

    if "31" not in reply:
        print("FAIL: the reply does not mention 31 -- the model ignored the tool result.")
        return 1
    print("\nPASS: tool declared -> called with valid arguments -> result consumed in the answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
