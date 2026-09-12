"""Single-stream decode throughput via streaming completion. usage: bench_tps.py [--url] [--model] [--tokens 256]"""
import json, sys, time, urllib.request, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8888/v1"); ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
ap.add_argument("--tokens", type=int, default=256); ap.add_argument("--prompt", default="Write a detailed, step-by-step explanation of how a hash map works, including collisions and resizing."); a = ap.parse_args()
body = json.dumps({"model": a.model, "prompt": a.prompt, "max_tokens": a.tokens, "temperature": 0, "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}).encode()
t0 = time.time(); first = None; n = 0; usage_tokens = None
with urllib.request.urlopen(urllib.request.Request(a.url + "/completions", body, {"Content-Type": "application/json"}), timeout=900) as r:
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
        d = json.loads(line[6:])
        if d.get("usage") and d["usage"].get("completion_tokens"): usage_tokens = d["usage"]["completion_tokens"]
        if d.get("choices") and d["choices"][0].get("text"):
            n += 1
            if first is None: first = time.time()
t1 = time.time()
tok = usage_tokens or n
print(f"TTFT {first-t0:.2f}s | {tok} tokens ({n} chunks) in {t1-first:.2f}s -> {tok/(t1-first):.1f} tok/s decode (single stream)")
