"""Quick chat prompts against a local vLLM server. usage: ask.py [--url ...] [--model ...] "prompt" ["prompt" ...]"""
import argparse, json, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8889/v1/chat/completions"); ap.add_argument("--model", default="DeepSeek-V4.1-Flash-EXL3")
ap.add_argument("--max-tokens", type=int, default=220); ap.add_argument("prompts", nargs="+"); a = ap.parse_args()
for q in a.prompts:
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": q}], "max_tokens": a.max_tokens, "temperature": 0}).encode()
    t0 = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(a.url, body, {"Content-Type": "application/json"}), timeout=900))
    m = r["choices"][0]["message"]; txt = (m.get("content") or "") + (m.get("reasoning_content") or "")
    print(f"Q: {q[:80]}\nA ({r['usage']['completion_tokens']} tok, {time.time()-t0:.1f}s): {txt[:500].replace(chr(10), ' ')}\n", flush=True)
