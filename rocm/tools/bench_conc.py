"""Aggregate decode throughput at N concurrent streams. usage: bench_conc.py [--streams 1,2,4,8] [--tokens 256]"""
import json, time, threading, urllib.request, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8888/v1/completions"); ap.add_argument("--model", default="GLM-5.3-Flash-EXL3")
ap.add_argument("--streams", default="1,2,4,8"); ap.add_argument("--tokens", type=int, default=256); a = ap.parse_args()
def run(i, out):
    body = json.dumps({"model": a.model, "prompt": f"Write a long, detailed technical explanation of how a B-tree index works, including insertion and node splits. Part {i}.", "max_tokens": a.tokens, "temperature": 0, "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}).encode()
    t0 = time.time(); n = 0; first = None
    with urllib.request.urlopen(urllib.request.Request(a.url, body, {"Content-Type": "application/json"}), timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage") and d["usage"].get("completion_tokens"): n = d["usage"]["completion_tokens"]
            if first is None and d.get("choices") and d["choices"][0].get("text"): first = time.time()
    out[i] = (n, (first or t0) - t0, time.time() - t0)
for c in [int(x) for x in a.streams.split(",")]:
    out = {}; th = [threading.Thread(target=run, args=(i, out)) for i in range(c)]; t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; el = time.time() - t0
    tot = sum(v[0] for v in out.values()); ttft = sum(v[1] for v in out.values()) / c
    print(f"{c} streams: {tot} tokens in {el:.1f}s -> aggregate {tot/el:.1f} tok/s | per-stream {tot/c/el:.1f} tok/s | mean TTFT {ttft:.2f}s", flush=True)
