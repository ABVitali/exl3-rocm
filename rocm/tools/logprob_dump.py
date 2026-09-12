"""Dump top-k logprobs for fixed prompts from an OpenAI-compatible server (greedy, echo).
usage: logprob_dump.py OUT.json [--url http://127.0.0.1:8888/v1] [--model NAME] [--max-tokens 64]
Run once per server configuration (e.g. VLLM_EXL3_MOE_KERNEL=native vs exllamav3), then logprob_compare.py A B."""
import json, sys, time, urllib.request, argparse
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--url", default="http://127.0.0.1:8888/v1")
ap.add_argument("--model", default="GLM-5.3-Flash-EXL3"); ap.add_argument("--max-tokens", type=int, default=64); ap.add_argument("--echo", action="store_true", help="teacher-forced: score fixed texts (echo=True, max_tokens=1)"); a = ap.parse_args()
PROMPTS = [
 "The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
 "In 1969, Neil Armstrong", "Explain briefly why the sky is blue.", "SELECT name, COUNT(*) FROM users GROUP BY",
 "Translate to Italian: The weather is nice today.", "The derivative of x^3 is", "Once upon a time, in a small village,",
 "import numpy as np\n\ndef softmax(x):\n", "List three prime numbers greater than 100:", "Q: What is 17 * 23? A:",
 "The main difference between TCP and UDP is",
]
TEXTS = [
 "The Hadamard transform is a linear map defined by a square matrix whose entries are all +1 or -1 and whose rows are mutually orthogonal. In weight quantization it is applied before rounding because it spreads outliers across all coordinates, making the per-block scale less sensitive to a single large value. The inverse transform is the same matrix scaled by one over the dimension, so it can be folded into adjacent layers at no runtime cost.",
 "def merge_sort(a):\n    if len(a) <= 1:\n        return a\n    mid = len(a) // 2\n    left = merge_sort(a[:mid])\n    right = merge_sort(a[mid:])\n    out = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    out.extend(left[i:]); out.extend(right[j:])\n    return out\n",
 "Il 20 luglio 1969 Neil Armstrong divenne il primo essere umano a camminare sulla Luna, durante la missione Apollo 11. Buzz Aldrin lo seguì pochi minuti dopo, mentre Michael Collins rimase in orbita lunare a bordo del modulo di comando.",
 "TCP provides a reliable, ordered byte stream with congestion control and retransmission, at the cost of connection setup and head-of-line blocking. UDP is connectionless and unordered, delivering datagrams with minimal overhead, which is why real-time media and DNS prefer it despite the possibility of loss.",
 "Theorem. Every bounded monotone sequence of real numbers converges. Proof. Let (a_n) be increasing and bounded above, and let L be the supremum of its range. For any epsilon > 0 there is an N with a_N > L - epsilon, and monotonicity gives L - epsilon < a_n <= L for all n >= N, which is the definition of convergence to L.",
]
res = []
if a.echo:
    for p in TEXTS:
        body = json.dumps({"model": a.model, "prompt": p, "max_tokens": 1, "temperature": 0, "logprobs": 5, "echo": True, "seed": 0}).encode()
        t0 = time.time(); r = urllib.request.urlopen(urllib.request.Request(a.url + "/completions", body, {"Content-Type": "application/json"}), timeout=600)
        d = json.load(r); c = d["choices"][0]; lp = c["logprobs"]
        res.append({"prompt": p, "text": "", "tokens": lp["tokens"], "token_logprobs": [x if x is not None else 0.0 for x in lp["token_logprobs"]], "top_logprobs": lp["top_logprobs"], "secs": time.time() - t0})
        print(f"{len(lp['tokens']):3d} prompt tok scored in {time.time()-t0:5.1f}s | {p[:50]!r}")
    json.dump(res, open(a.out, "w")); print("saved", a.out); sys.exit(0)
for p in PROMPTS:
    body = json.dumps({"model": a.model, "prompt": p, "max_tokens": a.max_tokens, "temperature": 0, "logprobs": 5, "echo": False, "seed": 0}).encode()
    t0 = time.time(); r = urllib.request.urlopen(urllib.request.Request(a.url + "/completions", body, {"Content-Type": "application/json"}), timeout=600)
    d = json.load(r); c = d["choices"][0]; lp = c["logprobs"]
    res.append({"prompt": p, "text": c["text"], "tokens": lp["tokens"], "token_logprobs": lp["token_logprobs"], "top_logprobs": lp["top_logprobs"], "secs": time.time() - t0})
    print(f"{len(lp['tokens']):3d} tok {time.time()-t0:5.1f}s | {p[:40]!r} -> {c['text'][:60]!r}")
json.dump(res, open(a.out, "w")); print("saved", a.out)
