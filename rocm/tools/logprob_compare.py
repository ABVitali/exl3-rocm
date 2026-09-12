"""Compare two logprob dumps (same prompts, greedy): top-1 agreement over the generated tokens, mean |delta logprob|
of the chosen token, and KL(A||B) over the union of top-5 sets where both are available. usage: logprob_compare.py A.json B.json"""
import json, sys, math
A, B = (json.load(open(f)) for f in sys.argv[1:3])
agree = total = 0; dl = []; kls = []
for a, b in zip(A, B):
    n = min(len(a["tokens"]), len(b["tokens"]))
    for i in range(n):
        total += 1; agree += a["tokens"][i] == b["tokens"][i]
        if a["tokens"][i] == b["tokens"][i]: dl.append(abs(a["token_logprobs"][i] - b["token_logprobs"][i]))
        ta, tb = a["top_logprobs"][i] or {}, b["top_logprobs"][i] or {}
        keys = set(ta) & set(tb)
        if len(keys) >= 3:
            pa = {k: math.exp(ta[k]) for k in keys}; pb = {k: math.exp(tb[k]) for k in keys}
            za, zb = sum(pa.values()), sum(pb.values())
            kls.append(sum((pa[k]/za) * math.log((pa[k]/za) / (pb[k]/zb)) for k in keys))
    same_text = a["text"] == b["text"]
    print(f"{'SAME ' if same_text else 'DIFF '} {a['prompt'][:38]!r}")
print(f"\ntop-1 agreement: {agree}/{total} = {100*agree/max(total,1):.2f}%   mean |dlogprob| on agreed tokens: {sum(dl)/max(len(dl),1):.4f}   mean KL(top5): {sum(kls)/max(len(kls),1):.5f}")
