"""Patch vllm_exl3/exl3.py fat-expert fallback: reconstruct the down projection with its own K (from the
trellis word count) instead of the layer's gate/up K. Mixed-K V4.1 layers (K3 gate/up, K4 down) crashed
with "packed dimension 2 is incorrect size" once prefill batches made experts exceed the fat threshold."""
import pathlib, sys
p = pathlib.Path("/work/upstream-vllm-exl3/src/vllm_exl3/exl3.py"); s = p.read_text()
old = "ext.reconstruct(w2, down.trellis, k, mcg, mul1)"
new = "ext.reconstruct(w2, down.trellis, int(down.trellis.shape[-1]) // 16, mcg, mul1)  # down K from the trellis"
if "down K from the trellis" in s: print("already patched"); sys.exit(0)
n = s.count(old); assert n >= 1, "fat down reconstruct not found"
s = s.replace(old, new); p.write_text(s); print(f"patched {n} site(s) in", p)
