"""Patch vllm_exl3/exl3.py loader: copy each loaded EXL3 tensor into a fresh anonymous buffer before the
device copy. On ROCm, an H2D copy straight from safetensors' private file mapping pins the pages with write
intent, breaking copy-on-write: every touched shard page becomes private anonymous memory that stays
resident for the mapping's lifetime (~6.5 GB per shard, ~300 GB for the V4.1 pack -> host OOM)."""
import pathlib, sys
p = pathlib.Path("/work/upstream-vllm-exl3/src/vllm_exl3/exl3.py"); s = p.read_text()
old = "loaded = loaded_weight.detach().contiguous()"
if "clone()  # anon copy" in s: print("already patched"); sys.exit(0)
n = s.count(old); assert n >= 1, "loader line not found"
s = s.replace(old, "loaded = loaded_weight.detach().clone()  # anon copy: keep the file mapping's pages clean (see rocm/patches)")
p.write_text(s); print(f"patched {n} loader sites in", p)
