"""Per-layer MoE timing: batched exl3_moe (one call) vs per-row p2b_fused_moe loop (the plugin's
native path). GLM-5.3-Flash-like layer shape unless overridden. Usage: bench_moe_kernel.py [E] [rows...]"""
import os, sys, time, torch
import exllamav3_ext as ext, vllm_exl3_c as native
dev = torch.device("cuda"); torch.manual_seed(0)
H, I, K, TOPK = 4096, 2048, 2, 8
E = int(sys.argv[1]) if len(sys.argv) > 1 else 32
ROWS = [int(r) for r in sys.argv[2:]] or [1, 2, 4, 8, 16, 32]

def trellis(k, n): return torch.randint(0, 65536, (k // 16, n // 16, 256 * K // 16), dtype=torch.int32, device=dev).to(torch.short)
def mk(in_f, out_f):
    tr = [trellis(in_f, out_f) for _ in range(E)]
    su = [torch.sign(torch.randn(in_f, device=dev)).half() for _ in range(E)]
    sv = [torch.sign(torch.randn(out_f, device=dev)).half() for _ in range(E)]
    ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
    return dict(tr=tr, su=su, sv=sv, ptr=(ptr(tr), ptr(su), ptr(sv)))
g, u, d = mk(H, I), mk(H, I), mk(I, H)
C, R = ext.exl3_moe_max_concurrency(0), 2048
tg, tu = (torch.empty(C, R, H, dtype=torch.half, device=dev) for _ in range(2))
ig, iu = (torch.empty(C, R, I, dtype=torch.half, device=dev) for _ in range(2))

def timeit(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3

print(f"E={E} H={H} I={I} K={K} topk={TOPK} | {torch.cuda.get_device_name(0)}")
print(f"{'rows':>5} {'active':>6} {'batched ms':>11} {'per-row ms':>11} {'speedup':>8}")
for rows in ROWS:
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(rows)])
    w = torch.softmax(torch.randn(rows, TOPK, device=dev), -1).half()
    flat_e = ids.reshape(-1); flat_t = torch.arange(rows, device=dev).repeat_interleave(TOPK)
    order = flat_e.argsort(stable=True)
    ec = torch.zeros(E + 1, dtype=torch.long, device=dev); ec.scatter_add_(0, flat_e, torch.ones_like(flat_e))
    ts, ws = flat_t[order].contiguous(), w.reshape(-1)[order].contiguous()
    x = torch.randn(rows, H, dtype=torch.half, device=dev) * 0.1
    out = torch.zeros(rows, H, dtype=torch.float32, device=dev)
    def batched():
        ext.exl3_moe(x, out, ec, ts, ws, tg, tu, ig, iu, 0, K, K, K, *g["ptr"], *u["ptr"], *d["ptr"], True, False, True, False, True, False, 0.0, -1)
    tbl = [[torch.tensor([m[key][int(e)].data_ptr() for e in ids[r]], dtype=torch.int64, device=dev) for m in (g, u, d) for key in ("tr", "su", "sv")] for r in range(rows)]
    out2 = torch.zeros(1, H, dtype=torch.half, device=dev); sel = torch.arange(TOPK, dtype=torch.int32, device=dev)
    def per_row():
        for r in range(rows):
            native.p2b_fused_moe(x[r:r+1], out2, *tbl[r], sel, w[r].contiguous(), K, K, K, True, I, 0.0)
    b, p = timeit(batched), timeit(per_row)
    print(f"{rows:>5} {int((ec[:E] > 0).sum()):>6} {b:>11.2f} {p:>11.2f} {p / b:>7.2f}x")
    if os.environ.get("BENCH_PROFILE"):   # per-kernel GPU time, 10 iterations of each path
        from torch.profiler import profile, ProfilerActivity
        for name, fn in (("batched", batched), ("per-row", per_row)):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10): fn()
                torch.cuda.synchronize()
            print(f"--- {name} rows={rows}: kernel GPU time per iteration")
            for ev in sorted(prof.key_averages(), key=lambda e: -e.device_time_total)[:8]:
                if ev.device_time_total > 0: print(f"  {ev.key[:70]:<70} calls/iter {ev.count/10:>6.1f}  us/iter {ev.device_time_total/10:>9.1f}")
