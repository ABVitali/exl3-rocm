"""Minimal standalone p2b_fused_moe call to reproduce the exit-time segfault outside pytest.
    python -X faulthandler rocm/tests/p2b_min.py [--sync-and-free]
"""
import sys, torch, vllm_exl3_c as c
dev = torch.device("cuda")
torch.manual_seed(0)
hidden, inter, experts, K = 512, 256, 2, 2
def mk(in_f, out_f):
    tr = [torch.randint(0, 65536, (in_f // 16, out_f // 16, 16 * K), dtype=torch.int32, device=dev).to(torch.short) for _ in range(experts)]
    su = [torch.sign(torch.randn(in_f, device=dev)).half() for _ in range(experts)]
    sv = [torch.sign(torch.randn(out_f, device=dev)).half() for _ in range(experts)]
    ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
    return tr, su, sv, ptr(tr), ptr(su), ptr(sv)
g = mk(hidden, inter); u = mk(hidden, inter); d = mk(inter, hidden)
ids = torch.arange(experts, dtype=torch.int32, device=dev)
rw = torch.softmax(torch.randn(experts, device=dev), 0).half()
x = torch.randn(1, hidden, dtype=torch.half, device=dev) * 0.1
out = torch.zeros(1, hidden, dtype=torch.half, device=dev)
c.p2b_fused_moe(x, out, g[3], g[4], g[5], u[3], u[4], u[5], d[3], d[4], d[5], ids, rw, K, K, K, True, inter, 0.0)
torch.cuda.synchronize()
print("p2b ok, out[0,:4] =", out[0, :4].tolist())
if "--sync-and-free" in sys.argv:
    del g, u, d, ids, rw, x, out
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    print("freed")
print("exiting")
