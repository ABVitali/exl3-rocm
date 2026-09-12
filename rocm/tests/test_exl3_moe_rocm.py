"""exl3_moe (ROCm batched MoE) vs an independent CPU reference (dequant_trellis + torch), and vs
p2b_fused_moe for single rows. Rows sorted by expert exactly as the plugin builds them."""
import pytest, torch
torch = pytest.importorskip("torch")
if not torch.cuda.is_available(): pytest.skip("GPU required", allow_module_level=True)
ext = pytest.importorskip("exllamav3_ext"); native = pytest.importorskip("vllm_exl3_c")
dev = torch.device("cuda"); torch.manual_seed(0)

def random_trellis(k, n, K):
    return torch.randint(0, 65536, (k // 16, n // 16, 256 * K // 16), dtype=torch.int32, device=dev).to(torch.short)

def make_experts(E, H, I, K, K_down=None):
    def mk(in_f, out_f, Kx):
        tr = [random_trellis(in_f, out_f, Kx) for _ in range(E)]
        su = [torch.sign(torch.randn(in_f, device=dev)).half() for _ in range(E)]
        sv = [torch.sign(torch.randn(out_f, device=dev)).half() for _ in range(E)]
        ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        return dict(tr=tr, su=su, sv=sv, ptr=(ptr(tr), ptr(su), ptr(sv)))
    return dict(g=mk(H, I, K), u=mk(H, I, K), d=mk(I, H, K_down or K))

def dense(m, e, K, mcg=True):  # CPU fp32 original-basis weight of expert e (mcg or mul1 codebook)
    return native.dequant_trellis(m["tr"][e].cpu(), m["su"][e].cpu(), m["sv"][e].cpu(), K, mcg).float()

def routing(bsz, E, k):
    ids = torch.stack([torch.randperm(E)[:k] for _ in range(bsz)]).to(dev)          # [bsz, k]
    w = torch.softmax(torch.randn(bsz, k), -1).to(dev).half()
    flat_e = ids.reshape(-1); flat_t = torch.arange(bsz, device=dev).repeat_interleave(k); flat_w = w.reshape(-1)
    order = flat_e.argsort(stable=True)
    expert_count = torch.zeros(E + 1, dtype=torch.long, device=dev); expert_count.scatter_add_(0, flat_e, torch.ones_like(flat_e))
    return ids, w, expert_count, flat_t[order], flat_w[order]

def reference(x, ids, w, ex, K, limit, mcg=True, K_down=None):
    xf = x.float().cpu(); out = torch.zeros_like(xf); cache = {}
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            if e not in cache: cache[e] = (dense(ex["g"], e, K, mcg), dense(ex["u"], e, K, mcg), dense(ex["d"], e, K_down or K, mcg))
            wg, wu, wd = cache[e]
            g = xf[t] @ wg; u = xf[t] @ wu
            if limit > 0: g = g.clamp(max=limit); u = u.clamp(-limit, limit)
            out[t] += float(w[t, j]) * ((torch.nn.functional.silu(g) * u) @ wd)
    return out

@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("bsz,k,limit", [(1, 2, 0.0), (5, 3, 0.0), (16, 4, 7.0), (37, 8, 0.0)])
def test_exl3_moe_matches_cpu_reference(K, bsz, k, limit):
    run_case(K, bsz, k, limit, mcg=True)

@pytest.mark.parametrize("K,K_down", [(2, 2), (3, 4), (4, 3), (4, 4)])
@pytest.mark.parametrize("bsz,k", [(1, 2), (13, 6)])
def test_exl3_moe_mul1_codebook_mixed_k(K, K_down, bsz, k):   # V4.1 packs: mul1 codebook, per-layer gu/down bits
    run_case(K, bsz, k, 0.0, mcg=False, K_down=K_down)

def run_case(K, bsz, k, limit, mcg=True, K_down=None):
    E, H, I = 12, 512, 256
    ex = make_experts(E, H, I, K, K_down)
    x = torch.randn(bsz, H, dtype=torch.half, device=dev) * 0.1
    ids, w, expert_count, token_sorted, weight_sorted = routing(bsz, E, k)
    C = ext.exl3_moe_max_concurrency(0); R = 64
    tg, tu = (torch.empty(C, R, H, dtype=torch.half, device=dev) for _ in range(2))
    ig, iu = (torch.empty(C, R, I, dtype=torch.half, device=dev) for _ in range(2))
    out = torch.zeros(bsz, H, dtype=torch.float32, device=dev)
    Kd = K_down or K
    ext.exl3_moe(x, out, expert_count, token_sorted, weight_sorted, tg, tu, ig, iu, 0, K, K, Kd,
                 *ex["g"]["ptr"], *ex["u"]["ptr"], *ex["d"]["ptr"], mcg, not mcg, mcg, not mcg, mcg, not mcg, limit, -1)
    torch.cuda.synchronize()
    ref = reference(x, ids, w, ex, K, limit, mcg, K_down)
    got = out.cpu()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"K={K}/{Kd} mcg={mcg} bsz={bsz} k={k} limit={limit}: cosine={cos:.6f} rel_l2={rel:.4f}")
    assert cos > 0.999 and rel < 0.03

def test_exl3_moe_matches_p2b_single_row():
    E, H, I, K = 8, 1024, 512, 2
    ex = make_experts(E, H, I, K)
    x = torch.randn(1, H, dtype=torch.half, device=dev) * 0.1
    ids, w, expert_count, token_sorted, weight_sorted = routing(1, E, 4)
    C = ext.exl3_moe_max_concurrency(0); R = 16
    tg, tu = (torch.empty(C, R, H, dtype=torch.half, device=dev) for _ in range(2))
    ig, iu = (torch.empty(C, R, I, dtype=torch.half, device=dev) for _ in range(2))
    out = torch.zeros(1, H, dtype=torch.float32, device=dev)
    ext.exl3_moe(x, out, expert_count, token_sorted, weight_sorted, tg, tu, ig, iu, 0, K, K, K,
                 *ex["g"]["ptr"], *ex["u"]["ptr"], *ex["d"]["ptr"], True, False, True, False, True, False, 0.0, -1)
    # p2b on the same row: pointer tables restricted to the chosen experts
    sel = ids[0].tolist()
    def tbl(m, key): return torch.tensor([m[key][e].data_ptr() for e in sel], dtype=torch.int64, device=dev)
    out2 = torch.zeros(1, H, dtype=torch.half, device=dev)
    native.p2b_fused_moe(x, out2, tbl(ex["g"], "tr"), tbl(ex["g"], "su"), tbl(ex["g"], "sv"), tbl(ex["u"], "tr"), tbl(ex["u"], "su"), tbl(ex["u"], "sv"),
                         tbl(ex["d"], "tr"), tbl(ex["d"], "su"), tbl(ex["d"], "sv"), torch.arange(len(sel), dtype=torch.int32, device=dev), w[0].contiguous(), K, K, K, True, I, 0.0)
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(out.flatten(), out2.float().flatten(), dim=0).item()
    print(f"exl3_moe vs p2b single row: cosine={cos:.6f}")
    assert cos > 0.9995
