"""vllm_exl3's fused MoE entry (apply_exl3_fused_moe) driving the ROCm exl3_moe binding: routing sort,
sentinel expert for non-local routes, temp buffers, num_active detection, codebook flags, act limit."""
import os, sys
os.environ["VLLM_EXL3_MOE_KERNEL"] = "exllamav3"
sys.path.insert(0, "/work/upstream-vllm-exl3/src"); sys.path.insert(0, "/work/rocm/tests")
import pytest, torch
if not torch.cuda.is_available(): pytest.skip("GPU required", allow_module_level=True)
exl3 = pytest.importorskip("vllm_exl3.exl3")
from types import SimpleNamespace as NS
from test_exl3_moe_rocm import make_experts, dense, dev

def reference(x, local, w, ex, K, limit, n_local):
    xf = x.float().cpu(); out = torch.zeros_like(xf); cache = {}
    for t in range(x.shape[0]):
        for j in range(local.shape[1]):
            e = int(local[t, j])
            if e < 0 or e >= n_local: continue          # non-local route: contributes nothing here
            if e not in cache: cache[e] = (dense(ex["g"], e, K), dense(ex["u"], e, K), dense(ex["d"], e, K))
            wg, wu, wd = cache[e]; g = xf[t] @ wg; u = xf[t] @ wu
            if limit > 0: g = g.clamp(max=limit); u = u.clamp(-limit, limit)
            out[t] += float(w[t, j]) * ((torch.nn.functional.silu(g) * u) @ wd)
    return out

@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("bsz,k,limit,ep", [(1, 2, 0.0, False), (9, 4, 7.0, False), (20, 8, 0.0, True), (300, 8, 0.0, True)])
def test_plugin_fused_entry(K, bsz, k, limit, ep):
    torch.manual_seed(1)
    n_global, n_local, H, I = 12, (8 if ep else 12), 512, 256
    ex = make_experts(n_local, H, I, K)
    inners = [{w: NS(trellis=ex[m]["tr"][e], suh=ex[m]["su"][e], svh=ex[m]["sv"][e]) for w, m in (("gate", "g"), ("up", "u"), ("down", "d"))} for e in range(n_local)]
    layer = NS(w13_trellis=ex["g"]["tr"][0], _exl3_hidden_size=H, _exl3_intermediate_local=I, _exl3_bits=K)
    exl3.build_exl3_fused_state(layer, inners)
    assert layer._exl3_fused_temps is not None and layer._exl3_k == K
    expert_map = torch.tensor([i if i < n_local else -1 for i in range(n_global)], dtype=torch.long, device=dev) if ep else None
    x = torch.randn(bsz, H, dtype=torch.half, device=dev) * 0.1
    ids = torch.stack([torch.randperm(n_global)[:k] for _ in range(bsz)]).to(dev)
    w = torch.softmax(torch.randn(bsz, k), -1).to(dev).half()
    out = exl3.apply_exl3_fused_moe(x, ids, w, layer, inners, expert_map, limit)
    torch.cuda.synchronize()
    assert getattr(layer, "_exl3_last_apply", None) != "native"
    local = exl3.map_topk_to_local(ids, n_local, expert_map).view(bsz, k).cpu()
    ref = reference(x, local, w, ex, K, limit, n_local); got = out.float().cpu()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"plugin entry K={K} bsz={bsz} k={k} limit={limit} ep={ep}: cosine={cos:.6f} rel_l2={rel:.4f}")
    assert cos > 0.999 and rel < 0.03
