"""EXL3 ROCm port: numerical oracle tests. Every kernel is diffed against an independent
reference before it is believed (wave64 bugs do not crash, they return plausible garbage).

Run on the GPU box after building both extensions:
    python -m pytest -x -q rocm/tests/test_rocm_smoke.py
"""
import math
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("GPU required (torch reports ROCm devices as cuda)", allow_module_level=True)

ext = pytest.importorskip("exllamav3_ext")
native = pytest.importorskip("vllm_exl3_c")
dev = torch.device("cuda")
torch.manual_seed(0)


def sylvester(n):
    h = torch.ones(1, 1, dtype=torch.float, device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


H128 = sylvester(128) / 128 ** 0.5


def ref_had_r_128(x, pre=None, post=None, scale=1.0):
    """x: [rows, cols], cols % 128 == 0. Row-wise 128-block Hadamard as an explicit matmul."""
    xf = x.float()
    if pre is not None:
        xf = xf * pre.float()[None, :]
    r, c = xf.shape
    y = torch.einsum("rbi,ij->rbj", xf.view(r, c // 128, 128), H128).reshape(r, c) * scale
    if post is not None:
        y = y * post.float()[None, :]
    return y


def random_trellis(k, n, K):
    return torch.randint(0, 65536, (k // 16, n // 16, 256 * K // 16), dtype=torch.int32, device=dev).to(torch.short)


@pytest.mark.parametrize("rows,cols", [(1, 128), (7, 512), (64, 4096)])
@pytest.mark.parametrize("mode", ["none", "pre", "post"])
def test_had_r_128_matches_fp32_sylvester(rows, cols, mode):
    """Upstream had_r_128 applies EITHER a pre-scale OR a post-scale (if / else-if), never both."""
    x = torch.randn(rows, cols, dtype=torch.half, device=dev)
    sc = torch.sign(torch.randn(cols, device=dev)).half()
    pre = sc if mode == "pre" else None
    post = sc if mode == "post" else None
    y = torch.empty_like(x)
    ext.had_r_128(x, y, pre, post, 1.0)
    ref = ref_had_r_128(x, pre, post)
    err = (y.float() - ref).abs().max().item()
    assert err < 2e-2 * (1 + ref.abs().max().item()), f"had_r_128[{mode}] max err {err}"


@pytest.mark.parametrize("k,n,K,mcg", [(256, 128, 2, True), (384, 256, 3, True), (256, 256, 4, True), (256, 128, 3, False)])
def test_reconstruct_matches_cpu_dequant(k, n, K, mcg):
    """GPU reconstruct (fragment swizzle + codebook) vs the plugin's host-side decoder.
    dequant_cpu applies the Hadamards and sign vectors, so feed it unit vectors and
    undo the transform with the explicit fp32 Sylvester reference."""
    trellis = random_trellis(k, n, K)
    w = torch.empty((k, n), dtype=torch.half, device=dev)
    ext.reconstruct(w, trellis, K, mcg, not mcg)
    ones_k = torch.ones(k, dtype=torch.half); ones_n = torch.ones(n, dtype=torch.half)
    w_cpu = native.dequant_trellis(trellis.cpu(), ones_k, ones_n, K, mcg).float()  # = H W_hat H (per 128-block)
    # reconstruct the same transform from the GPU W_hat
    wf = w.float().to("cpu")
    Hc = H128.cpu()
    t = torch.einsum("ij,bjn->bin", Hc, wf.view(k // 128, 128, n)).reshape(k, n)
    t = torch.einsum("bki,ij->bkj", t.view(k, n // 128, 128).transpose(0, 1), Hc).transpose(0, 1).reshape(k, n)
    err = (t - w_cpu).abs().max().item()
    assert err < 5e-2, f"reconstruct vs dequant_cpu max err {err}"


@pytest.mark.parametrize("k,n,K", [(256, 128, 3), (512, 256, 2), (384, 256, 4)])
def test_reconstruct_had_slice_matches_cpu_dequant(k, n, K):
    trellis = random_trellis(k, n, K)
    suh = torch.sign(torch.randn(k, device=dev)).half()
    svh = torch.sign(torch.randn(n, device=dev)).half()
    w = torch.empty((k, n), dtype=torch.half, device=dev)
    ext.reconstruct_had_slice(w, trellis, suh, svh, K, True, False, 0)
    w_cpu = native.dequant_trellis(trellis.cpu(), suh.cpu(), svh.cpu(), K, True).float()
    err = (w.float().cpu() - w_cpu).abs().max().item()
    assert err < 5e-2, f"reconstruct_had_slice vs dequant_cpu max err {err}"


@pytest.mark.parametrize("m,k,n,K", [(1, 256, 128, 3), (5, 512, 256, 2), (200, 1024, 512, 4)])
def test_bc_linear_exl3_run_alloc(m, k, n, K):
    trellis = random_trellis(k, n, K)
    suh = torch.sign(torch.randn(k, device=dev)).half()
    svh = torch.sign(torch.randn(n, device=dev)).half()
    x = torch.randn(m, k, dtype=torch.half, device=dev) * 0.1
    bc = ext.BC_LinearEXL3(trellis, suh, svh, K, None, True, False, torch.empty_like(x))
    y = bc.run_alloc(x, n, True)
    w_ref = native.dequant_trellis(trellis.cpu(), suh.cpu(), svh.cpu(), K, True).float()
    y_ref = x.float().cpu() @ w_ref
    cos = torch.nn.functional.cosine_similarity(y.float().cpu().flatten(), y_ref.flatten(), dim=0).item()
    assert cos > 0.999, f"BC_LinearEXL3 cosine {cos}"


@pytest.mark.parametrize("K", [2, 3, 4])
def test_p2b_fused_moe_matches_cpu_reference(K):
    """The decode path. Reference is fully independent of the GPU decode: host dequant + torch."""
    hidden, inter, experts = 1024, 512, 4
    def mk(in_f, out_f):
        tr = [random_trellis(in_f, out_f, K) for _ in range(experts)]
        su = [torch.sign(torch.randn(in_f, device=dev)).half() for _ in range(experts)]
        sv = [torch.sign(torch.randn(out_f, device=dev)).half() for _ in range(experts)]
        ptr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        return tr, su, sv, ptr(tr), ptr(su), ptr(sv)
    g = mk(hidden, inter); u = mk(hidden, inter); d = mk(inter, hidden)
    ids = torch.arange(experts, dtype=torch.int32, device=dev)
    rw = torch.softmax(torch.randn(experts, device=dev), 0).half()
    x = torch.randn(1, hidden, dtype=torch.half, device=dev) * 0.1
    out = torch.zeros(1, hidden, dtype=torch.half, device=dev)
    native.p2b_fused_moe(x, out, g[3], g[4], g[5], u[3], u[4], u[5], d[3], d[4], d[5], ids, rw, K, K, K, True, inter, 0.0)
    torch.cuda.synchronize()
    xf = x.float().cpu(); acc = torch.zeros(1, hidden)
    for e in range(experts):
        wg = native.dequant_trellis(g[0][e].cpu(), g[1][e].cpu(), g[2][e].cpu(), K, True).float()
        wu = native.dequant_trellis(u[0][e].cpu(), u[1][e].cpu(), u[2][e].cpu(), K, True).float()
        wd = native.dequant_trellis(d[0][e].cpu(), d[1][e].cpu(), d[2][e].cpu(), K, True).float()
        gg = xf @ wg; uu = xf @ wu
        h = torch.nn.functional.silu(gg) * uu
        acc += rw[e].float().cpu() * (h @ wd)
    cos = torch.nn.functional.cosine_similarity(out.float().cpu().flatten(), acc.flatten(), dim=0).item()
    rel = ((out.float().cpu() - acc).norm() / acc.norm()).item()
    print(f"K={K} cosine={cos:.6f} rel_l2={rel:.4f}")
    assert cos > 0.999 and rel < 0.05, f"p2b_fused_moe K={K}: cosine {cos}, rel {rel}"
