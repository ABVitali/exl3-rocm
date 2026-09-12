"""Patch vllm_exl3/exl3.py so fused MoE calls pass the down-projection's own K (V4.1 packs mix K=3 gate/up
with K=4 down in some layers). Both ROCm kernels (p2b_fused_moe, exl3_moe) take K_gate/K_up/K_down."""
import pathlib, sys
p = pathlib.Path("/work/upstream-vllm-exl3/src/vllm_exl3/exl3.py"); s = p.read_text()
if "k_down_from_layer" in s: print("already patched"); sys.exit(0)
helper = '''

def k_down_from_layer(layer, k: int) -> int:
    """Down-projection K from the stacked w2 trellis (k_words = 16 * K); falls back to the layer K."""
    w2 = getattr(layer, "w2_trellis", None)
    try:
        return int(w2.shape[-1]) // 16 if w2 is not None else k
    except Exception:
        return k

'''
anchor = "def _apply_native_fused_moe("
assert anchor in s; s = s.replace(anchor, helper.strip("\n") + "\n\n\n" + anchor, 1)
old1 = '''            safe_ids[row],
            safe_weights[row],
            k,
            k,
            k,
            True,
'''
new1 = '''            safe_ids[row],
            safe_weights[row],
            k,
            k,
            k_down_from_layer(layer, k),
            True,
'''
assert old1 in s; s = s.replace(old1, new1, 1)
old2 = '''        MOE_ACT_SILU,
        k,
        k,
        k,
'''
new2 = '''        MOE_ACT_SILU,
        k,
        k,
        k_down_from_layer(layer, k),
'''
assert old2 in s, "exl3_moe args block not found"; s = s.replace(old2, new2, 1)
p.write_text(s); print("patched:", p)
