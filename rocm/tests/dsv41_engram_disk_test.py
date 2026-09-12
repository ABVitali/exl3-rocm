"""Disk-resident Engram lookup vs the original kernel on a device-resident copy of the first M rows of the
real layer-1 table. No model load. Run inside exl3dsv41 after rocm/patches/engram_disk.py."""
import os, time, json, struct, sys
os.environ["VLLM_ENGRAM_DISK"] = "1"
import torch, numpy as np
from vllm.models.deepseek_v4_1.common import engram as E

MODEL = "/models/DeepSeek-V4.1-Flash-EXL3-3.5bpw"
index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
wname = next(k for k in index if k.endswith("engram.embed.weight") and "layers.1." in k)
prefix = wname[: -len(".embed.weight")]
cfg = json.load(open(f"{MODEL}/config.json"))["text_config"]
ROWS, DIM, BLOCK = cfg["engram_num_embeddings"][0], cfg["engram_head_dim"], 32
print("table:", wname, "rows", ROWS, "dim", DIM)

table = E.EngramDiskTable(MODEL, prefix, ROWS, DIM, BLOCK)
M = 2_000_000                                   # device-resident reference window
ref_w, ref_s = table.gather(torch.arange(M))    # sequential read of the first M rows (528 MB)
ref_w, ref_s = ref_w.cuda(), ref_s.cuda()
LH, HEADS = 24, 24

def ref_lookup(ids):                            # original kernel over the device window, vocab [0, M)
    T = ids.shape[0]; rows = T * LH
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device="cuda")
    grid = min((rows + 15) // 16, 256)
    E._engram_lookup_kernel[(grid,)](ref_w, ref_s, ids, out, 0, M, rows, ids.stride(0), ids.stride(1),
        HEAD_START=0, LOCAL_HEADS=LH, TOTAL_HEADS=HEADS, DIM=DIM, QUANT_BLOCK=BLOCK, BLOCK_R=16, GRID=grid)
    return out

class Fake:  # minimal stand-in for ParallelEngramEmbedding's disk path
    part_n_hash_cols, head_start, vocab_start_idx, vocab_end_idx, num_embeddings, dim, block_size = LH, 0, 0, M, ROWS, DIM, BLOCK
    _num_sms = 256; disk = table; disk_prefix = prefix; _disk_pending = True
    _disk_table = lambda self: table
    _lookup_disk = E.ParallelEngramEmbedding._lookup_disk

fake = Fake()
torch.manual_seed(0)
for T in (1, 8, 256):
    ids = torch.randint(0, M, (T, HEADS), dtype=torch.int64, device="cuda")
    ids[0, 0] = -1; ids[-1, -1] = M + 5                     # invalid + out-of-window rows must yield zeros
    out_ref = ref_lookup(ids)
    out_disk = torch.empty(T * LH, DIM, dtype=torch.bfloat16, device="cuda")
    fake._lookup_disk(ids, out_disk, False); torch.cuda.synchronize()
    ok = torch.equal(out_ref, out_disk)
    print(f"T={T}: bit-exact={ok} | nonzero rows {int((out_disk.abs().sum(1) > 0).sum())}/{T*LH}")
    assert ok

# timing on random rows over the whole table (cold NVMe reads)
for T in (1, 8, 64, 512):
    ids = torch.randint(0, ROWS, (T, HEADS), dtype=torch.int64, device="cuda")
    out = torch.empty(T * LH, DIM, dtype=torch.bfloat16, device="cuda")
    fake.vocab_end_idx = ROWS
    torch.cuda.synchronize(); t0 = time.time(); fake._lookup_disk(ids, out, False); torch.cuda.synchronize()
    print(f"disk lookup T={T} ({T*LH} rows): {(time.time()-t0)*1e3:.1f} ms")
print("PASS")
