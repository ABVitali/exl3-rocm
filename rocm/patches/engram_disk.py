"""Disk-resident Engram tables for vllm/models/deepseek_v4_1/common/engram.py (image deepseekv41-flash-0909).

With VLLM_ENGRAM_DISK=1 the two ~94 GiB tables are not pinned in host RAM: rows are read on demand
from the memory-mapped safetensors shard (page cache decides residency) and dequantized by the
unchanged Triton lookup kernel on the gathered slice, so numerics are identical to the pinned path.
Per token each Engram layer needs (max_ngram-1) * n_heads = 24 rows of 256 B, so NVMe random reads are
cheap; a thread pool issues them in parallel (pread releases the GIL). Idempotent; applies on top of
engram_pinned_exact.py."""
import pathlib, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py")
s = p.read_text()
if "class EngramDiskTable" in s:
    print("already patched"); sys.exit(0)

disk_cls = '''

class EngramDiskTable:
    """fp8 rows + ue8m0 scales of one Engram table, read straight from the safetensors shard."""

    def __init__(self, model_dir: str, key_prefix: str, rows: int, dim: int, block_size: int):
        import json, os, struct
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        wname = self._resolve(index, key_prefix, "embed.weight")
        try:
            sname = self._resolve(index, key_prefix, "embed.weight_scale_inv")
        except RuntimeError:
            sname = self._resolve(index, key_prefix, "embed.scale")   # checkpoint name before vLLM's mapper renames it
        self.rows, self.dim, self.scale_cols = rows, dim, dim // block_size
        self._open = {}
        self.w_fd, self.w_off = self._locate(model_dir, index[wname], wname, rows * dim)
        self.s_fd, self.s_off = self._locate(model_dir, index[sname], sname, rows * self.scale_cols)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=int(os.environ.get("VLLM_ENGRAM_DISK_THREADS", "48")))
        logger.info("Engram table on disk: %s (%d rows x %d) read on demand from %s", wname, rows, dim, index[wname])

    @staticmethod
    def _resolve(index: dict, key_prefix: str, suffix: str) -> str:
        cands = [k for k in index if k.endswith("." + suffix) and "engram" in k]
        parts = [x for x in key_prefix.split(".") if x not in ("model", "language_model", "embed_tokens")]
        layer = next((parts[i + 1] for i, x in enumerate(parts) if x == "layers" and i + 1 < len(parts)), None)
        hits = [k for k in cands if layer is not None and f"layers.{layer}." in k]
        if len(hits) != 1:
            raise RuntimeError(f"Engram disk table: cannot resolve {key_prefix}.{suffix} in index (hits={hits[:4]})")
        return hits[0]

    def _locate(self, model_dir: str, shard: str, name: str, nbytes: int):
        import json, os, struct
        path = os.path.join(model_dir, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        begin, end = header[name]["data_offsets"]
        assert end - begin == nbytes, f"{name}: {end - begin} bytes in shard, expected {nbytes}"
        fd = os.open(path, os.O_RDONLY)
        return fd, 8 + n + begin

    def gather(self, local: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """local: int64 CPU row ids (already clamped to [0, rows)). Returns fp8 [n, dim], u8 [n, scale_cols]."""
        import os
        import numpy as np
        ids = local.numpy()
        n = ids.shape[0]
        w = np.empty((n, self.dim), dtype=np.uint8)
        sc = np.empty((n, self.scale_cols), dtype=np.uint8)
        uniq, inv = np.unique(ids, return_inverse=True)
        uw = np.empty((uniq.shape[0], self.dim), dtype=np.uint8)
        us = np.empty((uniq.shape[0], self.scale_cols), dtype=np.uint8)
        dim, scols, w_fd, w_off, s_fd, s_off = self.dim, self.scale_cols, self.w_fd, self.w_off, self.s_fd, self.s_off

        def work(lo: int, hi: int) -> None:
            for j in range(lo, hi):
                r = int(uniq[j])
                uw[j] = np.frombuffer(os.pread(w_fd, dim, w_off + r * dim), dtype=np.uint8)
                us[j] = np.frombuffer(os.pread(s_fd, scols, s_off + r * scols), dtype=np.uint8)

        m = uniq.shape[0]
        if m <= 64:
            work(0, m)
        else:
            step = max(64, (m + self.pool._max_workers - 1) // self.pool._max_workers)
            list(self.pool.map(lambda a: work(a, min(a + step, m)), range(0, m, step)))
        w[:] = uw[inv]; sc[:] = us[inv]
        return torch.from_numpy(w).view(torch.float8_e4m3fn), torch.from_numpy(sc)

'''
anchor = "class ParallelEngramEmbedding(nn.Module):"
assert anchor in s
s = s.replace(anchor, disk_cls.strip("\n") + "\n\n\n" + anchor, 1)

# constructor: disk mode -> placeholder params, no pinned allocation
old_init = '''        if cpu_offload:
            self.weight = nn.Parameter(
                _pinned_exact(self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                _pinned_exact(self.part_num_embeddings, dim // block_size, dtype=torch.uint8),
                requires_grad=False,
            )
        else:'''
new_init = '''        import os as _os
        self.disk: EngramDiskTable | None = None
        self.disk_prefix = prefix
        if cpu_offload and _os.environ.get("VLLM_ENGRAM_DISK", "1") != "0":
            # Disk-resident table: rows are read on demand, see EngramDiskTable. Placeholder params keep the
            # loader contract (the weight loader skips them).
            self.weight = nn.Parameter(torch.empty(1, dim, dtype=torch.float8_e4m3fn), requires_grad=False)
            self.weight_scale_inv = nn.Parameter(torch.empty(1, dim // block_size, dtype=torch.uint8), requires_grad=False)
            self._disk_pending = True
            from vllm.config import get_current_vllm_config
            self._disk_model_dir = get_current_vllm_config().model_config.model   # resolved while the config context is set
        elif cpu_offload:
            self.weight = nn.Parameter(
                _pinned_exact(self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                _pinned_exact(self.part_num_embeddings, dim // block_size, dtype=torch.uint8),
                requires_grad=False,
            )
        else:'''
assert old_init in s; s = s.replace(old_init, new_init, 1)
old_sig = '''        block_size: int = 32,
        cpu_offload: bool = False,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()'''
new_sig = '''        block_size: int = 32,
        cpu_offload: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()'''
assert old_sig in s; s = s.replace(old_sig, new_sig, 1)
old_attr = '''        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {
                    "weight_loader": _engram_head_shard_weight_loader,
                    "engram_vocab_start": self.vocab_start_idx,
                },
            )'''
new_attr = '''        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {
                    "weight_loader": _engram_head_shard_weight_loader,
                    "engram_vocab_start": self.vocab_start_idx,
                    "engram_disk": getattr(self, "_disk_pending", False),
                },
            )'''
assert old_attr in s; s = s.replace(old_attr, new_attr, 1)
# weight loader: skip placeholders
old_loader = '''    part_rows = param.shape[0]
    if loaded_weight.dtype == torch.float8_e8m0fnu:'''
new_loader = '''    if getattr(param, "engram_disk", False):
        return  # disk-resident table: rows are read from the shard on demand
    part_rows = param.shape[0]
    if loaded_weight.dtype == torch.float8_e8m0fnu:'''
assert old_loader in s; s = s.replace(old_loader, new_loader, 1)
# lookup: disk path (gather + same kernel on the slice with identity ids)
old_lookup = '''        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()'''
new_lookup = '''        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        if getattr(self, "_disk_pending", False):
            return self._lookup_disk(indices, out, background)
        weight, scales = self._storage()'''
assert old_lookup in s; s = s.replace(old_lookup, new_lookup, 1)
disk_lookup = '''
    def _disk_table(self) -> "EngramDiskTable":
        if self.disk is None:
            self.disk = EngramDiskTable(self._disk_model_dir, self.disk_prefix, self.num_embeddings, self.dim, self.block_size)
        return self.disk

    def _lookup_disk(self, indices: torch.Tensor, out: torch.Tensor, background: bool) -> None:
        """Same result as the pinned path: gather this rank's rows from the shard, run the lookup kernel on
        the slice with identity ids (rows outside [vocab_start, vocab_end) or padded heads stay zero)."""
        table = self._disk_table()
        T, lh = indices.shape[0], self.part_n_hash_cols
        head_ids = indices[:, self.head_start : self.head_start + lh]
        if head_ids.shape[1] < lh:  # padded heads on the last rank
            head_ids = torch.nn.functional.pad(head_ids, (0, lh - head_ids.shape[1]), value=-1)
        flat = head_ids.reshape(-1).to("cpu", dtype=torch.int64)
        owned = (flat >= self.vocab_start_idx) & (flat < self.vocab_end_idx)
        local = torch.where(owned, flat - self.vocab_start_idx, torch.zeros_like(flat))
        w_cpu, s_cpu = table.gather(local)
        dev = indices.device
        w_dev = w_cpu.to(dev, non_blocking=True)
        s_dev = s_cpu.to(dev, non_blocking=True)
        rows = T * lh
        ident = torch.where(owned, torch.arange(rows, dtype=torch.int64), torch.full((rows,), -1, dtype=torch.int64))
        ident = ident.view(T, lh).to(dev, non_blocking=True)
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        _engram_lookup_kernel[(grid,)](
            w_dev, s_dev, ident, out, 0, rows, rows, ident.stride(0), ident.stride(1),
            HEAD_START=0, LOCAL_HEADS=lh, TOTAL_HEADS=lh, DIM=self.dim, QUANT_BLOCK=self.block_size,
            BLOCK_R=16, GRID=grid,
        )
        out._engram_disk_keepalive = (w_dev, s_dev, ident)

    def lookup('''
assert "\n    def lookup(" in s; s = s.replace("\n    def lookup(", disk_lookup, 1)
# Engram passes its prefix so the table can find its checkpoint tensors
old_ctor = '''            cpu_offload=engram_config.cpu_offload if engram_config else True,
        )'''
new_ctor = '''            cpu_offload=engram_config.cpu_offload if engram_config else True,
            prefix=prefix,
        )'''
assert old_ctor in s; s = s.replace(old_ctor, new_ctor, 1)
p.write_text(s); print("patched:", p)
