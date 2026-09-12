"""Patch vllm/models/deepseek_v4_1/common/engram.py (image deepseekv41-flash-0909) so Engram tables are
pinned at their exact size. torch's caching host allocator rounds pinned requests to the next power of two:
each 91.6 GiB table became a 128 GiB hipHostMalloc, which fails on this host (and 2 x 128 GiB exceeds RAM).
Pageable allocation + hipHostRegister pins exactly; UVA views (get_accelerator_view_from_cpu_tensor) accept it.
Idempotent; keeps engram.py.orig."""
import pathlib, shutil, sys
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py")
s = p.read_text()
if "_pinned_exact" in s:
    print("already patched"); sys.exit(0)
shutil.copy(p, p.with_suffix(".py.orig"))
helper = '''

def _pinned_exact(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """Host tensor pinned at its exact size (pageable alloc + host register), avoiding the caching host
    allocator's power-of-two rounding that turns a 91.6 GiB Engram table into a 128 GiB request."""
    t = torch.empty(*shape, dtype=dtype, device="cpu")
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 3)  # Portable | Mapped
    if rc != cudart.cudaError.success:
        raise RuntimeError(f"cudaHostRegister failed for {tuple(shape)} {dtype}: {rc}")
    return t

'''
anchor = "class ParallelEngramEmbedding(nn.Module):"
assert anchor in s; s = s.replace(anchor, helper.strip("\n") + "\n\n\n" + anchor, 1)
old = '''        kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        self.weight = nn.Parameter(
            torch.empty(
                self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn, **kwargs
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(
                self.part_num_embeddings,
                dim // block_size,
                dtype=torch.uint8,
                **kwargs,
            ),
            requires_grad=False,
        )
'''
new = '''        if cpu_offload:
            self.weight = nn.Parameter(
                _pinned_exact(self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                _pinned_exact(self.part_num_embeddings, dim // block_size, dtype=torch.uint8),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.empty(self.part_num_embeddings, dim // block_size, dtype=torch.uint8),
                requires_grad=False,
            )
'''
assert old in s, "allocation block not found (image version differs)"
s = s.replace(old, new, 1); p.write_text(s); print("patched:", p)
