"""snapshot_download with ignore patterns (the hf CLI misparses --exclude). usage: hf_dl.py REPO DEST [ignore ...]"""
import os, sys, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download
repo, dest, ignore = sys.argv[1], sys.argv[2], sys.argv[3:]
t0 = time.time()
p = snapshot_download(repo_id=repo, local_dir=dest, ignore_patterns=ignore or None, max_workers=16)
print(f"DONE {repo} -> {p} in {time.time()-t0:.0f}s", flush=True)
