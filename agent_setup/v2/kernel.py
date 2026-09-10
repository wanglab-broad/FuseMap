"""Persistent Jupyter kernel session for the FuseMap Agent (CodeAct executor).

The agent harness (modern LangChain env) drives a kernel running in the PINNED
scientific environment (FuseMap_952261_env), so the agent writes code against
the exact scanpy/torch/dgl stack the pipelines were validated on. Variables
persist across turns; the kernel boots with the FuseMap arsenal pre-imported
(one-call API, molCCF paths, atlas lazy-loaders).

Output discipline (context-rot control):
- combined text output longer than ``PREVIEW_CHARS`` is written to an artifact
  file and replaced by a head/tail preview plus the file path (handle pattern)
- every matplotlib/display image is saved to ``artifacts/`` and returned as a
  path, never inlined
"""

import os
import queue
import re
import time
from pathlib import Path

FUSEMAP_PY = "/ewsc/yhe/miniconda3/envs/FuseMap_952261_env/bin/python"
REPO = str(Path(__file__).resolve().parents[2])
MOLCCF_DIR = os.path.join(REPO, "molCCF")
ATLAS_DIR = os.path.join(REPO, "agent_setup", "atlas_data")
PREVIEW_CHARS = 2000

_BOOTSTRAP = r'''
import warnings, os, sys
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
sys.path.insert(0, {repo!r})
os.chdir({workdir!r})

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt

import fusemap  # one-call API: integrate / map_to_reference / deconvolve_beads / transfer_labels
from fusemap import metrics as fm_metrics, pl as fm_pl

MOLCCF_DIR = {molccf!r}      # pretrained molCCF model weights (for fusemap.map_to_reference)
ATLAS_DIR = {atlas!r}        # molCCF atlas files (ad_cell.h5ad 10.4M cells, ad_gene.h5ad, lookups)

_ATLAS_CACHE = {{}}
def load_atlas(which="cell"):
    """Lazy-load the molCCF atlas ('cell' -> 10.4M-cell AnnData with region/type
    labels + 64-dim latents; 'gene' -> universal gene embedding). Cached."""
    if which not in _ATLAS_CACHE:
        f = os.path.join(ATLAS_DIR, f"ad_{{which}}.h5ad")
        _ATLAS_CACHE[which] = sc.read_h5ad(f)
    return _ATLAS_CACHE[which]

def atlas_lookup(kind="region"):
    """Region / cell-type vocabularies of the atlas ('region' or 'type')."""
    return pd.read_csv(os.path.join(ATLAS_DIR, f"{{kind}}_lookup.csv"))

def impute_gene_on(ad_embed, genes, gene_embedding=None):
    """Impute expression of genes for the cells in ad_embed (latents in .X)
    via the universal gene embedding: E_hat = Z_c @ Z_g^T. Returns a DataFrame."""
    ge = gene_embedding if gene_embedding is not None else load_atlas("gene")
    genes = [g.upper() for g in ([genes] if isinstance(genes, str) else genes)]
    gv = np.asarray(ge[genes].X)
    return pd.DataFrame(np.asarray(ad_embed.X) @ gv.T, columns=genes,
                        index=ad_embed.obs_names)

print("[kernel] FuseMap arsenal ready: fusemap one-call API, molCCF weights at MOLCCF_DIR, "
      "atlas via load_atlas()/atlas_lookup()/impute_gene_on().")
'''


class KernelSession:
    """One persistent kernel per chat thread."""

    def __init__(self, thread_id, workspace_root=None):
        from jupyter_client import KernelManager

        root = Path(workspace_root or os.path.join(REPO, "agent_setup", "v2", "workspace"))
        self.workdir = root / str(thread_id)
        self.artifacts = self.workdir / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self._counter = 0

        self.km = KernelManager(kernel_name="fusemap-kernel")
        self.km.start_kernel(cwd=str(self.workdir))
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=120)
        boot = _BOOTSTRAP.format(repo=REPO, workdir=str(self.workdir),
                                 molccf=MOLCCF_DIR, atlas=ATLAS_DIR)
        self.execute(boot, timeout=300)

    def execute(self, code, timeout=1800):
        """Run a code block; return dict(text, images, error, artifact)."""
        msg_id = self.kc.execute(code)
        chunks, images, error = [], [], None
        deadline = time.time() + timeout
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=max(1, deadline - time.time()))
            except queue.Empty:
                error = f"TIMEOUT after {timeout}s (kernel interrupted - variables preserved)"
                self.km.interrupt_kernel()
                break
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "stream":
                chunks.append(c["text"])
            elif t in ("execute_result", "display_data"):
                data = c.get("data", {})
                if "image/png" in data:
                    self._counter += 1
                    p = self.artifacts / f"fig_{self._counter:03d}.png"
                    import base64
                    p.write_bytes(base64.b64decode(data["image/png"]))
                    images.append(str(p))
                elif "text/plain" in data:
                    chunks.append(data["text/plain"] + "\n")
            elif t == "error":
                error = "\n".join(c.get("traceback", []))
                error = re.sub(r"\x1b\[[0-9;]*m", "", error)
            elif t == "status" and c.get("execution_state") == "idle":
                break

        text = "".join(chunks)
        artifact = None
        if len(text) > PREVIEW_CHARS:
            self._counter += 1
            artifact = self.artifacts / f"out_{self._counter:03d}.txt"
            artifact.write_text(text)
            text = (text[: PREVIEW_CHARS // 2] + f"\n... [{len(text)} chars total, "
                    f"full output: {artifact}] ...\n" + text[-PREVIEW_CHARS // 2:])
        return {"text": text, "images": images, "error": error,
                "artifact": str(artifact) if artifact else None}

    def shutdown(self):
        try:
            self.kc.stop_channels()
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass


_SESSIONS = {}

def get_session(thread_id):
    if thread_id not in _SESSIONS:
        _SESSIONS[thread_id] = KernelSession(thread_id)
    return _SESSIONS[thread_id]
