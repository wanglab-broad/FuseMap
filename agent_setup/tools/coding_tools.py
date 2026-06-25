### ---coding_tools.py--- ###
"""
Stateful Python REPL for the CodingAgent.

Provides a persistent execution namespace with pre-imported scientific
libraries and references to the atlas data (ad_cell, ad_gene).
"""

import io
import os
import sys
import threading
from contextlib import redirect_stdout, redirect_stderr
from langchain_core.tools import tool


# Patterns that are blocked for security.
# Layer-1 defence: simple string match, fast and easy to audit.
# These cover both the common direct usages and import-then-call patterns.
_BLOCKED_PATTERNS = [
    # Shell / process execution
    "subprocess",
    "os.system",
    "os.popen",
    "__import__",
    # File deletion
    "shutil.rmtree",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    # Binary file writers that bypass Python's open() and _safe_open:
    # h5py writes raw HDF5 directly through C extensions, skipping our hook.
    "h5py.File",
    # shutil file operations (copy / move bypass _safe_open)
    "shutil.move",
    "shutil.copy",
    "shutil.copy2",
    "shutil.copyfile",
    # Serialization writers
    "pickle.dump",
    "pickle.dumps",
    "joblib.dump",
    # pathlib write helpers
    ".write_bytes",
    ".write_text",
]


class StatefulPythonREPL:
    """A persistent Python REPL with pre-loaded scientific environment."""

    MAX_EXECUTIONS = 30
    MAX_EXECUTIONS_PER_INVOCATION = 8
    TIMEOUT_SECONDS = 120

    def __init__(self, output_dir: str = "output"):
        self._output_dir = os.path.abspath(output_dir)
        os.makedirs(self._output_dir, exist_ok=True)
        self._execution_count = 0
        self._invocation_count = 0

        # Build the persistent namespace with pre-imports
        self._namespace = {"__builtins__": __builtins__}
        self._init_namespace()

    def reset_invocation_count(self):
        """Reset the per-invocation execution counter.

        Called at the start of each CodingAgent invocation so that
        the 8-execution limit applies per user prompt, not across
        the entire Streamlit session.
        """
        self._invocation_count = 0

    # ── 资源释放：保护白名单 ──────────────────────────────────────────────────
    # 这些 key 是预加载的科学库 + atlas 引用，永远不应被清理
    _PROTECTED_KEYS = frozenset({
        "__builtins__", "__name__", "__doc__",
        "np", "pd", "sc", "ad", "plt", "sns",
        "stats", "sparse", "os",
        "ad_cell", "ad_gene", "OUTPUT_DIR", "open",
    })

    def cleanup_large_objects(self, size_threshold_mb: float = 50.0) -> list:
        """Remove user-created namespace variables that exceed *size_threshold_mb*.

        Preserves all pre-imported libraries and atlas data references listed
        in ``_PROTECTED_KEYS``. This prevents memory from accumulating across
        multiple CodingAgent invocations within a single Streamlit session.

        Note: ``sys.getsizeof`` reports the *shallow* size of each object, which
        underestimates containers (e.g. DataFrames).  A 50 MB default threshold
        is deliberately conservative so that even moderately large DataFrames
        that ``sys.getsizeof`` reports as small are still caught on the next
        cycle.  For precise tracking use the ``psutil``-based watchdog instead.

        Args:
            size_threshold_mb: Objects larger than this (in MB) are deleted.

        Returns:
            List of variable names that were removed.
        """
        import gc
        import sys

        threshold_bytes = size_threshold_mb * 1024 * 1024
        keys_to_delete = []

        for key, val in list(self._namespace.items()):
            if key in self._PROTECTED_KEYS:
                continue
            try:
                if sys.getsizeof(val) > threshold_bytes:
                    keys_to_delete.append(key)
            except Exception:
                pass  # getsizeof can fail on some objects; skip safely

        for key in keys_to_delete:
            try:
                del self._namespace[key]
            except KeyError:
                pass

        if keys_to_delete:
            gc.collect()
            try:
                from agent_setup.progress_utils import log_progress
                log_progress(
                    f"🧹 [REPL] Released {len(keys_to_delete)} large object(s) "
                    f"from namespace: {keys_to_delete}"
                )
            except Exception:
                pass  # log_progress may not be available in all contexts

        return keys_to_delete

    def _init_namespace(self):
        """Pre-import common libraries and load atlas data."""
        setup_code = """
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, sparse
import os
"""
        try:
            exec(setup_code, self._namespace)
        except ImportError:
            # Some imports may not be available; continue with what works
            pass

        # Load atlas data references
        try:
            from agent_setup.tools.brain_atlas_tools import ad_cell, ad_gene
            self._namespace["ad_cell"] = ad_cell
            self._namespace["ad_gene"] = ad_gene
        except Exception:
            pass

        # Set OUTPUT_DIR in namespace
        self._namespace["OUTPUT_DIR"] = self._output_dir

        # Install a safe open() that restricts writes to OUTPUT_DIR
        _original_open = open
        _output_dir = self._output_dir

        def _safe_open(path, mode="r", *args, **kwargs):
            if any(m in mode for m in ("w", "a", "x")):
                abs_path = os.path.abspath(path)
                if not abs_path.startswith(_output_dir):
                    raise PermissionError(
                        f"Writing is only allowed under OUTPUT_DIR ({_output_dir}). "
                        f"Got: {abs_path}"
                    )
                # Ensure parent directory exists
                parent = os.path.dirname(abs_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            return _original_open(path, mode, *args, **kwargs)

        self._namespace["open"] = _safe_open

    def _validate_code(self, code: str):
        """Two-layer security check on code before execution.

        Layer 1 – string blocklist
        --------------------------
        Rejects code containing any pattern from ``_BLOCKED_PATTERNS``.
        Fast and auditable, but can't see through variable aliasing.

        Layer 2 – AST absolute-path scan
        ---------------------------------
        Parses the code as an AST and inspects every string literal.
        If a literal looks like an absolute path (starts with ``/`` or
        a Windows drive letter e.g. ``C:\\"``) **and** is not a prefix of
        ``OUTPUT_DIR``, execution is rejected.

        Rationale: ``adata.write_h5ad('/sensitive/secret.h5ad')``
        passes the blocklist (no forbidden *function* name is present)
        but the path literal is caught here.  Conversely,
        ``adata.write_h5ad(os.path.join(OUTPUT_DIR, 'result.h5ad'))``
        contains no offending string literal and is allowed through.

        F-strings and variable concatenation cannot be analysed statically;
        those cases fall back to the runtime ``_safe_open`` guard.
        """
        import ast
        import re

        # ── Layer 1: blocklist ──────────────────────────────────────────────
        for pattern in _BLOCKED_PATTERNS:
            if pattern in code:
                raise ValueError(
                    f"Blocked: code contains forbidden pattern '{pattern}'. "
                    f"For safety, {pattern} is not allowed."
                )

        # ── Layer 2: AST absolute-path scan ────────────────────────────────
        # Absolute-path regex: starts with '/' (Unix) or 'X:\\' / 'X:/' (Windows)
        _abs_path_re = re.compile(r'^(?:/|[A-Za-z]:[/\\])')

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Let exec() surface the real error rather than masking it here.
            return

        for node in ast.walk(tree):
            # ast.Constant covers string literals in Python 3.8+
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidate = node.value
                if _abs_path_re.match(candidate):
                    # Allow paths that are inside OUTPUT_DIR
                    abs_candidate = os.path.abspath(candidate)
                    if not abs_candidate.startswith(self._output_dir):
                        raise ValueError(
                            f"Blocked: absolute path '{candidate}' is outside "
                            f"OUTPUT_DIR ({self._output_dir}). "
                            f"Use os.path.join(OUTPUT_DIR, 'filename') instead."
                        )

    def _strip_markdown_fences(self, code: str) -> str:
        """Remove ```python fences from LLM output."""
        lines = code.strip().splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    def _snapshot_files(self):
        """Take a snapshot of files under output_dir with modification times."""
        snapshot = {}
        for root, _dirs, files in os.walk(self._output_dir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    snapshot[fp] = os.path.getmtime(fp)
                except OSError:
                    pass
        return snapshot

    def _diff_files(self, before, after):
        """Return list of new or modified files."""
        changed = []
        for fp, mtime in after.items():
            if fp not in before or before[fp] < mtime:
                changed.append(fp)
        return sorted(changed)

    def execute(self, code: str) -> dict:
        """
        Execute Python code in the persistent namespace.

        Returns:
            dict with keys: stdout, stderr, success, new_files
        """
        if self._execution_count >= self.MAX_EXECUTIONS:
            return {
                "stdout": "",
                "stderr": f"Session execution limit reached ({self.MAX_EXECUTIONS}). "
                          "Please reset chat history to start a new session.",
                "success": False,
                "new_files": [],
            }

        if self._invocation_count >= self.MAX_EXECUTIONS_PER_INVOCATION:
            return {
                "stdout": "",
                "stderr": f"Per-query execution limit reached ({self.MAX_EXECUTIONS_PER_INVOCATION}). "
                          "Please simplify your request or split it into multiple messages.",
                "success": False,
                "new_files": [],
            }

        self._execution_count += 1
        self._invocation_count += 1
        code = self._strip_markdown_fences(code)

        try:
            self._validate_code(code)
        except ValueError as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "new_files": [],
            }

        before = self._snapshot_files()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result = {"exception": None}

        def _exec():
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, self._namespace)
            except Exception:
                import traceback
                stderr_buf.write(traceback.format_exc())
                result["exception"] = True

        thread = threading.Thread(target=_exec)
        thread.start()
        thread.join(timeout=self.TIMEOUT_SECONDS)

        if thread.is_alive():
            return {
                "stdout": stdout_buf.getvalue(),
                "stderr": f"Execution timed out after {self.TIMEOUT_SECONDS}s.",
                "success": False,
                "new_files": [],
            }

        success = result["exception"] is None

        # Close any open matplotlib figures to free memory
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass

        # After every successful execution, purge oversized user variables
        # so that large intermediate DataFrames / AnnData objects do not
        # accumulate across invocations within the same Streamlit session.
        if success:
            self.cleanup_large_objects()

        after = self._snapshot_files()
        new_files = self._diff_files(before, after)

        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "success": success,
            "new_files": new_files,
        }


def create_execute_python_tool(repl: StatefulPythonREPL):
    """Factory that returns a @tool-decorated function bound to the given REPL."""

    @tool
    def execute_python(code: str) -> str:
        """Execute Python code in a persistent namespace with pre-imported libraries.

        Pre-imported: numpy (np), pandas (pd), scanpy (sc), anndata (ad),
        matplotlib.pyplot (plt), seaborn (sns), scipy (stats, sparse), os.

        Namespace variables:
        - ad_cell: AnnData — cell EMBEDDINGS (64-dim), not raw expression.
          obs has: tissue_main, tissue_sub, main_STARmap, sub_STARmap,
          main_Allen, sub_Allen, x, y, global_x/y/z, ap_order, batch.
        - ad_gene: AnnData — gene EMBEDDINGS (64-dim), not raw expression.
          obs index = UPPERCASE gene symbols (e.g. 'VIP', 'MBP', 'SATB2').
        - OUTPUT_DIR: path where all output files must be saved.

        CRITICAL — Imputed gene expression = dot product of embeddings:
          imputed = ad_cell.X @ ad_gene[['VIP','MBP']].X.T  → (n_cells, 2)

        Rules:
        - Gene names must be UPPERCASE (e.g. 'VIP', 'MBP', 'SATB2')
        - Save all outputs to OUTPUT_DIR (e.g. os.path.join(OUTPUT_DIR, 'result.h5ad'))
        - Use plt.savefig() instead of plt.show()
        - Do NOT modify ad_cell or ad_gene in-place; make copies first

        Security restrictions (violations raise PermissionError / ValueError):
        - h5py.File(), pickle.dump(), joblib.dump() are BLOCKED — use anndata write instead
        - Absolute paths outside OUTPUT_DIR are BLOCKED by AST scan
          ❌  adata.write_h5ad('/sensitive/secret.h5ad')
          ✅  adata.write_h5ad(os.path.join(OUTPUT_DIR, 'result.h5ad'))
        - shutil.move(), shutil.copy(), pathlib .write_bytes()/.write_text() are BLOCKED
        - subprocess, os.system, __import__ are BLOCKED

        Args:
            code: Python code to execute. Do not wrap in markdown fences.
        """
        MAX_OUTPUT = 4000
        result = repl.execute(code)

        parts = []
        if result["success"]:
            parts.append("EXECUTION SUCCESS")
        else:
            parts.append("EXECUTION FAILED")

        if result["stdout"]:
            stdout = result["stdout"]
            if len(stdout) > MAX_OUTPUT:
                stdout = stdout[:MAX_OUTPUT] + "\n... (truncated)"
            parts.append(f"STDOUT:\n{stdout}")

        if result["stderr"]:
            stderr = result["stderr"]
            if len(stderr) > MAX_OUTPUT:
                stderr = stderr[:MAX_OUTPUT] + "\n... (truncated)"
            parts.append(f"STDERR:\n{stderr}")

        if result["new_files"]:
            parts.append("NEW/MODIFIED FILES:\n" + "\n".join(result["new_files"]))

        return "\n\n".join(parts)

    return execute_python
