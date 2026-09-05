"""One-call convenience API.

These wrappers make the common workflows single function calls::

    import fusemap

    fusemap.integrate("./data", "./output")
    fusemap.map_to_reference("./new_data", "./output_map", "./pretrained_model")
    fusemap.deconvolve_beads("./output", "./data",
                             bead_files="slideseq", sig_ref="starmap")

Every argument that ``main.py`` accepts on the command line is available as a
keyword argument; everything else uses the validated defaults.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import scanpy as sc

from fusemap.logger import setup_logging
from fusemap.spatial_integrate import spatial_integrate
from fusemap.spatial_map import spatial_map
from fusemap.utils import seed_all

__all__ = ["integrate", "map_to_reference", "deconvolve_beads", "read_input_folder"]


def read_input_folder(input_data_folder_path):
    """Read every ``.h5ad`` in a folder and normalize spatial coordinates.

    Coordinates are taken from ``obs['x']/obs['y']``, falling back to
    ``obs['col']/obs['row']`` and then ``obsm['spatial']``.

    Returns
    -------
    list of :class:`anndata.AnnData`, one per section.
    """
    folder = Path(input_data_folder_path)
    file_names = sorted(str(p) for p in folder.iterdir() if p.is_file() and p.suffix == ".h5ad")
    if not file_names:
        raise ValueError(f"No .h5ad files found in {folder}")

    X_input = []
    for ind, file_name_i in enumerate(file_names):
        X = sc.read_h5ad(file_name_i)
        if "x" not in X.obs.columns:
            if "col" in X.obs.columns and "row" in X.obs.columns:
                X.obs["x"] = X.obs["col"]
                X.obs["y"] = X.obs["row"]
            elif "spatial" in X.obsm:
                X.obs["x"] = X.obsm["spatial"][:, 0]
                X.obs["y"] = X.obsm["spatial"][:, 1]
            else:
                raise ValueError(
                    f"{file_name_i}: provide spatial coordinates in obs['x']/obs['y'], "
                    "obs['col']/obs['row'], or obsm['spatial']"
                )
        X.obs["name"] = f"section{ind}"
        X.obs["file_name"] = os.path.basename(file_name_i)
        X_input.append(X)
    return X_input


def _make_args(output_save_dir, keep_celltype, keep_tissueregion,
               use_llm_gene_embedding, pretrain_model_path):
    Path(output_save_dir).mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        output_save_dir=str(output_save_dir),
        keep_celltype=keep_celltype,
        keep_tissueregion=keep_tissueregion,
        use_llm_gene_embedding=use_llm_gene_embedding,
        pretrain_model_path=str(pretrain_model_path),
    )


def integrate(input_data_folder_path, output_save_dir,
              keep_celltype="", keep_tissueregion="",
              use_llm_gene_embedding="false"):
    """Integrate all spatial sections in a folder into shared embeddings.

    Equivalent to ``python main.py --mode integrate``. Training resumes
    automatically from ``snapshot.pt`` if the same command is re-run.

    Parameters
    ----------
    input_data_folder_path
        Folder with one ``.h5ad`` file per section.
    output_save_dir
        Output directory; embeddings are written here as
        ``ad_celltype_embedding.h5ad``, ``ad_tissueregion_embedding.h5ad``,
        and ``ad_gene_embedding.h5ad``.
    keep_celltype, keep_tissueregion
        Optional ``obs`` column names with existing labels to carry into the
        output embedding files.
    """
    seed_all(0)
    args = _make_args(output_save_dir, keep_celltype, keep_tissueregion,
                      use_llm_gene_embedding, "")
    setup_logging(args.output_save_dir)
    logging.info("Arguments: %s", vars(args))

    X_input = read_input_folder(input_data_folder_path)
    kneighbor = ["delaunay"] * len(X_input)
    input_identity = ["ST"] * len(X_input)
    spatial_integrate(X_input, args, kneighbor, input_identity)


def map_to_reference(input_data_folder_path, output_save_dir, pretrain_model_path,
                     keep_celltype="", keep_tissueregion="",
                     use_llm_gene_embedding="false"):
    """Map each section in a folder onto a pretrained FuseMap model.

    Equivalent to ``python main.py --mode map``. Results are written to one
    subdirectory of ``output_save_dir`` per input file.
    """
    import copy as _copy

    seed_all(0)
    X_input = read_input_folder(input_data_folder_path)
    for X in X_input:
        args_i = _make_args(
            os.path.join(str(output_save_dir), X.obs["file_name"].unique()[0]),
            keep_celltype, keep_tissueregion,
            use_llm_gene_embedding, pretrain_model_path,
        )
        setup_logging(args_i.output_save_dir)
        spatial_map([X], args_i, ["delaunay"], ["ST"])


def deconvolve_beads(output_save_dir, input_data_folder_path,
                     bead_files, sig_ref,
                     entropy_weight=None, signature_mode=None):
    """Deconvolve bead/spot-resolution datasets after an integration run.

    Each bead is decomposed into a mixture over cell archetypes learned from
    the single-cell sections (Stage-B); its cell and tissue embeddings are
    rebuilt from the mixture. Writes ``ad_celltype_embedding_stageB.h5ad``,
    ``ad_tissueregion_embedding_stageB.h5ad``, and ``stageB_pi.npz`` into
    ``output_save_dir``.

    Parameters
    ----------
    output_save_dir
        Output directory of the finished :func:`integrate` run.
    input_data_folder_path
        The data folder that was integrated.
    bead_files
        Comma-separated file-name substrings, each matching exactly one input
        file: the bead-resolution dataset(s) to deconvolve.
    sig_ref
        Comma-separated substrings selecting the single-cell reference
        dataset(s) used to build archetype signatures.
    entropy_weight
        Entropy regularization of the mixture weights (default ``5e-4``).
    signature_mode
        ``"empirical"`` (default, recommended) or ``"decode"``.
    """
    import runpy

    os.environ["STAGEB_OUT_DIR"] = str(output_save_dir)
    os.environ["STAGEB_DATA_DIR"] = str(input_data_folder_path)
    os.environ["FUSEMAP_BEAD_FILES"] = bead_files
    os.environ["FUSEMAP_SIG_REF"] = sig_ref
    if entropy_weight is not None:
        os.environ["STAGEB_ENT_W"] = str(entropy_weight)
    if signature_mode is not None:
        os.environ["STAGEB_SIG"] = str(signature_mode)

    script = Path(__file__).resolve().parent / "_stage_b_script.py"
    runpy.run_path(str(script), run_name="__main__")
