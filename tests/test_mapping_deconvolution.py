"""Numerical and workflow checks for frozen-reference bead mapping."""

import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from fusemap import api
from fusemap.postprocess import mapping
from fusemap.postprocess.mixtures import fit_mixtures


def tree_hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def reference(tmp_path):
    root, data = tmp_path / "reference", tmp_path / "reference_data"
    (root / "trained_model").mkdir(parents=True)
    data.mkdir()
    # Signature preparation must never deserialize or train the checkpoint.
    (root / "trained_model" / "FuseMap_final_model_final.pt").write_bytes(b"frozen-test-checkpoint")
    frames = []
    for filename, genes in (("ref_a.h5ad", ["g1", "g2", "g3"]),
                            ("ref_b.h5ad", ["g3", "g1"])):
        labels = np.repeat([0, 1], 30)
        signatures = np.array([[5.2, 0.2, 1.3], [0.2, 5.2, 1.3]], dtype=np.float32)
        columns = [["g1", "g2", "g3"].index(g) for g in genes]
        names = [f"cell_{i}" for i in range(len(labels))]
        x = ad.AnnData(sp.csr_matrix(signatures[labels][:, columns]),
                       obs=pd.DataFrame(index=names), var=pd.DataFrame(index=genes))
        x.write_h5ad(data / filename)
        # Reverse the embedding rows to exercise ID alignment.
        latent = np.column_stack([labels * 4.0, 4.0 - labels * 4.0]).astype(np.float32)
        emb = ad.AnnData(latent, obs=pd.DataFrame({"file_name": filename}, index=names))
        frames.append(emb[::-1].copy())
    ad.concat(frames).write_h5ad(root / "ad_celltype_embedding.h5ad")
    return root, data


def make_query(folder, name="beads.h5ad", genes=("g3", "g2", "g1")):
    folder.mkdir(exist_ok=True)
    weights = np.linspace(0, 1, 16, dtype=np.float32)
    full = np.column_stack([0.2 + 5 * weights, 5.2 - 5 * weights, np.full(16, 1.3)])
    columns = [["g1", "g2", "g3"].index(g) for g in genes]
    coordinates = np.array([(x, y) for x in range(4) for y in range(4)], dtype=float)
    x = ad.AnnData(sp.csr_matrix(full[:, columns].astype(np.float32)),
                   obs=pd.DataFrame({"x": coordinates[:, 0], "y": coordinates[:, 1]},
                                    index=[f"bead_{i}" for i in range(16)]),
                   var=pd.DataFrame(index=genes))
    x.write_h5ad(folder / name)
    return x


def test_reference_signatures_align_ids_and_pool_only_measured_genes(reference, tmp_path):
    root, data = reference
    before = tree_hashes(root), tree_hashes(data)
    bundle = mapping.build_reference_signatures(root, data, "ref_b,ref_a", n_archetypes=2)
    order = np.argsort(bundle["archetype_centers"][:, 0])
    np.testing.assert_allclose(bundle["signatures"][order], [[5.2, 0.2, 1.3], [0.2, 5.2, 1.3]], atol=1e-6)
    np.testing.assert_array_equal(bundle["genes"], ["G1", "G2", "G3"])
    assert (bundle["signature_member_counts"][:, 1] == 30).all()
    assert before == (tree_hashes(root), tree_hashes(data))
    cache = tmp_path / "signatures.npz"
    mapping.save_npz(cache, **bundle)
    loaded = mapping.load_reference_signatures(cache, root)
    np.testing.assert_array_equal(loaded["signatures"], bundle["signatures"])
    (root / "trained_model" / "FuseMap_final_model_final.pt").write_bytes(b"different-model")
    with pytest.raises(ValueError, match="different pretrained model"):
        mapping.load_reference_signatures(cache, root)


def test_shared_solver_matches_original_stage_b_objective():
    # Independent dense transcription of the original masked Stage-B objective.
    x = torch.tensor([[1.2, 2.4, 9.0], [3.8, 0.4, 6.0], [2.5, 1.5, 100.0]])
    s = torch.tensor([[4.2, 0.2, 0.0], [0.2, 3.2, 0.0]])
    mask = torch.tensor([1.0, 1.0, 0.0])
    logits = torch.zeros(3, 2, requires_grad=True)
    beta = torch.full((3,), float(np.log(np.expm1(1.0))), requires_grad=True)
    alpha = torch.full((3,), float(np.log(np.expm1(0.01))), requires_grad=True)
    optimizer = torch.optim.Adam([logits, beta, alpha], lr=0.05)
    for _ in range(10):
        optimizer.zero_grad()
        pi = torch.softmax(logits, dim=1)
        prediction = torch.nn.functional.softplus(beta) * (pi @ s) + torch.nn.functional.softplus(alpha)
        loss = (((x - prediction) ** 2) * mask).sum() / 6
        loss += 5e-4 * (-(pi * torch.log(pi + 1e-8)).sum(dim=1)).mean()
        loss.backward()
        optimizer.step()
    result = fit_mixtures(x.numpy(), s.numpy(), [True, True, False], n_steps=10, chunk_size=2, device="cpu")
    np.testing.assert_allclose(result["pi"], torch.softmax(logits, dim=1).detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(result["pi"].sum(1), 1, atol=1e-6)


def test_known_synthetic_mixtures_are_recovered():
    signatures = np.array([[5.2, 0.2, 1.3], [0.2, 5.2, 1.3]], dtype=np.float32)
    weights = np.linspace(0, 1, 100, dtype=np.float32)
    truth = np.column_stack([weights, 1 - weights])
    x = truth @ signatures
    result = fit_mixtures(x, signatures, n_steps=800, device="cpu")
    assert np.abs(result["pi"] - truth).mean() < 0.06
    assert result["reconstruction_mse"].mean() < 0.003


def test_bead_mapping_writes_mixtures_and_preserves_reference(reference, tmp_path, monkeypatch):
    root, data = reference
    cache = tmp_path / "prepared.npz"
    api.prepare_reference_signatures(root, data, "ref_a,ref_b", cache, n_archetypes=2)
    queries = tmp_path / "queries"
    make_query(queries)
    make_query(queries, "beads_second.h5ad", genes=("g1", "g3"))
    make_query(queries, "cells.h5ad")
    calls = []

    def map_only_query(inputs, args, *unused):
        assert Path(args.pretrain_model_path) == root
        assert len(inputs) == 1
        x = mapping.normalized_expression(inputs[0])
        calls.append(str(x.obs["file_name"].iloc[0]))
        # Only expensive neural adaptation is stubbed; the complete Stage-B
        # reference loading, gene matching, solver, spatial readout and writes run.
        for level in ("celltype", "tissueregion"):
            out = ad.AnnData(np.full((x.n_obs, 2), -7.0, dtype=np.float32), obs=x.obs.copy())
            out.write_h5ad(Path(args.output_save_dir) / f"ad_{level}_embedding.h5ad")

    monkeypatch.setattr(api, "spatial_map", map_only_query)
    monkeypatch.setattr(api, "spatial_integrate", lambda *a, **kw: pytest.fail("reference must not be retrained"))
    before = tree_hashes(root), tree_hashes(data), tree_hashes(queries)
    output = tmp_path / "mapped"
    api.map_to_reference(queries, output, root, bead_files="beads.h5ad,beads_second", reference_signatures_path=cache)
    assert calls == ["beads.h5ad", "beads_second.h5ad", "cells.h5ad"]
    assert before == (tree_hashes(root), tree_hashes(data), tree_hashes(queries))
    bead_output = output / "beads.h5ad"
    cell = ad.read_h5ad(bead_output / "ad_celltype_embedding.h5ad")
    tissue = ad.read_h5ad(bead_output / "ad_tissueregion_embedding.h5ad")
    original = ad.read_h5ad(bead_output / "ad_celltype_embedding_nodeconv.h5ad")
    np.testing.assert_array_equal(original.X, -7.0)
    with np.load(bead_output / "stageB_pi.npz", allow_pickle=False) as mixtures:
        np.testing.assert_allclose(cell.X, mixtures["pi"] @ mixtures["archetype_centers"], atol=1e-6)
        np.testing.assert_array_equal(mixtures["obs_names"], cell.obs_names)
        np.testing.assert_array_equal(cell.obsm["stageB_pi"], mixtures["pi"])
    x = api.read_input_folder(queries)[0]
    mapping.construct_graph([x], 1, ["delaunay"], ["ST"])
    mapping.preprocess_adj_sparse([x], 1, ["ST"])
    np.testing.assert_allclose(tissue.X, x.obsm["adj_normalized"].T @ cell.X, atol=1e-6)
    assert not (output / "cells.h5ad" / "stageB_pi.npz").exists()
    with np.load(output / "beads_second.h5ad" / "stageB_pi.npz", allow_pickle=False) as second:
        np.testing.assert_allclose(second["pi"].sum(1), 1, atol=1e-6)
        assert list(second["genes"]) == ["G1", "G3"]
    # A rerun must retain the original mapping backup and the frozen reference.
    backup_hash = tree_hashes(bead_output)["ad_celltype_embedding_nodeconv.h5ad"]
    api.map_to_reference(queries, output, root, bead_files="beads.h5ad,beads_second", reference_signatures_path=cache)
    assert backup_hash == tree_hashes(bead_output)["ad_celltype_embedding_nodeconv.h5ad"]
    assert before == (tree_hashes(root), tree_hashes(data), tree_hashes(queries))


def test_invalid_bead_requests_fail_before_training(reference, tmp_path, monkeypatch):
    root, data = reference
    queries = tmp_path / "queries"
    make_query(queries)
    monkeypatch.setattr(api, "spatial_map", lambda *a: pytest.fail("must fail before mapping"))
    with pytest.raises(ValueError, match="requires reference_data_folder_path"):
        api.map_to_reference(queries, tmp_path / "out", root, bead_files="beads")
    with pytest.raises(ValueError, match="preserve the reference"):
        api.map_to_reference(queries, root / "mapped", root)
    cache = tmp_path / "prepared.npz"
    api.prepare_reference_signatures(root, data, "ref_a", cache, n_archetypes=2)
    with pytest.raises(ValueError, match="exactly one"):
        api.map_to_reference(queries, tmp_path / "out", root, bead_files="missing", reference_signatures_path=cache)
    x = ad.read_h5ad(queries / "beads.h5ad")
    x.var_names = ["absent1", "absent2", "absent3"]
    x.write_h5ad(queries / "beads.h5ad")
    with pytest.raises(ValueError, match="No shared genes"):
        api.map_to_reference(queries, tmp_path / "out", root, bead_files="beads", reference_signatures_path=cache)


def test_expression_id_mismatch_is_rejected(reference):
    root, data = reference
    x = ad.read_h5ad(data / "ref_a.h5ad")
    x.obs_names = [f"wrong_{i}" for i in range(x.n_obs)]
    x.write_h5ad(data / "ref_a.h5ad")
    with pytest.raises(ValueError, match="observations do not match"):
        mapping.build_reference_signatures(root, data, "ref_a", n_archetypes=2)


@pytest.mark.parametrize("bad", [np.nan, -1.0])
def test_solver_rejects_invalid_expression(bad):
    with pytest.raises(ValueError):
        fit_mixtures([[bad, 1.0]], [[1.0, 2.0]], n_steps=1)


def test_cli_mapping_forwards_deconvolution_options(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from fusemap.config import parse_input_args

    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("fusemap_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "map",
                        "--input_data_folder_path", "new_data",
                        "--output_save_dir", str(tmp_path / "out"),
                        "--pretrain_model_path", "ref_model", "--bead_files", "beads",
                        "--reference_signatures_path", "reference_signatures.npz"])
    calls = []
    monkeypatch.setattr(api, "map_to_reference", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.main(parse_input_args())
    assert calls[0][0] == ("new_data", str(tmp_path / "out"), "ref_model")
    assert calls[0][1]["bead_files"] == "beads"
    assert calls[0][1]["reference_signatures_path"] == "reference_signatures.npz"
    assert calls[0][1]["reference_data_folder_path"] is None


def test_integration_stage_b_still_writes_canonical_outputs(tmp_path, monkeypatch):
    """Run the existing integration Stage-B script through its public wrapper."""
    import pickle

    root, data = tmp_path / "integrated", tmp_path / "data"
    (root / "trained_model").mkdir(parents=True)
    rng = np.random.default_rng(4)
    centers = rng.normal(size=(40, 64)).astype(np.float32)
    latent = np.repeat(centers, 40, axis=0)
    # Forty clearly separated archetypes, with finite normalized expressions.
    signatures = rng.uniform(0.1, 5, size=(40, 3)).astype(np.float32)
    x = ad.AnnData(sp.csr_matrix(np.repeat(signatures, 40, axis=0)),
                   obs=pd.DataFrame(index=[f"cell_{i}" for i in range(1600)]),
                   var=pd.DataFrame(index=["g1", "g2", "g3"]))
    bead = make_query(data, genes=("g1", "g2", "g3"))
    x.write_h5ad(data / "reference.h5ad")
    obs = pd.concat([x.obs.assign(file_name="reference.h5ad"), bead.obs.assign(file_name="beads.h5ad")])
    bead_latent = np.zeros((bead.n_obs, 64), dtype=np.float32)
    combined = ad.AnnData(np.vstack([latent, bead_latent]), obs=obs)
    for level in ("celltype", "tissueregion"):
        combined.write_h5ad(root / f"ad_{level}_embedding.h5ad")
    with (root / "latent_embeddings_all_single_final.pkl").open("wb") as handle:
        pickle.dump([latent, bead_latent], handle)
    torch.save({"gene_embedding": torch.ones(64, 3)}, root / "trained_model/FuseMap_final_model_final.pt")
    # The legacy wrapper uses env vars; isolate them from the rest of the tests.
    for name in ("STAGEB_OUT_DIR", "STAGEB_DATA_DIR", "FUSEMAP_BEAD_FILES", "FUSEMAP_SIG_REF", "STAGEB_SIG"):
        monkeypatch.setenv(name, "")
    api.deconvolve_beads(root, data, "beads", "reference", signature_mode="empirical")
    canonical = ad.read_h5ad(root / "ad_celltype_embedding.h5ad")
    np.testing.assert_array_equal(canonical.X[:1600], latent)
    with np.load(root / "stageB_pi.npz", allow_pickle=False) as mixtures:
        np.testing.assert_allclose(canonical.X[1600:], mixtures["pi"] @ mixtures["archetype_centers"], atol=1e-6)
    assert (root / "ad_celltype_embedding_nodeconv.h5ad").exists()
