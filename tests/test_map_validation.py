"""Exercise the mapping validation/checkpoint path without a large atlas or GPU."""
import importlib
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("with_hooks", [False, True])
def test_mapping_reaches_validation_and_writes_final_checkpoint(tmp_path, monkeypatch, with_hooks):
    training = importlib.import_module("fusemap.training.train_model")
    value = lambda x: SimpleNamespace(value=x)
    config = SimpleNamespace(
        optim_kw=value("RMSprop"), learning_rate=value(0.001),
        lr_factor_pretrain=value(0.5), lr_patience_pretrain=value(2),
        epochs_run_pretrain=0, epochs_run_final=0, n_epochs=value(2), n_atlas=1,
        use_llm_gene_embedding="false", verbose=value(False),
        TRAIN_WITHOUT_EVAL=value(0), patience_limit_final=value(5),
        lr_limit_final=value(1e-5), save_dir=str(tmp_path),
        snapshot_path=str(tmp_path / "snapshot.pt"),
    )
    monkeypatch.setattr(training, "ModelType", config)
    (tmp_path / "trained_model").mkdir()
    model = torch.nn.Module()
    for name in ("discriminator_single", "discriminator_spatial", "encoder", "decoder", "scrna_seq_adj"):
        model.add_module(name, torch.nn.Linear(2, 2))

    # Replace only graph preparation and loss construction; execute the actual
    # mapping loop, optimizers, validation, scheduling, and checkpoint promotion.
    monkeypatch.setattr(training, "get_data", lambda *a: (None,) * 9)
    monkeypatch.setattr(training, "compute_dis_loss_map",
                        lambda m, *a: {"dis": m.discriminator_single.weight.square().sum()})
    validations = []

    def ae_loss(m, *args):
        loss = m.encoder.weight.square().sum()
        if not torch.is_grad_enabled():
            validations.append(float(loss))
        return {"loss_all": loss, "loss_AE_all": [loss], "dis_ae": loss}

    monkeypatch.setattr(training, "compute_ae_loss_map", ae_loss)
    synced = []

    def sync_value(v):
        synced.append(v)
        return v

    kwargs = {"dist_hooks": SimpleNamespace(sync_value=sync_value)} if with_hooks else {}
    training.map_model(
        model, [None], None, None, torch.device("cpu"), None, None, str(tmp_path),
        [[torch.ones(2, 2)]], [[torch.ones(2, 2)]], 1,
        SimpleNamespace(align_anneal=1e10, lambda_disc_single=1), **kwargs,
    )
    assert len(validations) == 1
    assert len(synced) == int(with_hooks)
    checkpoint = tmp_path / "trained_model" / "FuseMap_map_model_final.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert torch.equal(state["encoder.weight"], model.encoder.weight)
    assert not (tmp_path / "trained_model" / "FuseMap_map_model.pt").exists()
