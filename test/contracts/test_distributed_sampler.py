"""DistributedSampler wiring in build_dataloader (multi-GPU hang root cause).

Flight-recorder evidence (server 20260820_091919): both ranks deadlocked on
DIFFERENT collectives — rank0 in iter N+1 forward (SyncBN allgather,
SeqNum 1027960), rank1 still in iter N loss reduction (allreduce, SeqNum
1027956). Without a DistributedSampler each rank shuffles the FULL dataset
independently, so per-rank batches drive different conditional paths
(``find_unused_parameters=True``) and the collective streams desync by 4
before deadlocking. The drift window opened exactly at epoch 10 — the
``policy.epoch=[10,30,50]`` augmentation switch.

Contract: under an initialized process group, ``build_dataloader`` must
attach a DistributedSampler (shuffle honored, epoch-resettable); single
process must remain sampler-free. The engine ``create`` seam is patched so
the test asserts wiring only — no dataset on disk needed.

Run:
    pytest test/contracts/test_distributed_sampler.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import yaml_config as yc  # noqa: E402
from engine.core.yaml_config import YAMLConfig  # noqa: E402

_CFG = os.path.join(ROOT, "configs/custom_obb/synthetic_configs/synthetic_exp_002.yml")


class _FakeLoader:
    def __init__(self, **kw):
        self.dataset = list(range(100))
        self.__dict__.update(kw)


@pytest.fixture()
def _capture_create(monkeypatch):
    calls: list[dict] = []

    def fake_create(name, global_cfg, **kw):
        calls.append({"name": name, **kw})
        if not kw:
            calls[-1]["config"] = global_cfg[name]
            return list(range(100))  # dataset pre-build (no kwargs at all)
        return _FakeLoader(batch_size=kw.get("batch_size", 1), sampler=kw.get("sampler"))

    monkeypatch.setattr(yc, "create", fake_create)
    return calls


@pytest.fixture()
def _two_rank_world(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)


@pytest.fixture()
def _single_process(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: False)


def _cfg_with_shuffle(flag):
    cfg = YAMLConfig(_CFG)
    cfg.yaml_cfg["train_dataloader"]["shuffle"] = flag
    return cfg


def test_two_rank_build_attaches_distributed_sampler(
    _two_rank_world, _capture_create
):
    from torch.utils.data import DistributedSampler

    cfg = _cfg_with_shuffle(True)
    loader = cfg.build_dataloader("train_dataloader")
    assert isinstance(loader.sampler, DistributedSampler), (
        "distributed build must shard data via DistributedSampler; "
        "independent full-dataset shuffles desync the ranks' collective "
        "streams (server 20260820_091919 deadlock root cause)"
    )
    assert loader.sampler.shuffle is True
    # batch is total/world_size
    loader_calls = [c for c in _capture_create if "batch_size" in c]
    assert loader_calls[0]["batch_size"] == 3  # total 6 / world 2


def test_distributed_dataset_create_preserves_config(_two_rank_world, _capture_create):
    cfg = _cfg_with_shuffle(True)

    cfg.build_dataloader("train_dataloader")

    dataset_call = _capture_create[0]
    assert dataset_call["name"] == "__distributed_dataset__"
    assert dataset_call["config"]["classes_file"] is not None


def test_single_process_stays_sampler_free(_single_process, _capture_create):
    from torch.utils.data import DistributedSampler

    cfg = _cfg_with_shuffle(True)
    loader = cfg.build_dataloader("train_dataloader")
    assert not isinstance(loader.sampler, DistributedSampler)
    assert _capture_create[0]["batch_size"] == 6  # single create pass, total


def test_two_rank_shuffle_false_sampler_unshuffled(_two_rank_world, _capture_create):
    from torch.utils.data import DistributedSampler

    cfg = _cfg_with_shuffle(False)
    loader = cfg.build_dataloader("train_dataloader")
    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.shuffle is False


def test_create_kwargs_override_injected_dataset(monkeypatch):
    from engine.core.workspace import create
    import engine.data.dataloader as dataloader_module

    global_cfg = {
        "DataLoader": {
            "_name": "DataLoader",
            "_pymodule": dataloader_module,
        }
    }

    class _Dataset:
        pass

    configured = _Dataset()
    explicit = _Dataset()
    global_cfg["DataLoader"].update(
        {
            "_inject": ["dataset"],
            "_share": [],
            "dataset": "configured_dataset",
            "batch_size": 1,
        }
    )
    global_cfg["configured_dataset"] = configured

    monkeypatch.setattr(yc, "create", create)
    loader = create("DataLoader", global_cfg, dataset=explicit)

    assert loader.dataset is explicit
