"""Gradient accumulation engine tests.

锁三件事:
1. 梯度等效性 —— GT 占比加权(α_k = c_k/c_ref)下, accum=K 的累计梯度与
   整 batch 一步 backward 精确相等; 并固化"朴素 1/K 不等效"的设计依据
   (DOTA 单 patch GT 数 3~2449 极端倾斜, 1/K 会高估稀疏 micro 上百倍)。
2. 调用契约 —— optimizer.step/zero_grad/EMA/clip 只在组边界发生;
   accum=1 与原逐迭代行为一致; 尾组(数据不足 K)按实际数量结算。
3. 异常路径 —— micro 内 NaN loss 照常触发 raise_for_nonfinite_*。

CPU 即可运行:
    pytest test/engine/test_gradient_accumulation.py -v
"""

import contextlib
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import engine.solver.det_engine as det_engine
from engine.solver.det_engine import train_one_epoch


# ---------------------------------------------------------------------------
# Fakes (复用 test_det_engine_diagnostics.py 的模式)
# ---------------------------------------------------------------------------


class CoefModel(nn.Module):
    """forward 返回与参数连通的标量系数, 供 criterion 构造线性损失."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, samples, targets=None):
        return {"coef": self.w.reshape(1)}


class GTSumCriterion:
    """loss = coef · Σ(box_vals) / clamp(num_boxes, 1) —— 模拟 DEIMCriterion
    的 /num_boxes 归一化(不含 DDP all_reduce 分支, CPU 单进程语义一致)."""

    def __init__(self, nan_on_label=None):
        self.nan_on_label = nan_on_label

    def train(self):
        pass

    def __call__(self, outputs, targets, **metas):
        n_boxes = sum(len(t["labels"]) for t in targets)
        vals = sum(t["val_sum"].sum() for t in targets if len(t["labels"]) > 0)
        loss = outputs["coef"].sum() * vals / max(n_boxes, 1)
        if self.nan_on_label is not None and metas.get("step") == self.nan_on_label:
            loss = torch.tensor(float("nan"))
        return {"loss_main": loss}


class CountingOptimizer(torch.optim.SGD):
    def __init__(self, params, **kw):
        super().__init__(params, **kw)
        self.n_steps = 0
        self.n_zero = 0

    def step(self, closure=None):
        self.n_steps += 1
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.n_zero += 1
        return super().zero_grad(set_to_none)


class FakeEMA:
    def __init__(self):
        self.n = 0

    def update(self, model):
        self.n += 1


def _make_item(n_gt, val_total, seed=0):
    g = torch.Generator().manual_seed(seed)
    samples = torch.zeros(1, 3, 8, 8)
    targets = [
        {
            "labels": torch.ones(n_gt, dtype=torch.long),
            "boxes": torch.rand(n_gt, 4, generator=g),
            # 每框值 = val_total / n_gt, 保证 Σ = val_total 与 GT 数解耦
            "val_sum": torch.full((n_gt,), val_total / max(n_gt, 1)),
        }
    ]
    return samples, targets


def _run(loader, accum, max_norm=0.0, criterion=None, **kwargs):
    model = CoefModel()
    optimizer = CountingOptimizer(model.parameters(), lr=0.0)
    ema = FakeEMA()
    stats = train_one_epoch(
        False, None, model, criterion or GTSumCriterion(), loader, optimizer,
        torch.device("cpu"), 0, max_norm=max_norm, print_freq=1000,
        ema=ema, use_amp=False, gradient_accumulation_steps=accum, **kwargs,
    )
    return model, optimizer, ema, stats


# ---------------------------------------------------------------------------
# 1. 梯度等效性
# ---------------------------------------------------------------------------


def test_accum_gradient_equals_full_batch_with_gt_proportional_weights():
    """倾斜 GT 分布下 accum=2 的梯度必须与整 batch 精确相等.

    micro1: 1 框 / Σval=2; micro2: 9 框 / Σval=9。
    整 batch 梯度 = (2+9)/10 = 1.1;
    α 加权累计 = 1·(2/1)·(1/10) + 9·(9/9)·(9/10) = 0.2 + 0.9 = 1.1 ✓
    """
    full_item = _make_item(10, 11.0, seed=1)  # 10 框, Σval=11 → 梯度 1.1
    m1 = _make_item(1, 2.0, seed=2)
    m2 = _make_item(9, 9.0, seed=3)

    model_full, *_ = _run([full_item], accum=1)
    model_accum, *_ = _run([m1, m2], accum=2)

    g_full = model_full.w.grad
    g_accum = model_accum.w.grad
    assert g_full is not None and g_accum is not None
    assert torch.allclose(g_full, g_accum, rtol=1e-6, atol=1e-7), (
        f"α 加权累积梯度 {g_accum.item():.6f} != 整 batch 梯度 {g_full.item():.6f}"
    )
    assert abs(g_full.item() - 1.1) < 1e-5


def test_naive_uniform_weighting_is_not_equivalent():
    """固化设计依据: 朴素 1/K 加权在倾斜 GT 分布下不等效(不允许回退)."""
    # 直接计算朴素 1/K 的等效梯度: 0.5·(2/1) + 0.5·(9/9) = 1.5 ≠ 1.1
    full_item = _make_item(10, 11.0, seed=1)
    m1 = _make_item(1, 2.0, seed=2)
    m2 = _make_item(9, 9.0, seed=3)

    model_full, *_ = _run([full_item], accum=1)
    model_accum, *_ = _run([m1, m2], accum=2)
    naive = 0.5 * (2.0 / 1) + 0.5 * (9.0 / 9)

    assert not torch.allclose(
        model_full.w.grad, torch.tensor(naive), rtol=1e-4
    ), "1/K 均匀加权意外等于整 batch 梯度, 前提失效需重审"


def test_accum_gradient_equals_full_batch_empty_gt_micro():
    """0 GT micro: criterion 内部 clamp(num_boxes,1) 与引擎权重 clamp 一致, 仍精确等效."""
    full_a = _make_item(4, 4.0, seed=4)
    full_b = _make_item(6, 6.0, seed=5)  # 整 batch: (4+6)/10 = 1.0
    empty = (_make_item(0, 0.0, seed=6)[0],
             [{"labels": torch.zeros(0, dtype=torch.long),
               "boxes": torch.zeros(0, 4),
               "val_sum": torch.zeros(0)}])
    nonempty = _make_item(10, 10.0, seed=7)  # 累积: (0 + 10)/10 = 1.0

    model_full, *_ = _run([full_a, full_b], accum=1)
    model_accum, *_ = _run([empty, nonempty], accum=2)

    assert torch.allclose(model_full.w.grad, model_accum.w.grad, rtol=1e-6, atol=1e-7)
    assert abs(model_full.w.grad.item() - 1.0) < 1e-5


def test_tail_group_uses_actual_micro_count():
    """数据迭代数不是 K 的整数倍时, 尾组按实际 micro 数结算, 不丢数据."""
    items = [_make_item(2, 2.0 + i, seed=10 + i) for i in range(5)]  # Σval = 12
    # 整 batch 等效梯度 = Σval / Σgt = 12/10 = 1.2
    model_full, *_ = _run(
        [_make_item(10, 12.0, seed=99)], accum=1
    )
    model_tail, opt, _, _ = _run(items, accum=2)

    assert opt.n_steps == 3  # 2 + 2 + 1
    # 尾组仅 1 个 micro, 归一化基准为该组 GT 数(2), 与整 batch 10 不同 ——
    # 语义为"等效 batch 略小的尾步", 梯度不必等于整 batch; 但必须有梯度且有限
    assert model_tail.w.grad is not None
    assert torch.isfinite(model_tail.w.grad)


# ---------------------------------------------------------------------------
# 2. 调用契约
# ---------------------------------------------------------------------------


def test_boundary_actions_fire_once_per_group():
    """accum=3 × 6 iters → step/zero/EMA 各 2 次; clip 在边界发生."""
    items = [_make_item(3, 3.0, seed=20 + i) for i in range(6)]
    model, opt, ema, _ = _run(items, accum=3, max_norm=0.5)
    assert opt.n_steps == 2
    assert opt.n_zero == 2
    assert ema.n == 2


def test_accum1_matches_legacy_iteration_behavior():
    """accum=1: 每数据迭代一次 step/zero/EMA —— 与原实现一致."""
    items = [_make_item(3, 3.0, seed=30 + i) for i in range(4)]
    model, opt, ema, _ = _run(items, accum=1)
    assert opt.n_steps == 4
    assert opt.n_zero == 4
    assert ema.n == 4


def test_inspect_gradients_runs_per_micro():
    """NaN 检查按 micro 执行(每 backward 一次), 而非每 optimizer step 一次."""
    calls = {"n": 0}
    real = det_engine.inspect_gradients

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    items = [_make_item(2, 2.0, seed=40 + i) for i in range(4)]
    model = CoefModel()
    optimizer = CountingOptimizer(model.parameters(), lr=0.0)
    saved = det_engine.inspect_gradients
    det_engine.inspect_gradients = counting
    try:
        train_one_epoch(
            False, None, model, GTSumCriterion(), items, optimizer,
            torch.device("cpu"), 0, max_norm=0, print_freq=1000,
            ema=None, use_amp=False, gradient_accumulation_steps=2,
        )
    finally:
        det_engine.inspect_gradients = saved
    assert calls["n"] == 4  # 每 micro 一次, 与 optimizer step 数(2)无关


def test_no_sync_wraps_non_last_micros_only():
    """no_sync 仅包裹非末位 micro; 末位 micro 必须触发梯度同步."""
    entered = {"n": 0}

    class FakeDDP(CoefModel):
        def no_sync(self):
            outer = self

            class Ctx:
                def __enter__(self):
                    entered["n"] += 1
                    return outer

                def __exit__(self, *a):
                    return False

            return Ctx()

    items = [_make_item(2, 2.0, seed=50 + i) for i in range(4)]
    model = FakeDDP()
    optimizer = CountingOptimizer(model.parameters(), lr=0.0)
    train_one_epoch(
        False, None, model, GTSumCriterion(), items, optimizer,
        torch.device("cpu"), 0, max_norm=0, print_freq=1000,
        ema=None, use_amp=False, gradient_accumulation_steps=2,
    )
    # 2 组 × 每组 1 个非末位 micro = 2
    assert entered["n"] == 2


def test_step_cap_stops_after_n_optimizer_steps():
    items = [_make_item(2, 2.0, seed=60 + i) for i in range(6)]
    _, opt, _, stats = _run(items, accum=2, max_optimizer_steps=1)
    assert opt.n_steps == 1
    assert stats.get("_step_cap_reached") is True


def test_invalid_accum_steps_rejected():
    with pytest.raises(ValueError):
        _run([_make_item(2, 2.0, seed=1)], accum=0)
    with pytest.raises(ValueError):
        _run([_make_item(2, 2.0, seed=1)], accum=True)


# ---------------------------------------------------------------------------
# 3. 异常路径
# ---------------------------------------------------------------------------


def test_nan_loss_in_micro_still_raises():
    """组内第 2 个 micro 的 loss 为 NaN → raise_for_nonfinite 照常触发."""
    items = [_make_item(2, 2.0, seed=70), _make_item(2, 2.0, seed=71)]
    with pytest.raises(FloatingPointError):
        _run(items, accum=2, criterion=GTSumCriterion(nan_on_label=1))
