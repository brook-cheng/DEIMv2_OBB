"""Activation (gradient) checkpointing tests for DINOv3 backbone & DEIM decoder.

锁两件事:
1. 数值一致性 —— 开/关 checkpoint 的前向输出 allclose(checkpoint 只改显存
   换时间的记账方式, 不改数学);
2. 梯度流 —— 开启后 fz(冻结骨干)下 adapter/decoder 参数有梯度、dinov3 参数
   无梯度; ft 下 dinov3 参数恢复梯度。

ViT 测试直接构造 DinoVisionTransformer(离线, 不加载权重); decoder 测试用
test_deimv2_obb_smoke.py 同款小参数 DEIMTransformer。

    pytest test/engine/test_activation_checkpoint.py -v
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.backbone.dinov3.vision_transformer import DinoVisionTransformer
from engine.deim.deim_decoder import DEIMTransformer


VIT_NAME = "dinov3_vits16plus"  # configs 内最小变体, 离线构造不下载
VIT_WEIGHTS = os.path.join(ROOT, "ckpts", "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")


def _make_decoder(gradient_checkpoint, angle_rep=0):
    torch.manual_seed(0)
    return DEIMTransformer(
        num_classes=5,
        hidden_dim=32,
        num_queries=4,
        feat_channels=[32, 32],
        feat_strides=[4, 8],
        num_levels=2,
        num_points=2,
        nhead=4,
        num_layers=3,
        dim_feedforward=64,
        dropout=0.0,
        activation="relu",
        num_denoising=0,
        learn_query_content=False,
        eval_spatial_size=(16, 16),
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=4,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="obb",
        angle_rep=angle_rep,
        gradient_checkpoint=gradient_checkpoint,
    )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle_rep", [0, 3])
def test_decoder_checkpoint_outputs_match(angle_rep):
    """开/关 checkpoint 的 decoder 前向输出一致(训练模式, dropout=0)."""
    torch.manual_seed(42)
    feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]

    model_off = _make_decoder(False, angle_rep)
    model_on = _make_decoder(True, angle_rep)
    model_on.load_state_dict(model_off.state_dict())
    model_off.train()
    model_on.train()

    with torch.no_grad():
        out_off = model_off(feats)
        out_on = model_on(feats)

    assert torch.allclose(
        out_off["pred_boxes"], out_on["pred_boxes"], rtol=1e-5, atol=1e-6
    ), "开启激活重计算后 decoder 输出发生变化"
    assert torch.allclose(
        out_off["pred_logits"], out_on["pred_logits"], rtol=1e-5, atol=1e-6
    )


def test_decoder_checkpoint_backward_flow():
    """checkpoint 开启时 decoder 参数梯度可正常回传且与关闭时一致."""
    torch.manual_seed(42)
    feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]

    grads = {}
    for flag in (False, True):
        model = _make_decoder(flag)
        model.train()
        out = model(feats)
        loss = out["pred_boxes"].sum() + out["pred_logits"].sum()
        loss.backward()
        grads[flag] = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad()

    common = set(grads[False]) & set(grads[True])
    assert len(common) > 10, "开启 checkpoint 后几乎无参数获得梯度"
    for n in list(common)[:20]:
        assert torch.allclose(
            grads[False][n], grads[True][n], rtol=1e-4, atol=1e-6
        ), f"参数 {n} 的梯度在开/关 checkpoint 间不一致"


# ---------------------------------------------------------------------------
# DINOv3 backbone
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vit():
    """随机初始化的 DINOv3 首 block 即产出 NaN(init 伪影, 真实训练由 adapter
    加载预训练权重覆盖); 与线上一致地加载本地权重, 缺文件则跳过."""
    if not os.path.exists(VIT_WEIGHTS):
        pytest.skip(f"本地预训练权重缺失: {VIT_WEIGHTS}")
    m = DinoVisionTransformer(name=VIT_NAME)
    w = torch.load(VIT_WEIGHTS, weights_only=True, map_location="cpu")
    m.load_state_dict(w if "model" not in w else w["model"], strict=True)
    m.eval()
    return m


def _vit_forward_features(vit, x):
    """走 adapter 实际使用的 get_intermediate_layers 全层路径."""
    n = vit.n_blocks
    outs = vit.get_intermediate_layers(x, n=n, return_class_token=True)
    return [p for p, _ in outs]


@pytest.mark.parametrize("size", [(2, 3, 64, 64)])
def test_vit_checkpoint_outputs_match(vit, size):
    """等价性对比在 eval 模式: train 模式下 drop_path 随机掩码随 RNG 流变化,
    两次独立前向本就不可逐元素相等(与 checkpoint 无关)."""
    torch.manual_seed(0)
    x = torch.randn(*size)

    vit.gradient_checkpointing = False
    vit.eval()
    with torch.no_grad():
        ref = _vit_forward_features(vit, x)

    vit.gradient_checkpointing = True
    with torch.no_grad():
        on = _vit_forward_features(vit, x)
    vit.gradient_checkpointing = False

    assert len(ref) == len(on) == vit.n_blocks
    for i, (a, b) in enumerate(zip(ref, on)):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), f"layer {i} 输出不一致"


def test_vit_checkpoint_gradient_flow_frozen(vit):
    """fz 场景: requires_grad_(False)(同 adapter 冻结路径)后经 checkpoint 路径,
    输入叶张量仍能回传梯度(use_reentrant=False 的关键验证点)."""
    vit.requires_grad_(False)
    vit.gradient_checkpointing = True
    vit.train()
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    outs = _vit_forward_features(vit, x)
    loss = outs[-1].sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # 冻结参数不被触碰
    for p in vit.parameters():
        assert p.grad is None, "冻结参数不应有梯度"
    vit.gradient_checkpointing = False
    vit.zero_grad(set_to_none=True)
    # 恢复可训练默认, 避免污染 module 级 fixture 的后续用例
    for p in vit.parameters():
        p.requires_grad_(True)


def test_vit_checkpoint_gradient_flow_finetune(vit):
    """ft 场景: 参数可训练时 checkpoint 路径下参数梯度正常."""
    for p in vit.parameters():
        p.requires_grad_(True)
    vit.gradient_checkpointing = True
    vit.train()
    x = torch.randn(1, 3, 64, 64)
    outs = _vit_forward_features(vit, x)
    loss = outs[-1].sum()
    loss.backward()

    block_params = list(vit.blocks.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in block_params), (
        "ft 下开启 checkpoint, block 参数未获得梯度"
    )
    vit.zero_grad(set_to_none=True)
    vit.gradient_checkpointing = False
    # 恢复默认冻结语义, 避免污染 module 级 fixture 的后续使用
    for p in vit.parameters():
        p.requires_grad_(False)
