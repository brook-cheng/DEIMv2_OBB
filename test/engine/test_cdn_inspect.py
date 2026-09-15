"""CDN inspection test — verifies get_contrastive_denoising_training_group outputs.

Usage:
    python -m pytest test/test_cdn_inspect.py::test_cdn_generation -v
"""

import sys
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from engine.deim.denoising import get_contrastive_denoising_training_group


def test_cdn_generation():
    """Verify CDN query generation produces correct shapes, masks, metadata and edge-case handling."""

    num_classes = 10
    hidden_dim = num_classes  # so input_query_logits shape is (B, D, num_classes)
    num_denoising = 100
    num_queries = 200  # → tgt_size = 200+200 = 400 → attn_mask (400,400)

    # ── mock class embedding ──
    class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

    # ── GT boxes in cxcywhθ format, θ ∈ [0, π] ──
    # batch of 2 images, 2 boxes each: classes 3 and 7
    targets = [
        {
            "labels": torch.tensor([3, 7]),
            "boxes": torch.tensor(
                [
                    [0.3, 0.4, 0.20, 0.30, 0.5],
                    [0.6, 0.6, 0.15, 0.25, 1.2],
                ]
            ),
        },
        {
            "labels": torch.tensor([3, 7]),
            "boxes": torch.tensor(
                [
                    [0.2, 0.3, 0.10, 0.20, 2.0],
                    [0.7, 0.5, 0.20, 0.15, 0.8],
                ]
            ),
        },
    ]

    # ── main invocation ──
    input_query_logits, input_query_bbox_unact, attn_mask, dn_meta = (
        get_contrastive_denoising_training_group(
            targets=targets,
            num_classes=num_classes,
            num_queries=num_queries,
            class_embed=class_embed,
            num_denoising=num_denoising,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            box_mode="obb",
        )
    )

    # ── computed intermediates (for clarity) ──
    # max_gt_num = 2, num_group = 100//2 = 50
    # actual_denoising = 2 * 2 * 50 = 200
    actual_denoising = 200

    # ── shape assertions ──
    assert input_query_logits.shape == (
        2,
        actual_denoising,
        num_classes,
    ), f"Expected (2, {actual_denoising}, {num_classes}), got {input_query_logits.shape}"
    assert input_query_bbox_unact.shape == (
        2,
        actual_denoising,
        5,
    ), f"Expected (2, {actual_denoising}, 5), got {input_query_bbox_unact.shape}"
    assert attn_mask.shape == (
        actual_denoising + num_queries,
        actual_denoising + num_queries,
    ), (
        f"Expected ({actual_denoising + num_queries}, {actual_denoising + num_queries}), "
        f"got {attn_mask.shape}"
    )
    assert attn_mask.dtype == torch.bool, f"Expected bool, got {attn_mask.dtype}"

    # ── metadata assertions ──
    assert dn_meta is not None, "dn_meta should not be None"
    assert dn_meta["dn_num_group"] >= 1, f"dn_num_group={dn_meta['dn_num_group']} < 1"
    assert dn_meta["dn_num_split"] == [
        actual_denoising,
        num_queries,
    ], f"Expected [{actual_denoising}, {num_queries}], got {dn_meta['dn_num_split']}"
    assert (
        len(dn_meta["dn_positive_idx"]) == 2
    ), f"Expected 2 positive index groups, got {len(dn_meta['dn_positive_idx'])}"

    # ── value-domain assertions ──
    assert torch.isfinite(
        input_query_logits
    ).all(), "input_query_logits contains NaN/Inf"
    assert torch.isfinite(
        input_query_bbox_unact
    ).all(), "input_query_bbox_unact contains NaN/Inf"

    # ── edge case: num_denoising=0 → all None ──
    r_zero = get_contrastive_denoising_training_group(
        targets,
        num_classes,
        num_queries,
        class_embed,
        num_denoising=0,
        box_mode="obb",
    )
    assert all(
        v is None for v in r_zero
    ), f"Expected all-None for num_denoising=0, got {[type(v).__name__ for v in r_zero]}"

    # ── edge case: empty targets → all None ──
    empty_targets = [
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 5)},
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 5)},
    ]
    r_empty = get_contrastive_denoising_training_group(
        empty_targets,
        num_classes,
        num_queries,
        class_embed,
        num_denoising=100,
        box_mode="obb",
    )
    assert all(
        v is None for v in r_empty
    ), f"Expected all-None for empty targets, got {[type(v).__name__ for v in r_empty]}"


# ═══════════════════════════════════════════════════════════════════════
# GPU-backed tests (shared setup)
# ═══════════════════════════════════════════════════════════════════════


def _cdn_gpu_setup():
    """Shared GPU setup for CDN visualization and gradient tests.

    Returns (model, criterion, postprocessor, val_dataloader, device).
    Calls pytest.skip if GPU memory < 5 GB.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    free_mem_gb = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated(0)
    ) / 1e9
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if free_mem_gb < 5.0:
        pytest.skip(
            f"GPU memory insufficient: {free_mem_gb:.1f} GB free "
            f"of {total_mem_gb:.1f} GB total (need ≥ 5 GB)"
        )
    device = torch.device("cuda")

    from engine.core import YAMLConfig

    cfg = YAMLConfig("configs/custom_obb/synthetic_configs/synthetic_exp_050.yml")

    ckpt = torch.load(
        "outputs/synthetic_exp_050/last.pth",
        map_location="cpu",
        weights_only=False,
    )

    model = cfg.model
    criterion = cfg.criterion
    postprocessor = cfg.postprocessor

    model.load_state_dict(ckpt["model"])
    criterion.load_state_dict(ckpt["criterion"])
    postprocessor.load_state_dict(ckpt["postprocessor"])

    model = model.cuda()
    criterion = criterion.cuda()

    # Set actual decoder query count (differs from test num_queries param)
    model.decoder.num_queries = 300

    val_dataloader = cfg.val_dataloader

    return model, criterion, postprocessor, val_dataloader, device


def _get_batch(val_dataloader, device):
    """Fetch one batch and move to device."""
    samples, targets = next(iter(val_dataloader))
    samples = samples.to(device)
    for t in targets:
        for k in t:
            if isinstance(t[k], torch.Tensor):
                t[k] = t[k].to(device)
    return samples, targets


def _denormalize_obb_boxes(boxes_cxcywhtheta, orig_size):
    """Convert normalized OBB boxes [cx,cy,w,h,theta] to pixel coordinates.

    boxes_cxcywhtheta: (N, 5) tensor in [0,1] normalized coords
    orig_size: (2,) tensor [width, height]

    Returns: (N, 4) tensor of corner points as (x1,y1,x2,y2) approx rect
    or (N, 5) tensor in pixel coordinates.
    """
    w_img, h_img = orig_size[0].item(), orig_size[1].item()
    boxes_px = boxes_cxcywhtheta.clone()
    boxes_px[:, 0] *= w_img
    boxes_px[:, 1] *= h_img
    boxes_px[:, 2] *= w_img
    boxes_px[:, 3] *= h_img
    # theta stays in radians [0, π]
    return boxes_px


def _draw_obb_boxes(ax, boxes_px, color, labels=None, scores=None, linewidth=1.5):
    """Draw OBB boxes as rotated rectangles on a matplotlib axis.

    boxes_px: (N, 5) — [cx, cy, w, h, theta] in pixel coords, theta in radians.
    """
    import matplotlib.patches as patches
    import numpy as np

    for i in range(len(boxes_px)):
        cx, cy, w, h, theta = boxes_px[i].tolist()
        rect = patches.Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            angle=np.degrees(theta),
            rotation_point="center",
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        if labels is not None:
            lbl = str(int(labels[i].item())) if labels.dim() > 0 else str(labels)
            ax.text(cx, cy - h / 2 - 2, lbl, color=color, fontsize=6, ha="center")
        if scores is not None:
            sc = f"{scores[i].item():.2f}" if scores.dim() > 0 else f"{scores:.2f}"
            ax.text(cx, cy + h / 2 + 4, sc, color=color, fontsize=5, ha="center")


def _draw_obb_ellipses(ax, boxes_px, color, labels=None, linewidth=1.5):
    """Draw OBB boxes as Gaussian covariance ellipses (1-sigma contour).

    boxes_px: (N, 5) — [cx, cy, w, h, theta] in pixel coords, theta in radians.
    labels:   (N,) optional — label IDs drawn next to each ellipse center.
    """
    import numpy as np
    from matplotlib.patches import Ellipse

    if boxes_px.numel() == 0:
        return
    for i in range(boxes_px.shape[0]):
        cx, cy, w, h, theta = boxes_px[i].tolist()
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        D = np.diag([(w / 2) ** 2, (h / 2) ** 2])
        sigma = R @ D @ R.T
        eigvals, eigvecs = np.linalg.eigh(sigma)
        width = 2 * np.sqrt(max(eigvals[0], 1e-6))
        height = 2 * np.sqrt(max(eigvals[1], 1e-6))
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(
            (cx, cy),
            width,
            height,
            angle=angle,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            alpha=0.6,
        )
        ax.add_patch(ell)
        if labels is not None:
            lbl = str(int(labels[i].item())) if labels.dim() > 0 else str(labels)
            ax.text(
                cx,
                cy,
                lbl,
                color=color,
                fontsize=7,
                ha="center",
                va="center",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, lw=0.5
                ),
            )


def test_cdn_visualization():
    """Verify CDN query visualization: 4-panel per image showing GT / normal / CDN+ / CDN-.

    Asserts output PNG files exist, have size > 10 KB, and start with PNG magic bytes.
    """
    model, criterion, postprocessor, val_dataloader, device = _cdn_gpu_setup()
    samples, targets = _get_batch(val_dataloader, device)

    output_dir = "test/outputs/cdn_inspect"
    os.makedirs(output_dir, exist_ok=True)

    # ── forward (CDN only active in training mode) ──
    model.train()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    # Verify CDN keys exist
    assert "dn_meta" in outputs, "outputs missing dn_meta"
    assert "dn_outputs" in outputs, "outputs missing dn_outputs"

    dn_meta = outputs["dn_meta"]

    # ── normal predictions (already split by forward pass) ──
    normal_bboxes = outputs["pred_boxes"]  # (bs, num_queries, 5)
    normal_logits = outputs["pred_logits"]  # (bs, num_queries, num_classes)

    # ── CDN predictions from dn_outputs ──
    dn_output_last = outputs["dn_outputs"][-1]
    dn_bboxes = dn_output_last["pred_boxes"]  # (bs, dn_n, 5)
    dn_logits = dn_output_last["pred_logits"]  # (bs, dn_n, num_classes)

    dn_positive_idx = dn_meta["dn_positive_idx"]  # list of tensors per image

    batch_size = samples.shape[0]
    num_vis = min(3, batch_size)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ── per-image denorm helpers ──
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    for img_idx in range(num_vis):
        orig_size = targets[img_idx]["orig_size"]  # (2,) [w, h]

        # Denormalize image
        img_tensor = samples[img_idx] * std + mean  # (3, H, W)
        img_tensor = img_tensor.clamp(0, 1)
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

        # ── GT boxes ──
        gt_boxes = targets[img_idx]["boxes"]  # (num_gt, 5) normalized
        gt_labels = targets[img_idx]["labels"]
        gt_boxes_px = _denormalize_obb_boxes(gt_boxes, orig_size)

        # ── Normal decoder top-10 ──
        nb = normal_bboxes[img_idx]  # (num_queries, 5)
        nl = normal_logits[img_idx]  # (num_queries, num_classes)
        scores, pred_cls = nl.sigmoid().max(dim=-1)
        topk = min(10, len(scores))
        _, topk_idx = scores.topk(topk)
        nb_topk = nb[topk_idx]
        nb_topk_px = _denormalize_obb_boxes(nb_topk, orig_size)
        scores_topk = scores[topk_idx]

        # ── CDN positive/negative split ──
        pos_idx = dn_positive_idx[img_idx]  # indices of positive CDN queries
        all_dn_idx = set(range(dn_bboxes.shape[1]))
        pos_set = set(pos_idx.tolist())
        neg_idx_list = sorted(all_dn_idx - pos_set)
        neg_idx = torch.tensor(neg_idx_list, device=device)

        dn_pos_boxes_px = _denormalize_obb_boxes(dn_bboxes[img_idx][pos_idx], orig_size)
        dn_pos_scores = dn_logits[img_idx][pos_idx].sigmoid().max(dim=-1).values

        if len(neg_idx) > 0:
            dn_neg_boxes_px = _denormalize_obb_boxes(
                dn_bboxes[img_idx][neg_idx], orig_size
            )
            dn_neg_scores = dn_logits[img_idx][neg_idx].sigmoid().max(dim=-1).values
        else:
            dn_neg_boxes_px = torch.zeros(0, 5)
            dn_neg_scores = torch.zeros(0)

        # ── 4-panel figure ──
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        # Panel 1: original + GT
        axes[0].imshow(img_np)
        axes[0].set_title(f"Image {img_idx}: GT boxes (green)")
        _draw_obb_boxes(axes[0], gt_boxes_px, "green", labels=gt_labels)
        axes[0].axis("off")

        # Panel 2: normal decoder top-10
        axes[1].imshow(img_np)
        axes[1].set_title(f"Image {img_idx}: Normal top-10 (blue)")
        _draw_obb_boxes(axes[1], nb_topk_px, "blue", scores=scores_topk)
        axes[1].axis("off")

        # Panel 3: CDN positive queries
        axes[2].imshow(img_np)
        axes[2].set_title(f"Image {img_idx}: CDN positive (red, n={len(pos_idx)})")
        _draw_obb_boxes(axes[2], dn_pos_boxes_px, "red", scores=dn_pos_scores)
        axes[2].axis("off")

        # Panel 4: CDN negative queries
        axes[3].imshow(img_np)
        axes[3].set_title(f"Image {img_idx}: CDN negative (orange, n={len(neg_idx)})")
        _draw_obb_boxes(axes[3], dn_neg_boxes_px, "orange", scores=dn_neg_scores)
        axes[3].axis("off")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"cdn_viz_sample{img_idx}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # ── Assert file output ──
        assert os.path.exists(out_path), f"Missing output: {out_path}"
        fsize = os.path.getsize(out_path)
        assert fsize > 10240, f"File too small: {fsize} bytes"
        with open(out_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"\x89PNG", f"Bad PNG magic: {magic!r}"


def test_cdn_gradient():
    """Verify CDN loss gradients flow through decoder attention weights.

    Asserts CDN loss ratio is in [0.05, 0.95] and gradient norms > 1e-6.
    """
    model, criterion, postprocessor, val_dataloader, device = _cdn_gpu_setup()
    samples, targets = _get_batch(val_dataloader, device)

    # ── forward ──
    model.train()
    outputs = model(samples, targets=targets)

    # ── compute losses ──
    loss_dict = criterion(outputs, targets)

    # ── CDN loss ratio ──
    dn_loss = sum(v.item() for k, v in loss_dict.items() if "_dn_" in k)
    total_loss = sum(v.item() for v in loss_dict.values())
    ratio = dn_loss / total_loss if total_loss > 0 else 0.0
    print(f"dn_loss={dn_loss:.2f}, total_loss={total_loss:.2f}, ratio={ratio:.4f}")
    assert 0.05 < ratio < 0.95, f"dn/total ratio {ratio:.4f} outside [0.05, 0.95]"

    # ── grad checkpoint param ──
    grad_param_name = "decoder.decoder.layers.0.self_attn.in_proj_weight"

    # ── Gradient from non-CDN loss (loss_mal) ──
    loss_mal = loss_dict.get("loss_mal")
    if loss_mal is None:
        # fallback: pick first non-dn, non-aux key
        for k in loss_dict:
            if "_dn_" not in k and "_aux_" not in k and "_enc_" not in k:
                loss_mal = loss_dict[k]
                break

    assert loss_mal is not None, "No non-CDN loss found in loss_dict"

    model.zero_grad()
    loss_mal.backward(retain_graph=True)

    grad_tensor = dict(model.named_parameters()).get(grad_param_name)
    assert grad_tensor is not None, f"No parameter named {grad_param_name}"
    mal_grad_norm = grad_tensor.grad.norm().item()
    print(f"loss_mal grad norm ({grad_param_name}): {mal_grad_norm:.6f}")
    assert mal_grad_norm > 1e-6, f"Non-CDN loss gradient too small: {mal_grad_norm:.6e}"

    # ── Gradient from CDN loss (sum of _dn_.*_mal) ──
    dn_mal = sum(v for k, v in loss_dict.items() if "_dn_" in k and "_mal" in k)
    if dn_mal > 0:
        model.zero_grad()
        dn_mal.backward(retain_graph=True)

        dn_grad_norm = grad_tensor.grad.norm().item()
        print(f"dn_mal grad norm ({grad_param_name}): {dn_grad_norm:.6f}")
        assert dn_grad_norm > 1e-6, f"CDN loss gradient too small: {dn_grad_norm:.6e}"
    else:
        print("dn_mal is zero — skipping CDN gradient check")


def test_cdn_generation_visualization():
    """Visualize CDN input queries (before decoder): GT ellipses vs positive/negative noisy queries.

    num_denoising=20, num_queries=40. GT targets as Gaussian ellipses. Positive CDN queries
    (correct label + box noise) in red, negative (wrong label + box noise) in orange.
    """
    num_classes = 10
    hidden_dim = 64
    num_denoising = 20
    num_queries = 40

    class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

    # ── generate random samples across GT quantity levels ──
    num_gts_per_image = [1, 2, 5, 10, 20, 50, 100]
    rng = torch.Generator().manual_seed(42)
    img_w, img_h = 640.0, 640.0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = "test/outputs/cdn_inspect"
    os.makedirs(output_dir, exist_ok=True)

    for level_idx, n_gt in enumerate(num_gts_per_image):
        # ── random GT: cx,cy ∈ [0.15,0.85], w,h ∈ [0.03,0.25], θ ∈ [0,π] ──
        cx = 0.15 + 0.70 * torch.rand(n_gt, generator=rng)
        cy = 0.15 + 0.70 * torch.rand(n_gt, generator=rng)
        w = 0.03 + 0.22 * torch.rand(n_gt, generator=rng)
        h = 0.03 + 0.22 * torch.rand(n_gt, generator=rng)
        theta = torch.pi * torch.rand(n_gt, generator=rng)
        gt_boxes = torch.stack([cx, cy, w, h, theta], dim=-1)
        gt_labels = torch.randint(0, num_classes, (n_gt,))

        targets = [{"labels": gt_labels, "boxes": gt_boxes}]

        # ── CDN generation (batch=1 for simplicity) ──
        logits, bbox_unact, _attn, dn_meta = get_contrastive_denoising_training_group(
            targets=targets,
            num_classes=num_classes,
            num_queries=num_queries,
            class_embed=class_embed,
            num_denoising=num_denoising,
            label_noise_ratio=0.5,
            box_noise_scale=0.5,
            box_mode="obb",
        )
        assert logits is not None, f"CDN generation returned None for n_gt={n_gt}"

        actual_denoising = bbox_unact.shape[1]
        dn_positive_idx = dn_meta["dn_positive_idx"]

        # Denormalize CDN bboxes: inverse_sigmoid → sigmoid → θ back to [0,π]
        cdn_bboxes_norm = bbox_unact.sigmoid()
        cdn_bboxes_norm[..., 4] = cdn_bboxes_norm[..., 4] * torch.pi

        # ── denormalize GT to pixel ──
        gt_px = gt_boxes.clone()
        gt_px[:, 0] *= img_w
        gt_px[:, 1] *= img_h
        gt_px[:, 2] *= img_w
        gt_px[:, 3] *= img_h

        # CDN positive: correct label + box noise
        pos_idx = dn_positive_idx[0]  # batch=1
        pos_labels = gt_labels[pos_idx % (2 * n_gt)]
        pos_px = cdn_bboxes_norm[0][pos_idx].clone()
        pos_px[:, 0] *= img_w
        pos_px[:, 1] *= img_h
        pos_px[:, 2] *= img_w
        pos_px[:, 3] *= img_h

        # CDN negative: wrong label + box noise
        all_idx = set(range(actual_denoising))
        neg_idx_list = sorted(all_idx - set(pos_idx.tolist()))
        neg_idx = torch.tensor(neg_idx_list)
        neg_px = cdn_bboxes_norm[0][neg_idx].clone()
        neg_px[:, 0] *= img_w
        neg_px[:, 1] *= img_h
        neg_px[:, 2] *= img_w
        neg_px[:, 3] *= img_h

        # recover negative query labels from embedding (cosine nearest match)
        neg_logits = logits[0][neg_idx]
        neg_logits_n = F.normalize(neg_logits.float(), dim=-1)
        embed_w = class_embed.weight[:num_classes]
        embed_w_n = F.normalize(embed_w.float(), dim=-1)
        neg_labels = (neg_logits_n @ embed_w_n.T).argmax(dim=-1)

        # ── 3-panel figure ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"n_gt={n_gt}  |  CDN: {actual_denoising} queries (pos={len(pos_idx)}, neg={len(neg_idx)})",
            fontsize=13,
        )

        axes[0].set_xlim(0, img_w)
        axes[0].set_ylim(img_h, 0)
        axes[0].set_aspect("equal")
        axes[0].set_title(f"GT ({n_gt} boxes)")
        _draw_obb_ellipses(
            axes[0],
            gt_px,
            "green",
            labels=gt_labels,
            linewidth=max(0.5, 3.0 - 0.03 * n_gt),
        )
        axes[0].scatter(gt_px[:, 0], gt_px[:, 1], c="green", s=10, marker="x", zorder=5)

        axes[1].set_xlim(0, img_w)
        axes[1].set_ylim(img_h, 0)
        axes[1].set_aspect("equal")
        axes[1].set_title(f"CDN Positive ({len(pos_idx)} queries)")
        _draw_obb_ellipses(axes[1], gt_px, "green", labels=gt_labels, linewidth=2)
        _draw_obb_boxes(axes[1], gt_px, "green", linewidth=2)
        _draw_obb_boxes(axes[1], pos_px, "red", labels=pos_labels, linewidth=0.8)
        axes[1].scatter(
            pos_px[:, 0], pos_px[:, 1], c="red", s=8, marker="o", alpha=0.4, zorder=5
        )

        axes[2].set_xlim(0, img_w)
        axes[2].set_ylim(img_h, 0)
        axes[2].set_aspect("equal")
        axes[2].set_title(f"CDN Negative ({len(neg_idx)} queries)")
        _draw_obb_ellipses(axes[2], gt_px, "green", labels=gt_labels, linewidth=2)
        _draw_obb_boxes(axes[2], gt_px, "green", linewidth=2)
        _draw_obb_boxes(axes[2], neg_px, "orange", labels=neg_labels, linewidth=0.8)
        axes[2].scatter(
            neg_px[:, 0], neg_px[:, 1], c="orange", s=8, marker="o", alpha=0.4, zorder=5
        )

        for ax in axes:
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"cdn_input_viz_n{level_idx}_{n_gt}.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  [{level_idx}] n_gt={n_gt:3d} → {out_path}")

    # ── assertions ──
    for level_idx, n_gt in enumerate(num_gts_per_image):
        path = os.path.join(output_dir, f"cdn_input_viz_n{level_idx}_{n_gt}.png")
        assert os.path.exists(path), f"Missing {path}"
        size = os.path.getsize(path)
        assert size > 5000, f"{path} too small: {size} bytes"
        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic == b"\x89PNG", f"{path} not a valid PNG: {magic}"
        print(f"  {path}: {size:,} bytes, valid PNG")


def test_cdn_dn_budget_clamp_on_dense_batch():
    """DN 查询预算钳位: max_gt > num_denoising 时 dn_total 封顶 2*num_denoising.

    背景: DOTA-v1.5 切图存在单 patch 2449 实例的超密集样本, 原逻辑
    num_group 0→1 钳位使 dn_total = 2*max_gt 无界膨胀, 注意力矩阵
    (num_queries+dn_total)^2 瞬态十几 GB 直接 OOM。
    """
    torch.manual_seed(0)
    num_classes = 16
    num_denoising = 100
    num_queries = 300
    class_embed = nn.Embedding(num_classes + 1, 64, padding_idx=num_classes)

    # 图0: 250 个 GT(超预算), 图1: 3 个 GT, 图2: 0 个 GT
    dense_gt = 250
    targets = [
        {
            "labels": torch.randint(0, num_classes, (dense_gt,)),
            "boxes": torch.rand(dense_gt, 5) * 0.5,
        },
        {
            "labels": torch.tensor([1, 5, 9]),
            "boxes": torch.rand(3, 5) * 0.5,
        },
        {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 5)},
    ]

    logits, bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
        targets=targets,
        num_classes=num_classes,
        num_queries=num_queries,
        class_embed=class_embed,
        num_denoising=num_denoising,
        box_mode="obb",
    )

    # dn_total 封顶 = 2 * num_group(=1) * num_denoising = 200
    assert dn_meta["dn_num_split"][0] == 2 * num_denoising, (
        f"dn_total 应封顶 {2 * num_denoising}, got {dn_meta['dn_num_split'][0]}"
    )
    tgt_size = 2 * num_denoising + num_queries
    assert logits.shape == (3, 2 * num_denoising, 64)  # embedding_dim=64
    assert bbox_unact.shape == (3, 2 * num_denoising, 5)
    assert attn_mask.shape == (tgt_size, tgt_size)

    # 每图 DN 正样本数 = min(num_gt, num_denoising) * num_group(=1)
    sizes = [s.numel() for s in dn_meta["dn_positive_idx"]]
    assert sizes == [100, 3, 0], f"per-image positives 应为 [100, 3, 0], got {sizes}"


def test_cdn_normal_batch_unchanged_by_budget_clamp():
    """普通批次(max_gt ≤ num_denoising)行为与钳位前完全一致."""
    torch.manual_seed(0)
    num_classes = 16
    num_denoising = 100
    num_queries = 300
    class_embed = nn.Embedding(num_classes + 1, 64, padding_idx=num_classes)

    # max_gt = 10 → num_group = 10 → dn_total = 2*10*10 = 200(与原逻辑一致)
    targets = [
        {
            "labels": torch.randint(0, num_classes, (10,)),
            "boxes": torch.rand(10, 5) * 0.5,
        },
        {
            "labels": torch.randint(0, num_classes, (4,)),
            "boxes": torch.rand(4, 5) * 0.5,
        },
    ]

    logits, bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
        targets=targets,
        num_classes=num_classes,
        num_queries=num_queries,
        class_embed=class_embed,
        num_denoising=num_denoising,
        box_mode="obb",
    )

    assert dn_meta["dn_num_split"][0] == 200
    assert dn_meta["dn_num_group"] == 10
    assert logits.shape == (2, 200, 64)  # embedding_dim=64
    sizes = [s.numel() for s in dn_meta["dn_positive_idx"]]
    # 每图正样本 = min(num_gt, 100) * num_group = gt * 10
    assert sizes == [100, 40]
