"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

from typing import Iterable
from pathlib import Path

import torch
from torch.utils._pytree import tree_map

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..eval.obb_eval import obb_evaluate
from .training_diagnostics import (
    NanGradientTracker,
    validate_max_optimizer_steps,
    raise_for_nonfinite_losses,
    raise_for_nonfinite_total,
    inspect_gradients,
)
# Parameters whose NaN/Inf gradients are zeroed (not crashed) with a log warning.
_NAN_ZERO_PATTERNS: frozenset[str] = frozenset({"cls_token"})


# ---------------------------------------------------------------------------
#  Loss key parser: "loss_mal_aux_3" → ("loss_mal", "aux", 3)
# ---------------------------------------------------------------------------
import re

# 后缀模式 → source 映射（按优先级：enc > aux > _dn_pre > dn > pre）
_SUFFIX_PATTERNS = [
    ("_enc_", "enc"),
    ("_aux_", "aux"),
    ("_dn_pre", "dn_pre"),
    ("_dn_", "dn"),
    ("_pre", "pre"),
]


def parse_loss_key(key: str) -> tuple[str, str, int | None]:
    """Parse a criterion loss-dict key into (loss_family, source, layer_index).

    Examples:
        "loss_mal"         → ("loss_mal", "main", None)
        "loss_mal_aux_3"   → ("loss_mal", "aux", 3)
        "loss_kld_dn_1"    → ("loss_kld", "dn", 1)
        "loss_bbox_enc_0"  → ("loss_bbox", "enc", 0)
        "loss_mal_dn_pre"  → ("loss_mal", "dn_pre", None)
        "loss_fgl_pre"     → ("loss_fgl", "pre", None)
    """
    for suffix, source in _SUFFIX_PATTERNS:
        if suffix in key:
            prefix = key[: key.index(suffix)]
            idx_str = key[key.index(suffix) + len(suffix) :]
            return (prefix, source, int(idx_str) if idx_str else None)
    return (key, "main", None)


def compute_param_norm(model: torch.nn.Module) -> float:
    """sqrt(sum(param.norm(2)^2)) over all parameters.

    One device→host ``.item()`` sync per parameter tensor — call only where
    the value is consumed (comet batch log, rank0, every 50 steps), not on
    every iteration (server 20260820_182623 slowdown-wave amplification).
    """
    return (
        sum(p.data.float().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
    )


# 梯度模块组映射：按参数名前缀聚合到 backbone / encoder / decoder / head
_GRAD_GROUPS = {
    "backbone": "backbone",
    "encoder": "encoder",
    "decoder": "decoder",
    "head": ("class_embed", "bbox_embed", "enc_score_head", "enc_bbox_head"),
}
_GRAD_HIST_EVERY_N_EPOCH = 5


def _log_gradient_stats(model: torch.nn.Module, comet_exp, epoch: int) -> None:
    """按模块组聚合梯度统计量（norm/max/mean），替代逐参数直方图。

    Called every _GRAD_HIST_EVERY_N_EPOCH epochs on the main process only.
    每次上报 4 组 × 3 标量 = 12 次 API 调用，避免 rate limit。
    """
    from collections import defaultdict

    group_grads = defaultdict(list)
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_abs = param.grad.detach().cpu().float().abs()
        for group_key, patterns in _GRAD_GROUPS.items():
            if isinstance(patterns, str):
                patterns = (patterns,)
            if any(name.startswith(p) for p in patterns):
                group_grads[group_key].append(grad_abs)
                break

    for group_key, grads in group_grads.items():
        if not grads:
            continue
        all_grads = torch.cat([g.flatten() for g in grads])
        comet_exp.log_metric(
            f"grad/{group_key}/norm", all_grads.norm(2).item(), epoch=epoch
        )
        comet_exp.log_metric(
            f"grad/{group_key}/max", all_grads.max().item(), epoch=epoch
        )
        comet_exp.log_metric(
            f"grad/{group_key}/mean", all_grads.mean().item(), epoch=epoch
        )


import json


def _save_nan_snapshot(
    tracker: NanGradientTracker,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_dict: dict[str, torch.Tensor],
    epoch: int,
    step: int,
    output_dir: Path | str | None,
) -> None:
    """Save a full training snapshot when NaN grads are first seen for a param."""
    firsts = tracker.pop_firsts()
    if not firsts or output_dir is None:
        return

    out = Path(output_dir) / "nan_snapshots"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"nan_e{epoch}_s{step}"

    loss_snapshot = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
    (out / f"{stem}_loss.json").write_text(json.dumps(loss_snapshot, indent=2))

    if dist_utils.is_main_process():
        dist_utils.save_on_master(
            model.state_dict(),
            out / f"{stem}_model.pth",
        )
        torch.save(optimizer.state_dict(), out / f"{stem}_optimizer.pth")

    (out / f"{stem}_params.txt").write_text(
        f"epoch={epoch} step={step}\n"
        + "\n".join(firsts)
        + f"\n\ntotal_nan_events={tracker.count}"
    )

    print(
        f"[WARN] NaN gradient snapshot saved to {out}/{stem}_* "
        f"(params: {', '.join(firsts)})"
    )


def train_one_epoch(
    self_lr_scheduler,
    lr_scheduler,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ", log_level="minimal")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    print_freq = kwargs.get("print_freq", 10)

    ema: ModelEMA = kwargs.get("ema", None)
    use_amp: bool = kwargs.get("use_amp", False)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    kendall = kwargs.get("kendall", None)
    kendall_optimizer = kwargs.get("kendall_optimizer", None)

    comet_exp = kwargs.get("comet_exp", None)
    comet_step = kwargs.get("comet_step", None)

    max_optimizer_steps = kwargs.get("max_optimizer_steps", None)
    fail_on_zero_grad = kwargs.get("fail_on_zero_grad", False)
    nan_max_events = kwargs.get("nan_max_events", 10)
    nan_output_dir = kwargs.get("output_dir", None)
    validate_max_optimizer_steps(max_optimizer_steps)

    nan_tracker = NanGradientTracker(max_events=int(nan_max_events))

    # Per-rank iteration timestamps for multi-GPU hang diagnosis; no-op
    # unless DEIM_DIAG_HEARTBEAT=1 (see engine/misc/diag.py).
    from ..misc.diag import Heartbeat

    _heartbeat = Heartbeat(output_dir=str(nan_output_dir) if nan_output_dir else ".")

    cur_iters = epoch * len(data_loader)

    _optimizer_step_count = 0
    _step_cap_reached = False

    use_tqdm = tqdm is not None
    if use_tqdm:
        data_loader_iter = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f"Epoch {epoch}",
            leave=True,
        )
    else:
        data_loader_iter = enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        )

    for i, (samples, targets) in data_loader_iter:
        _heartbeat.beat(
            epoch=epoch,
            step=i,
            extra={"global_step": epoch * len(data_loader) + i},
        )
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(
            epoch=epoch,
            step=i,
            global_step=global_step,
            epoch_step=len(data_loader),
        )

        # ── Step cap: break early if limit reached ────────────────────────────
        if max_optimizer_steps is not None and _optimizer_step_count >= max_optimizer_steps:
            _step_cap_reached = True
            break

        if use_amp:
            with torch.autocast(
                device_type=str(device), dtype=torch.bfloat16, cache_enabled=True
            ):
                outputs = model(samples, targets=targets)

            # BF16 前向输出递归转为 FP32，确保 criterion 全程以 FP32 计算
            outputs = tree_map(
                lambda t: t.float()
                if isinstance(t, torch.Tensor) and t.is_floating_point()
                else t,
                outputs,
            )

            if (
                torch.isnan(outputs["pred_boxes"]).any()
                or torch.isinf(outputs["pred_boxes"]).any()
            ):
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            raise_for_nonfinite_losses(loss_dict, epoch=epoch, step=i, global_step=global_step)

            loss = sum(loss_dict.values())
            raise_for_nonfinite_total(loss, epoch=epoch, step=i, global_step=global_step)
            loss.backward()

            _, zeroed = inspect_gradients(
                model.named_parameters(),
                fail_on_zero_grad=fail_on_zero_grad,
                epoch=epoch,
                step=i,
                global_step=global_step,
                nan_zero_patterns=_NAN_ZERO_PATTERNS,
                nan_tracker=nan_tracker,
            )
            _save_nan_snapshot(nan_tracker, model, optimizer, loss_dict, epoch, i, nan_output_dir)

            if max_norm > 0:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm
                )
                grad_norm_after = min(grad_norm_before, max_norm)
            else:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf")
                )
                grad_norm_after = grad_norm_before

            optimizer.step()
            optimizer.zero_grad()
            _optimizer_step_count += 1

            if ema is not None:
                ema.update(model)

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            raise_for_nonfinite_losses(loss_dict, epoch=epoch, step=i, global_step=global_step)

            optimizer.zero_grad()
            if kendall_optimizer is not None:
                kendall_optimizer.zero_grad()

            if kendall is not None:
                loss = kendall.weighted_loss(loss_dict)
                raise_for_nonfinite_total(loss, epoch=epoch, step=i, global_step=global_step)
                loss.backward()

                _, zeroed = inspect_gradients(
                    model.named_parameters(),
                    fail_on_zero_grad=fail_on_zero_grad,
                    epoch=epoch,
                    step=i,
                    global_step=global_step,
                    nan_zero_patterns=_NAN_ZERO_PATTERNS,
                    nan_tracker=nan_tracker,
                )
                _save_nan_snapshot(nan_tracker, model, optimizer, loss_dict, epoch, i, nan_output_dir)

                if max_norm > 0:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm
                    )
                    grad_norm_after = min(grad_norm_before, max_norm)
                else:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                    grad_norm_after = grad_norm_before

                optimizer.step()
                if kendall_optimizer is not None:
                    kendall_optimizer.step()
                _optimizer_step_count += 1

                if dist_utils.is_main_process() and global_step % 50 == 0:
                    weights = kendall.get_weights()
                    if comet_exp:
                        try:
                            for k, w in weights.items():
                                comet_exp.log_metric(
                                    f"kendall/{k}", w, step=global_step
                                )
                        except Exception:
                            pass
            else:
                loss: torch.Tensor = sum(loss_dict.values())
                raise_for_nonfinite_total(loss, epoch=epoch, step=i, global_step=global_step)
                loss.backward()

                _, zeroed = inspect_gradients(
                    model.named_parameters(),
                    fail_on_zero_grad=fail_on_zero_grad,
                    epoch=epoch,
                    step=i,
                    global_step=global_step,
                    nan_zero_patterns=_NAN_ZERO_PATTERNS,
                    nan_tracker=nan_tracker,
                )
                _save_nan_snapshot(nan_tracker, model, optimizer, loss_dict, epoch, i, nan_output_dir)

                if max_norm > 0:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm
                    )
                    grad_norm_after = min(grad_norm_before, max_norm)
                else:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                    grad_norm_after = grad_norm_before

                optimizer.step()
                _optimizer_step_count += 1

            if ema is not None:
                ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # -------- tqdm 进度条：显示主损失 --------
        if use_tqdm:
            postfix_dict = {
                "lr": f'{optimizer.param_groups[0]["lr"]:.8f}',
                "total": f"{loss_value:.4f}",
            }
            for k, v in loss_dict_reduced.items():
                family, source, _ = parse_loss_key(k)
                if source == "main":
                    short_name = k.replace("loss_", "")
                    postfix_dict[short_name] = f"{v:.4f}"
            data_loader_iter.set_postfix(postfix_dict)

        # -------- Comet batch 日志 --------
        if comet_exp and dist_utils.is_main_process() and global_step % 50 == 0:
            try:
                for k, v in loss_dict_reduced.items():
                    family, source, idx = parse_loss_key(k)
                    metric_name = f"{source}/{family}"
                    if idx is not None:
                        if source in ("aux", "enc"):
                            metric_name += f"/layer_{idx}"
                        elif source in ("dn", "dn_pre"):
                            metric_name += f"/group_{idx}"
                    comet_exp.log_metric(metric_name, v.item(), step=global_step)

                comet_exp.log_metric(
                    "main/loss_total", loss_value.item(), step=global_step
                )
                comet_exp.log_metric(
                    "lr", optimizer.param_groups[0]["lr"], step=global_step
                )
                comet_exp.log_metric(
                    "grad/norm/before_clip", grad_norm_before, step=global_step
                )
                comet_exp.log_metric(
                    "grad/norm/after_clip", grad_norm_after, step=global_step
                )
                comet_exp.log_metric(
                    "param/norm",
                    compute_param_norm(model),
                    step=global_step,
                )
            except Exception:
                pass

        # -------- 显存防碎片 --------
        # DOTA 变长 GT 使 loss/matcher 中间张量尺寸逐迭代漂移, 缓存分配器的
        # reserved 膨胀且空闲段无法复用(实测 allocated <1GB 而 reserved 在
        # 60 iter 内从 5GB 涨到 16.7GB, 最终 backward 时 CUDA OOM)。
        # 周期性把空闲段归还驱动, 封顶 reserved; 每次调用 ~几十 ms, 开销可忽略。
        if i % 50 == 0:
            torch.cuda.empty_cache()

    # ── Step cap early return: if reached, skip EMA, scheduler, and final logging ──
    if _step_cap_reached:
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stats["_step_cap_reached"] = True
        return train_stats

    metric_logger.synchronize_between_processes()

    if dist_utils.is_main_process():
        print("\n" + "=" * 60)
        print(f"Training Summary - Epoch {epoch}")
        print("=" * 60)
        for k, meter in metric_logger.meters.items():
            if k == "loss":
                print(f"  Total Loss:     {meter.global_avg:.4f}")
            elif k == "lr":
                print(f"  Learning Rate:  {meter.global_avg:.6f}")
            elif k.startswith("loss_"):
                family, source, _ = parse_loss_key(k)
                if source == "main":
                    label = k.replace("loss_", "").upper()
                    print(f"  {label:16s} {meter.global_avg:.4f}")
        print("=" * 60 + "\n")

    if comet_exp and dist_utils.is_main_process():
        try:
            for k, meter in metric_logger.meters.items():
                if k == "loss":
                    comet_exp.log_metric(
                        "main/loss_total_epoch", meter.global_avg, epoch=comet_step
                    )
                elif k == "lr":
                    comet_exp.log_metric("lr_epoch", meter.global_avg, epoch=comet_step)
                elif k.startswith("loss_"):
                    family, source, idx = parse_loss_key(k)
                    metric_name = f"{source}/{family}"
                    if idx is not None:
                        if source in ("aux", "enc"):
                            metric_name += f"/layer_{idx}"
                        elif source in ("dn", "dn_pre"):
                            metric_name += f"/group_{idx}"
                    comet_exp.log_metric(
                        f"{metric_name}_epoch", meter.global_avg, epoch=comet_step
                    )

            if epoch > 0 and epoch % _GRAD_HIST_EVERY_N_EPOCH == 0:
                _log_gradient_stats(model, comet_exp, comet_step)
        except Exception:
            pass

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    box_mode="hbb",
    **kwargs,
):
    # OBB evaluation path
    if box_mode == "obb":
        print("\n" + "=" * 60)
        print("Validation Summary (OBB)")
        print("=" * 60)
        stats = obb_evaluate(
            model,
            postprocessor,
            data_loader,
            device,
            num_classes=postprocessor.num_classes,
        )
        print(f"AP @ IoU=0.50      = {stats.get('AP50', 0):.4f}")
        print(f"AP @ IoU=0.75      = {stats.get('AP75', 0):.4f}")
        print(f"mAP@0.5:0.95       = {stats['mAP']:.4f}")
        print(f"Precision (max-F1) = {stats['precision']:.4f}")
        print(f"Recall    (max-F1) = {stats['recall']:.4f}")
        print(f"F1        (max-F1) = {stats.get('f1', 0):.4f}")
        print("-" * 50)
        print("=" * 60 + "\n")

        comet_exp = kwargs.get("comet_exp", None)
        comet_step = kwargs.get("comet_step", None)
        if comet_exp and dist_utils.is_main_process():
            for k, v in stats.items():
                comet_exp.log_metric(f"val_{k}", v, epoch=comet_step)
        return stats, None

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ", log_level="minimal")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"
    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    comet_exp = kwargs.get("comet_exp", None)
    comet_step = kwargs.get("comet_step", None)

    # 验证时也使用 tqdm
    use_tqdm = tqdm is not None
    # use_tqdm =None
    if use_tqdm:
        data_loader_iter = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Validating",
            leave=True,
        )
    else:
        data_loader_iter = enumerate(metric_logger.log_every(data_loader, 10, header))

    for idx, load_vars in data_loader_iter:
        samples = load_vars[0]
        targets = load_vars[1]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        stats = {}
        if "bbox" in coco_evaluator.iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
            print(f"AP @ IoU=0.50:0.95 = {stats['coco_eval_bbox'][0]:.4f}")
            print(f"AP @ IoU=0.50       = {stats['coco_eval_bbox'][1]:.4f}")
            print(f"AP @ IoU=0.75       = {stats['coco_eval_bbox'][2]:.4f}")
            print(f"AP @ IoU=0.50 (small)= {stats['coco_eval_bbox'][3]:.4f}")
            print(f"AP @ IoU=0.50 (medium)= {stats['coco_eval_bbox'][4]:.4f}")
            print(f"AP @ IoU=0.50 (large) = {stats['coco_eval_bbox'][5]:.4f}")

            stats_dict = coco_evaluator.coco_eval["bbox"].stats_as_dict
            print(f"AR @ IoU=0.50:0.95 = {stats_dict['AR_all']:.4f}")
            print(f"AR @ IoU=0.50       = {stats_dict['AR_50']:.4f}")
            print(f"AR @ IoU=0.75       = {stats_dict['AR_75']:.4f}")
            print(f"AR @ IoU=0.50 (small)= {stats_dict['AR_small']:.4f}")
            print(f"AR @ IoU=0.50 (medium)= {stats_dict['AR_medium']:.4f}")
            print(f"AR @ IoU=0.50 (large) = {stats_dict['AR_large']:.4f}")
            print("-" * 50)
            metrics_dict = coco_evaluator.coco_eval["bbox"].extended_metrics
            print(metrics_dict)
            print("-" * 50)

            if comet_exp is not None and dist_utils.is_main_process():
                comet_exp.log_metric("recall", metrics_dict["recall"], epoch=comet_step)
                comet_exp.log_metric(
                    "precision", metrics_dict["precision"], epoch=comet_step
                )
                comet_exp.log_metric(
                    "AR_0.5:95", stats_dict["AR_all"], epoch=comet_step
                )
                comet_exp.log_metric("AR_50", stats_dict["AR_50"], epoch=comet_step)
                comet_exp.log_metric(
                    "AP_0.5:95", stats_dict["AP_all"], epoch=comet_step
                )
                comet_exp.log_metric("AP_50", stats_dict["AP_50"], epoch=comet_step)
        print("=" * 60 + "\n")

    stats = {}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
