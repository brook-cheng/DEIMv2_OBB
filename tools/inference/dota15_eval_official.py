"""
DOTA-v1.5 官方评估链路 · 第二/三步: patch 结果合并 + 官方 16 类评估。

输入 dota15_task1_infer.py 的 Task1_{class}.txt, 依次:
  1) valset.txt: 从 GT 目录(原图标注 labelTxt-v1.5)生成 val 图名清单
  2) ResultMerge: patch 坐标还原原图 + polyiou 多边形 NMS(阈值 0.1)
  3) dota-v1.5_evaluation_task1: VOC07 11 点 AP @ IoU 0.5, difficult 全按 0

不改 devkit 源码: 通过 import 调用; dota-v1.5_evaluation_task1.py 的
np.bool(已在新版 numpy 移除)在本脚本内打补丁兼容。

用法:
    python tools/inference/dota15_eval_official.py \
        --task1-dir <Task1 输出目录> \
        --gt-dir <DOTA-V1.5/val/labelTxt-v1.5> \
        --work-dir <合并结果与报告落盘目录>
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEVKIT_PATH = os.path.join(REPO_ROOT, "third_party", "DOTA_devkit")
sys.path.insert(0, DEVKIT_PATH)

import numpy as np

# devkit 遗留别名兼容: numpy>=1.24 移除了 np.bool
if not hasattr(np, "bool"):
    np.bool = bool

import ResultMerge_multi_process as result_merge  # noqa: E402  (devkit 模块)


def load_eval_task1():
    """import 文件名带连字符的 dota-v1.5_evaluation_task1.py。"""
    spec = importlib.util.spec_from_file_location(
        "dota15_eval_task1", os.path.join(DEVKIT_PATH, "dota-v1.5_evaluation_task1.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="DOTA-v1.5 官方评估(合并+16类AP)")
    parser.add_argument("--task1-dir", required=True, help="dota15_task1_infer.py 输出目录")
    parser.add_argument("--gt-dir", required=True, help="val 原图标注目录(labelTxt-v1.5)")
    parser.add_argument("--work-dir", required=True, help="合并结果/valset/报告目录")
    parser.add_argument("--nms-thresh", type=float, default=0.1, help="合并 polyiou NMS 阈值")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="评估 IoU 阈值")
    args = parser.parse_args()

    task1_dir = Path(args.task1_dir)
    gt_dir = Path(args.gt_dir)
    work_dir = Path(args.work_dir)
    merged_dir = work_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) valset.txt: GT 目录即 val 全集 ----
    val_names = sorted(p.stem for p in gt_dir.glob("*.txt"))
    valset_file = work_dir / "valset.txt"
    valset_file.write_text("\n".join(val_names) + "\n")
    print(f"[1/3] valset: {len(val_names)} 张原图 -> {valset_file}")

    # ---- 2) ResultMerge: patch -> 原图 + polyiou NMS ----
    result_merge.nms_thresh = args.nms_thresh
    t0 = time.time()
    result_merge.mergebypoly(str(task1_dir), str(merged_dir))
    n_merged = len(list(merged_dir.glob("*.txt")))
    print(f"[2/3] ResultMerge 完成: {n_merged} 个类文件, NMS={args.nms_thresh}, "
          f"耗时 {time.time()-t0:.0f}s")

    # ---- 3) 官方评估: 16 类 VOC07 11 点 AP ----
    eval_mod = load_eval_task1()
    # 与 dota-v1.5_evaluation_task1.__main__ 一致的官方类序(与 classes.txt 相同)
    classnames = [
        "plane", "baseball-diamond", "bridge", "ground-track-field",
        "small-vehicle", "large-vehicle", "ship", "tennis-court",
        "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
        "harbor", "swimming-pool", "helicopter", "container-crane",
    ]
    detpath = os.path.join(str(merged_dir), "Task1_{:s}.txt")
    annopath = os.path.join(str(gt_dir), "{:s}.txt")

    classaps = {}
    for classname in classnames:
        # 空检测文件在 numpy 2.x 下会使 devkit 的 BB[sorted_ind, :] 崩溃;
        # 无检测 → AP=0 为 VOC 语义, 直接短路
        detfile = detpath.format(classname)
        if not any(line.strip() for line in open(detfile)):
            classaps[classname] = 0.0
            print(f"  {classname:22s} AP50 = 0.00 (无检测)")
            continue
        rec, prec, ap = eval_mod.voc_eval(
            detpath, annopath, str(valset_file), classname,
            ovthresh=args.iou_thresh, use_07_metric=True,
        )
        classaps[classname] = ap * 100
        print(f"  {classname:22s} AP50 = {ap*100:.2f}")

    mAP = sum(classaps.values()) / len(classaps)
    print(f"\nmAP50(官方 VOC07 11点, IoU {args.iou_thresh}) = {mAP:.2f}")

    report = work_dir / "eval_report.txt"
    lines = [f"mAP50 = {mAP:.2f} (VOC07 11pt, IoU {args.iou_thresh}, "
             f"NMS {args.nms_thresh})"] + [
        f"{name:22s} {ap:.2f}" for name, ap in classaps.items()
    ]
    report.write_text("\n".join(lines) + "\n")
    print(f"报告已写入 {report}")


if __name__ == "__main__":
    main()
