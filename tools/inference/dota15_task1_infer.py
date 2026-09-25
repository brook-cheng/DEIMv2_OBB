"""
DOTA-v1.5 官方评估链路 · 第一步: val_1024 patch 推理导出 Task1 格式结果。

对 val_1024 全部 patch 推理(预处理与训练 val 完全一致: 拉伸缩放到 eval_spatial_size
+ ImageNet 归一化), 预测框经 postprocessor 还原到 patch 原生像素坐标系(如 1024→640
训练则 ×1.6), 再转 8 坐标角点, 按类写盘:

    Task1_{class}.txt 每行: {patch名} {conf} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}

patch 名形如 P1868__1__6592___0 (原图名__rate__left___top), 供
DOTA_devkit/ResultMerge_multi_process.py 解析还原原图坐标。

用法:
    python tools/inference/dota15_task1_infer.py \
        -c configs/custom_obb/dota15/sp_fz_rep3.yml \
        -r outputs/dota15_sp_fz_rep3/best_stg2.pth \
        -i <val_1024/images 目录> \
        -o <Task1 输出目录> -d cuda:0

后续步骤(见仓库 docs / DOTA_devkit):
    2) ResultMerge_multi_process.py 合并 patch → 原图 (polyiou NMS 0.1)
    3) dota-v1.5_evaluation_task1.py 官方 16 类 VOC07 11 点 AP @ IoU 0.5
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from engine.core import YAMLConfig
from engine.deim.obb_geometry import xywhr_to_xyxyxyxy
from engine.solver._solver import model_is_obb

# 与 sp_fz_rep3.yml val_dataloader transforms 一致(ConvertPILImage scale=True + Normalize)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_deploy_model(cfg: YAMLConfig, ckpt_path: str, device: str) -> nn.Module:
    """加载 checkpoint(优先 EMA 权重)并转 deploy 模式, 口径同 tools/inference/torch_inf.py。"""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    if model_is_obb(cfg.model):
        from engine.solver._solver import assert_checkpoint_compat

        assert_checkpoint_compat(checkpoint)
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    return DeployModel().to(device).eval()


def load_patch(path: Path, size_hw):
    """单 patch 预处理: 拉伸缩放 + to_tensor(/255) + 归一化, 返回 (tensor, (w, h)原生尺寸)。"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = TF.resize(im, list(size_hw))
    x = TF.to_tensor(im)
    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x, (w, h)


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # 模型构造时缓存 anchor 用的分辨率 [H, W], 推理缩放目标与此一致
    size_hw = cfg.yaml_cfg["eval_spatial_size"]
    label_names = [
        line.strip() for line in open(args.classes_file, "r").readlines() if line.strip()
    ]

    model = build_deploy_model(cfg, args.resume, args.device)
    device = args.device

    img_dir = Path(args.img_dir)
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"{img_dir} 下未找到图像({IMG_EXTS})")
    print(f"待推理 patch: {len(files)} | 输入尺寸: {size_hw} | 阈值: {args.score_thr}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_files = {
        name: open(out_dir / f"Task1_{name}.txt", "w") for name in label_names
    }
    class_counts = {name: 0 for name in label_names}

    t0 = time.time()
    with torch.inference_mode(), ThreadPoolExecutor(args.num_workers) as pool:
        for start in range(0, len(files), args.batch_size):
            chunk = files[start : start + args.batch_size]
            results = list(pool.map(lambda p: load_patch(p, size_hw), chunk))
            images = torch.stack([r[0] for r in results]).to(device)
            # [B, 2] = [w, h]: postprocessor 据此把归一化框缩放回 patch 原生坐标系
            orig_sizes = torch.tensor([r[1] for r in results], device=device)

            labels, boxes, scores = model(images, orig_sizes)
            # boxes: [B, num_top_queries, 5] = (cx, cy, w, h, θ) patch 像素坐标, θ∈[0,π)
            corners = xywhr_to_xyxyxyxy(boxes)  # [B, N, 4, 2] 顺时针四角点

            labels = labels.cpu()
            scores = scores.cpu()
            corners = corners.cpu()
            for i, path in enumerate(chunk):
                stem = path.stem
                mask = scores[i] > args.score_thr
                for lab, sco, poly in zip(
                    labels[i][mask], scores[i][mask], corners[i][mask]
                ):
                    name = label_names[int(lab)]
                    coords = " ".join(f"{v:.2f}" for v in poly.reshape(-1).tolist())
                    class_files[name].write(f"{stem} {sco:.6f} {coords}\n")
                    class_counts[name] += 1

            done = min(start + args.batch_size, len(files))
            if done % (args.batch_size * 25) < args.batch_size or done == len(files):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"[{done}/{len(files)}] {rate:.1f} img/s", flush=True)

    for f in class_files.values():
        f.close()

    total = sum(class_counts.values())
    print(f"\n完成: 共 {total} 条检测 (阈值 {args.score_thr}), 耗时 {time.time()-t0:.0f}s")
    for name in label_names:
        print(f"  {name:22s} {class_counts[name]}")
    print(f"Task1 文件已写入 {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DOTA-v1.5 Task1 patch 推理导出")
    parser.add_argument("-c", "--config", type=str, required=True, help="训练配置 yml")
    parser.add_argument("-r", "--resume", type=str, required=True, help="checkpoint 路径 (优先取 ema.module)")
    parser.add_argument("-i", "--img-dir", type=str, required=True, help="val_1024/images 目录")
    parser.add_argument("-o", "--out-dir", type=str, required=True, help="Task1_* 输出目录")
    parser.add_argument("--classes-file", type=str, required=True, help="16 类类表 (顺序=标签 id)")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.05, help="导出阈值 (社区默认 0.05)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8, help="图像解码线程数")
    args = parser.parse_args()
    main(args)
