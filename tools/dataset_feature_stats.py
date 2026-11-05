"""数据集目标特征统计工具（OBB）.

在训练开始前统计 train/val(/test) 各 split 的 OBB 标注特征，输出按框、按图、
按类别汇总的数据与分布图，为后续模型差异分析提供可关联的输入。

标注格式（与 engine/data/dataset/dota_dataset.py 的解析口径一致）:
    YOLO-OBB: ``cls_id x1 y1 x2 y2 x3 y3 x4 y4``，坐标为归一化值(除以图宽/图高)
    DOTA:     ``x1 y1 x2 y2 x3 y3 x4 y4 cls_name [difficult]``，坐标为绝对像素

几何口径（固定，勿改；与 engine/deim/obb_geometry.py::xyxyxyxy_to_xywhr 同口径，
单元测试中有与该 torch 实现的对拍校验）:
    length      = max(两条邻边长)
    width       = min(两条邻边长)
    aspect_ratio= length / width  (>= 1)
    angle       = 长边方向与 x 轴夹角 atan2 归一到 [0, pi) 弧度, 输出同时给度数 [0, 180)
    area_px     = 四点多边形鞋带面积
    归一化值:   cx_norm/cy_norm 按 YOLO 口径除以图宽/图高;
                length_norm/width_norm 除以图像对角线长; area_ratio = area_px / 图面积

输出文件（--output-dir 下按组组织: src/ 为原图统计, dst_<N>/ 为按 --dst_sz N*N
非等比拉伸预处理后[与训练 OBBResize 同口径: sx=N/w, sy=N/h]的统计, 两组结构相同）:
    feature_boxes.csv        每个有效标注框一行
    feature_image_counts.csv 每图每类一行(长表)
    feature_class_summary.csv 按 split×类别 的特征分位数汇总
    feature_summary.json     运行参数/类别映射/输入文件身份/样本统计/分布差异/错误统计
    errors.jsonl             异常明细(不静默跳过)
    plots/<split>/*.png      长度/宽度/长宽比/角度/面积分布图

关联键: split / case / line / tower / image_path / cid (feature_boxes.csv 恒有这些列,
manifest 未提供时 case/line/tower 为空串)。

用法:
    python tools/dataset_feature_stats.py \
        --data-root /path/to/yolo_obb_root \
        --splits train val \
        --dst_sz 640 \
        --output-dir outputs/dataset_feature_stats
    # 输出: <output-dir>/src/... 与 <output-dir>/dst_640/...
    # 自定义单目录(非 data-root 布局):
    python tools/dataset_feature_stats.py \
        --images-dir ... --labels-dir ... --split-name test \
        --classes-file classes.txt --output-dir out
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

TOOL_VERSION = "1.0.0"

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

BOX_METRICS = ("length_px", "width_px", "aspect_ratio", "angle_deg", "area_px", "area_ratio")
# 图内文本用英文: matplotlib 默认字体不含 CJK 字形, 避免方框
METRIC_LABELS = {
    "length_px": "length (px)",
    "width_px": "width (px)",
    "aspect_ratio": "aspect ratio (length/width)",
    "angle_deg": "angle (deg, [0,180))",
    "area_px": "OBB area (px^2)",
    "area_ratio": "area / image area",
}
STAT_FIELDS = ("mean", "median", "p5", "p25", "p75", "p95", "min", "max")

ERROR_TYPES = ("parse_error", "class_unmapped", "zero_width", "image_missing", "image_unreadable")

GEOMETRY_DEFINITIONS = {
    "length": "max(两条邻边长)",
    "width": "min(两条邻边长)",
    "aspect_ratio": "length / width, >= 1",
    "angle": "长边方向与 x 轴夹角, atan2 后归一到 [0, pi) rad 即 [0, 180) deg, "
             "与 engine/deim/obb_geometry.py::xyxyxyxy_to_xywhr 同口径",
    "area_px": "四点多边形鞋带面积(绝对值)",
    "area_ratio": "area_px / (image_width * image_height)",
    "cx_norm/cy_norm": "中心点像素坐标 / 图宽 / 图高 (YOLO 口径)",
    "length_norm/width_norm": "边长像素值 / 图像对角线长度 sqrt(w^2+h^2)",
}


# --------------------------------------------------------------------------- #
# 标注与配置解析
# --------------------------------------------------------------------------- #
class AnnotationError(Exception):
    """一条标注无法参与统计。type 用于 errors.jsonl 分类。"""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


def load_classes(classes_file: str) -> list[str]:
    with open(classes_file, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) == 0:
        raise ValueError(f"classes 文件为空: {classes_file}")
    return names


def load_class_map(path: str) -> dict[str, tuple[str, str]]:
    """class_map CSV: raw_name,business_name,cid -> {raw_name: (business_name, cid)}."""
    mapping: dict[str, tuple[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("raw_name") or "").strip()
            if raw:
                mapping[raw] = (
                    (row.get("business_name") or raw).strip(),
                    (row.get("cid") or "").strip(),
                )
    return mapping


def load_manifest(path: str) -> dict[str, dict[str, str]]:
    """manifest CSV: image_stem,case,line,tower -> {stem: {...}}."""
    manifest: dict[str, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            stem = (row.get("image_stem") or "").strip()
            if stem:
                manifest[stem] = {
                    "case": (row.get("case") or "").strip(),
                    "line": (row.get("line") or "").strip(),
                    "tower": (row.get("tower") or "").strip(),
                }
    return manifest


def parse_yolo_obb_line(line: str, num_classes: int) -> tuple[int, np.ndarray]:
    """解析一行 YOLO-OBB 标注, 返回 (raw_class_id, 4x2 归一化角点)。失败抛 AnnotationError。"""
    parts = line.strip().split()
    if len(parts) < 9:
        raise AnnotationError("parse_error", f"字段数 {len(parts)} < 9 (cls + 8 坐标)")
    try:
        cls_id = int(parts[0])
        pts = np.array([float(p) for p in parts[1:9]], dtype=np.float64)
    except ValueError as e:
        raise AnnotationError("parse_error", f"数值解析失败: {e}") from e
    if cls_id < 0 or cls_id >= num_classes:
        raise AnnotationError("class_unmapped", f"类别 id {cls_id} 不在 classes 文件 [0, {num_classes}) 内")
    return cls_id, pts.reshape(4, 2)


def parse_dota_line(line: str, cat2id: dict[str, int]) -> tuple[int, np.ndarray]:
    """解析一行 DOTA 标注, 返回 (raw_class_id, 4x2 像素角点)。"""
    parts = line.strip().split()
    if len(parts) < 9:
        raise AnnotationError("parse_error", f"字段数 {len(parts)} < 9 (8 坐标 + 类别名)")
    try:
        pts = np.array([float(p) for p in parts[:8]], dtype=np.float64)
    except ValueError as e:
        raise AnnotationError("parse_error", f"数值解析失败: {e}") from e
    # 与 DotaDataset 一致: 类别名可能含空格; 末列为 difficult 数字标志时剔除
    cat_parts = parts[8:-1] if parts[-1].isdigit() else parts[8:]
    cat = " ".join(cat_parts)
    if cat not in cat2id:
        raise AnnotationError("class_unmapped", f"类别名 '{cat}' 不在 classes 文件内")
    return cat2id[cat], pts.reshape(4, 2)


# --------------------------------------------------------------------------- #
# 几何计算（与 xyxyxyxy_to_xywhr 同口径）
# --------------------------------------------------------------------------- #
def obb_features(poly: np.ndarray) -> dict:
    """4x2 角点 -> {cx, cy, length, width, angle_rad[0,pi), area_px}。

    边长/角度算法与 engine/deim/obb_geometry.py::xyxyxyxy_to_xywhr 逐行对应:
    取距首点最近/次近的两个点作为两条邻边端点, 长边方向角即 OBB 角度。
    """
    ctr = poly.mean(axis=0)
    v0 = poly[0]
    dists = ((poly - v0) ** 2).sum(axis=1)
    order = np.argsort(dists, kind="stable")
    edg1 = poly[order[1]] - v0
    edg2 = poly[order[2]] - v0
    len1 = math.hypot(edg1[0], edg1[1])
    len2 = math.hypot(edg2[0], edg2[1])

    length, width = (len1, len2) if len1 >= len2 else (len2, len1)
    w_dx, w_dy = (edg1[0], edg1[1]) if len1 >= len2 else (edg2[0], edg2[1])
    angle_rad = math.atan2(w_dy, w_dx) % math.pi

    x, y = poly[:, 0], poly[:, 1]
    area_px = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    return {
        "cx": float(ctr[0]),
        "cy": float(ctr[1]),
        "length": float(length),
        "width": float(width),
        "angle_rad": float(angle_rad),
        "area_px": area_px,
    }


# --------------------------------------------------------------------------- #
# 输入发现与身份
# --------------------------------------------------------------------------- #
def list_images(folder: str) -> list[str]:
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    )


def discover_splits(data_root: str) -> list[str]:
    img_root = os.path.join(data_root, "images")
    if not os.path.isdir(img_root):
        return []
    splits = sorted(
        d for d in os.listdir(img_root)
        if os.path.isdir(os.path.join(img_root, d))
        and os.path.isdir(os.path.join(data_root, "labels", d))
    )
    return splits


def dir_fingerprint(folder: str, files: list[str], hash_content: bool) -> str:
    """目录身份指纹: 排序的 (文件名, 大小[, 内容 sha256]) 列表的 sha256。可重复。"""
    h = hashlib.sha256()
    for name in files:
        path = os.path.join(folder, name)
        size = os.path.getsize(path)
        h.update(f"{name}|{size}".encode("utf-8"))
        if hash_content:
            with open(path, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def read_image_sizes(image_paths: list[str], num_workers: int) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """批量读取图片 (width, height)。返回 (成功表, 失败 path 列表)。只读 header, 不解码像素。"""
    from PIL import Image

    sizes: dict[str, tuple[int, int]] = {}
    failed: list[str] = []

    def _read(path: str):
        with Image.open(path) as im:
            return path, im.size  # (width, height), lazy header 读取

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_read, p): p for p in image_paths}
        for fut in futures:
            try:
                path, size = fut.result()
                sizes[path] = (int(size[0]), int(size[1]))
            except Exception:
                failed.append(futures[fut])
    return sizes, failed


# --------------------------------------------------------------------------- #
# 核心统计
# --------------------------------------------------------------------------- #
def percentile_stats(values: list[float]) -> dict:
    """count/mean/median/p5/p25/p75/p95/min/max; 空列表返回全 None。"""
    if not values:
        return {"count": 0, **{k: None for k in STAT_FIELDS}}
    arr = np.asarray(values, dtype=np.float64)
    q = np.percentile(arr, [5, 25, 50, 75, 95])
    # q 索引: q[0]=P5, q[1]=P25, q[2]=P50(中位数), q[3]=P75, q[4]=P95
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(q[2]),
        "p5": float(q[0]),
        "p25": float(q[1]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


BOX_COLUMNS = [
    "split", "image_path", "image_stem", "image_width", "image_height",
    "orig_image_width", "orig_image_height",
    "raw_class_id", "class_name", "business_class", "cid",
    "cx", "cy", "cx_norm", "cy_norm",
    "length_px", "width_px", "length_norm", "width_norm",
    "aspect_ratio", "angle_deg", "angle_rad",
    "area_px", "area_ratio", "objects_in_image",
    "case", "line", "tower",
]


def analyze_split(
    split: str,
    images_dir: str,
    labels_dir: str,
    classes: list[str],
    fmt: str,
    class_map: dict[str, tuple[str, str]],
    manifest: dict[str, dict[str, str]],
    num_workers: int,
    dst_size: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """统计一个 split。返回 (box_rows, error_rows, split_inventory)。

    dst_size=None 按原图几何统计(src 组); dst_size=N 模拟训练 OBBResize 的
    非等比拉伸(N/N): YOLO-OBB 归一化坐标直接乘 N, DOTA 像素坐标按
    (N/orig_w, N/orig_h) 缩放——与 engine/data/transforms/obb_transforms.py
    的 sx/sy 独立缩放同口径, 角度/长宽比会随拉伸改变。
    """
    cat2id = {name: i for i, name in enumerate(classes)}
    image_names = list_images(images_dir)
    image_paths = [os.path.join(images_dir, n) for n in image_names]
    label_names = sorted(f for f in os.listdir(labels_dir) if f.endswith(".txt"))
    image_stems = {os.path.splitext(n)[0] for n in image_names}

    sizes, failed_images = read_image_sizes(image_paths, num_workers)
    failed_images = set(failed_images)

    errors: list[dict] = []
    boxes: list[dict] = []
    per_image: dict[str, list[dict]] = {}
    orphan_images = [n for n in image_names if os.path.splitext(n)[0] + ".txt" not in set(label_names)]
    label_stems_used: set[str] = set()
    manifest_missing: list[str] = []
    class_map_unmapped: set[str] = set()
    out_of_range_boxes = 0

    # 有标注但图片缺失 -> image_missing (不静默跳过)
    for label_name in label_names:
        stem = os.path.splitext(label_name)[0]
        if stem not in image_stems:
            _emit_later = {
                "split": split, "error_type": "image_missing",
                "message": "标注文件存在但对应图片不存在",
                "label_path": os.path.join(labels_dir, label_name),
                "image_path": os.path.join(images_dir, stem),
                "line_no": 0, "raw_line": "",
            }
            errors.append(_emit_later)

    def _emit(err_type: str, message: str, label_path: str, image_path: str, line_no: int, raw_line: str):
        errors.append({
            "split": split, "error_type": err_type, "message": message,
            "label_path": label_path, "image_path": image_path,
            "line_no": line_no, "raw_line": raw_line.strip(),
        })

    for img_name in image_names:
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(images_dir, img_name)
        label_name = stem + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        if label_name not in label_names:
            continue  # 无标注图片 -> orphan, 不进 errors(非标注错误), 计入 inventory
        label_stems_used.add(stem)

        if img_path in failed_images:
            _emit("image_unreadable", "图片尺寸读取失败", label_path, img_path, 0, "")
            continue
        img_w, img_h = sizes[img_path]
        eff_w = dst_size if dst_size else img_w
        eff_h = dst_size if dst_size else img_h
        diag = math.hypot(eff_w, eff_h)

        meta = manifest.get(stem)
        if meta is None:
            manifest_missing.append(stem)
            meta = {"case": "", "line": "", "tower": ""}

        img_boxes: list[dict] = []
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line_no, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            try:
                if fmt == "YOLO-OBB":
                    raw_cls, poly = parse_yolo_obb_line(raw, len(classes))
                    poly_abs = np.empty_like(poly)
                    poly_abs[:, 0] = poly[:, 0] * eff_w
                    poly_abs[:, 1] = poly[:, 1] * eff_h
                else:
                    raw_cls, poly_abs = parse_dota_line(raw, cat2id)
                    poly_abs = poly_abs.copy()
                    poly_abs[:, 0] *= eff_w / img_w
                    poly_abs[:, 1] *= eff_h / img_h
            except AnnotationError as e:
                _emit(e.error_type, e.message, label_path, img_path, line_no, raw)
                continue

            feat = obb_features(poly_abs)
            if feat["width"] <= 0:
                _emit("zero_width", f"短边为 0 (length={feat['length']:.2f})", label_path, img_path, line_no, raw)
                continue
            if fmt == "YOLO-OBB":
                coords = np.array([float(p) for p in raw.strip().split()[1:9]])
                if coords.min() < -0.01 or coords.max() > 1.01:
                    out_of_range_boxes += 1

            class_name = classes[raw_cls]
            business, cid = class_map.get(class_name, (class_name, str(raw_cls)))
            if class_map and class_name not in class_map:
                class_map_unmapped.add(class_name)

            img_boxes.append({
                "split": split,
                "image_path": img_path,
                "image_stem": stem,
                "image_width": eff_w,
                "image_height": eff_h,
                "orig_image_width": img_w,
                "orig_image_height": img_h,
                "raw_class_id": raw_cls,
                "class_name": class_name,
                "business_class": business,
                "cid": cid,
                "cx": feat["cx"], "cy": feat["cy"],
                "cx_norm": feat["cx"] / eff_w, "cy_norm": feat["cy"] / eff_h,
                "length_px": feat["length"], "width_px": feat["width"],
                "length_norm": feat["length"] / diag, "width_norm": feat["width"] / diag,
                "aspect_ratio": feat["length"] / feat["width"],
                "angle_deg": math.degrees(feat["angle_rad"]),
                "angle_rad": feat["angle_rad"],
                "area_px": feat["area_px"],
                "area_ratio": feat["area_px"] / float(eff_w * eff_h),
                "objects_in_image": 0,  # 两遍填充
                "case": meta["case"], "line": meta["line"], "tower": meta["tower"],
            })
        for b in img_boxes:
            b["objects_in_image"] = len(img_boxes)
        boxes.extend(img_boxes)
        per_image[stem] = img_boxes

    n_img = len(label_stems_used)
    counts_per_img = [len(v) for v in per_image.values()]
    inventory = {
        "group": "src" if dst_size is None else f"dst_{dst_size}",
        "dst_size": dst_size,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "image_count": len(image_names),
        "label_count": len(label_names),
        "labeled_image_count": n_img,
        "orphan_image_count": len(orphan_images),
        "orphan_images": orphan_images,
        "box_count": len(boxes),
        "images_fingerprint": dir_fingerprint(images_dir, image_names, hash_content=False),
        "labels_fingerprint": dir_fingerprint(labels_dir, label_names, hash_content=True),
        "objects_per_image": percentile_stats(counts_per_img) if counts_per_img else
        {"count": 0, **{k: None for k in STAT_FIELDS}},
        "empty_label_images": sum(1 for c in counts_per_img if c == 0),
        "single_object_images": sum(1 for c in counts_per_img if c == 1),
        "multi_object_images": sum(1 for c in counts_per_img if c > 1),
        "manifest_missing_stems": manifest_missing,
        "class_map_unmapped_names": sorted(class_map_unmapped),
        "out_of_range_boxes": out_of_range_boxes,
    }
    return boxes, errors, inventory


def build_class_summary(box_rows: list[dict], classes: list[str]) -> list[dict]:
    """按 split×类别 汇总特征分位数。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for b in box_rows:
        groups.setdefault((b["split"], b["class_name"]), []).append(b)

    rows: list[dict] = []
    for (split, cname) in sorted(groups):
        grp = groups[(split, cname)]
        row: dict = {
            "split": split,
            "class_name": cname,
            "cid": grp[0]["cid"],
            "box_count": len(grp),
            "image_count": len({b["image_stem"] for b in grp}),
        }
        for metric in BOX_METRICS:
            stats = percentile_stats([b[metric] for b in grp])
            for k in STAT_FIELDS:
                row[f"{metric}_{k}"] = stats[k]
        rows.append(row)
    return rows


def build_image_counts(box_rows: list[dict]) -> list[dict]:
    """每图每类一行(长表); 空标注图不出现(其在 boxes 中无行, 见 summary.empty_label_images)。"""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for b in box_rows:
        key = (b["split"], b["image_stem"], b["class_name"])
        groups.setdefault(key, []).append(b)

    rows = []
    for (split, stem, cname) in sorted(groups):
        grp = groups[(split, stem, cname)]
        head = grp[0]
        rows.append({
            "split": split,
            "image_stem": stem,
            "image_path": head["image_path"],
            "case": head["case"], "line": head["line"], "tower": head["tower"],
            "cid": head["cid"],
            "class_name": cname,
            "box_count": len(grp),
            "image_total_boxes": head["objects_in_image"],
        })
    return rows


def build_scene_summary(box_rows: list[dict]) -> dict:
    """测试场景级特征: 按 split×case 聚合(需 manifest 提供 case)。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for b in box_rows:
        if not b["case"]:
            continue
        groups.setdefault((b["split"], b["case"]), []).append(b)
    summary: dict = {}
    for (split, case) in sorted(groups):
        grp = groups[(split, case)]
        summary[f"{split}|{case}"] = {
            "box_count": len(grp),
            "image_count": len({b["image_stem"] for b in grp}),
            **{m: percentile_stats([b[m] for b in grp]) for m in BOX_METRICS},
        }
    return summary


def compare_splits(box_rows: list[dict]) -> dict:
    """训练集与其他 split 的分布差异(按类别的 mean/P25/P75 对比)。"""
    by_split_class: dict[tuple[str, str], list[dict]] = {}
    for b in box_rows:
        by_split_class.setdefault((b["split"], b["class_name"]), []).append(b)

    splits = sorted({b["split"] for b in box_rows})
    comparison: dict = {}
    for ref in [s for s in splits if s != "train"]:
        diff: dict = {}
        classes = sorted({c for (s, c) in by_split_class if s in ("train", ref)})
        for cname in classes:
            tr = by_split_class.get(("train", cname), [])
            ot = by_split_class.get((ref, cname), [])
            if not tr or not ot:
                diff[cname] = {"note": f"train={len(tr)}框 / {ref}={len(ot)}框, 不足对比"}
                continue
            entry = {"train_box_count": len(tr), f"{ref}_box_count": len(ot)}
            for metric in BOX_METRICS:
                tr_s = percentile_stats([b[metric] for b in tr])
                ot_s = percentile_stats([b[metric] for b in ot])
                entry[metric] = {
                    "train_mean": tr_s["mean"], f"{ref}_mean": ot_s["mean"],
                    "mean_diff": ot_s["mean"] - tr_s["mean"],
                    "train_p25": tr_s["p25"], f"{ref}_p25": ot_s["p25"],
                    "train_p75": tr_s["p75"], f"{ref}_p75": ot_s["p75"],
                }
            diff[cname] = entry
        comparison[f"train_vs_{ref}"] = diff
    return comparison


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_csv(path: str, columns: list[str], rows: list[dict]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in columns])


def write_errors_jsonl(path: str, errors: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")


# 无量纲指标: src 与 dst 值域相同, 上下面板可共享 x 轴;
# 像素量纲指标(长/宽/面积): 两组 x 轴量纲不同(原图 px vs 目标尺寸 px), 各自范围
DIMENSIONLESS_METRICS = {"aspect_ratio", "angle_deg"}


def write_plots(out_dir: str, box_rows: list[dict], ref_rows: list[dict] | None = None,
                ref_label: str = "src") -> list[str]:
    """分布图: plots/<split>/<class_name>__<metric>.png + split 汇总对比图。

    柱体均带数值标签; 直方图 bins=20、加宽画幅以留出标签空间。

    ref_rows 非 None 时(dst 组)在图中附加 src 参考: src 柱状图直接贴在 dst
    柱状图下方(上下两面板, y 轴共享同一刻度区间, 柱高直接可比):
      - 无量纲度量(aspect_ratio/angle_deg): 两组值域相同, 额外共享 x 轴;
      - 像素量纲度量(length/width/area): x 轴各自量纲与范围, bins 数相同。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DST_COLOR, REF_COLOR = "#4C72B0", "#C44E52"

    def _annotate_bars(ax, bars, fmt="{:.0f}", fontsize=8):
        for rect in bars:
            h = rect.get_height()
            if h <= 0:
                continue
            ax.annotate(
                fmt.format(h),
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=fontsize,
            )

    def _ref_vals(split, cname, metric):
        if not ref_rows:
            return []
        grp = [b for b in ref_rows
               if b["split"] == split and (cname == "__ALL__" or b["class_name"] == cname)]
        return [b[metric] for b in grp]

    # 聚合网格布局: 第一行 angle, area(+信息格), 第二行 aspect, length, width
    GRID_ROW1 = ("angle_deg", "area_px")
    GRID_ROW2 = ("aspect_ratio", "length_px", "width_px")
    METRIC_SHORT = {
        "length_px": "length (px)", "width_px": "width (px)",
        "aspect_ratio": "aspect ratio", "angle_deg": "angle (deg)",
        "area_px": "area (px^2)",
    }

    def _auto_bin_count(data_lists, metric):
        """按维度数据自适应区间数(跨维度各不相同); 成对图用联合数据定数,
        保证同维度 src/dst 区间数量一致。angle 值域固定 [0,180) 用 20 格。"""
        if metric == "angle_deg":
            return 20
        data = np.concatenate([np.asarray(d) for d in data_lists if len(d)])
        edges = np.histogram_bin_edges(data, bins="auto")
        return int(max(10, min(len(edges) - 1, 40)))

    def _grid_plot(split, cname):
        """单类(含 __ALL__)的 2x3 特征聚合图; dst 组时 src 参考成对贴在每行下方。"""
        rows_c = rows if cname == "__ALL__" else [b for b in rows if b["class_name"] == cname]
        n_rows = 4 if ref_rows else 2
        fig, axes = plt.subplots(n_rows, 3, figsize=(16, 3.2 * n_rows))
        axes = np.atleast_2d(axes)

        pair_top = 0  # noqa: F841 (保留占位, 布局由上方 r 表达式决定)
        for r, grid_row in ((0, GRID_ROW1), (2 if ref_rows else 1, GRID_ROW2)):
            for c, metric in enumerate(grid_row):
                vals = [b[metric] for b in rows_c]
                ref_vals = _ref_vals(split, cname, metric)
                ax_d, ax_s = axes[r, c], (axes[r + 1, c] if ref_rows else None)
                dimless = metric in DIMENSIONLESS_METRICS
                # 区间数按本维度数据自适应(跨维度各不相同), 同维度 src/dst 一致
                data_lists = [vals] + ([ref_vals] if ref_rows else [])
                n_bins = _auto_bin_count(data_lists, metric)

                if dimless and ref_vals:
                    lo = min(min(vals), min(ref_vals))
                    hi = max(max(vals), max(ref_vals))
                    if lo == hi:
                        hi = lo + 1.0
                    bins = np.linspace(lo, hi, n_bins + 1)
                    cnt_d, _, _ = ax_d.hist(vals, bins=bins, color=DST_COLOR, edgecolor="white")
                    cnt_s, _, _ = ax_s.hist(ref_vals, bins=bins, color=REF_COLOR, edgecolor="white")
                    ymax = max(cnt_d.max(), cnt_s.max())
                    # 同域度量: y 同刻度逐柱可比
                    ax_d.set_ylim(0, ymax * 1.12)
                    ax_s.set_ylim(0, ymax * 1.12)
                elif ref_vals:
                    # 像素量纲: x 区间数量相同但边界各自, y 各自自适应
                    lo_d, hi_d = min(vals), max(vals)
                    if lo_d == hi_d:
                        hi_d = lo_d + 1.0
                    lo_s, hi_s = min(ref_vals), max(ref_vals)
                    if lo_s == hi_s:
                        hi_s = lo_s + 1.0
                    ax_d.hist(vals, bins=np.linspace(lo_d, hi_d, n_bins + 1),
                              color=DST_COLOR, edgecolor="white")
                    ax_s.hist(ref_vals, bins=np.linspace(lo_s, hi_s, n_bins + 1),
                              color=REF_COLOR, edgecolor="white")
                else:
                    lo, hi = min(vals), max(vals)
                    if lo == hi:
                        hi = lo + 1.0
                    ax_d.hist(vals, bins=np.linspace(lo, hi, n_bins + 1),
                              color=DST_COLOR, edgecolor="white")
                if ax_s is not None:
                    ax_s.set_title(f"{METRIC_SHORT[metric]} — {ref_label} ref (n={len(ref_vals)})")
                ax_d.set_title(f"{METRIC_SHORT[metric]} — this group (n={len(vals)})")
                ax_d.set_xlabel(f"{METRIC_SHORT[metric]} ({n_bins} bins)")
                if ax_s is not None:
                    ax_s.set_xlabel(f"{METRIC_SHORT[metric]} ({n_bins} bins)")
            pair_top = r

        # y 轴行标签(兼作组名)
        axes[0, 0].set_ylabel("this group\nbox count")
        if ref_rows:
            axes[1, 0].set_ylabel(f"{ref_label} ref\nbox count")
            axes[2, 0].set_ylabel("this group\nbox count")
            axes[3, 0].set_ylabel(f"{ref_label} ref\nbox count")

        # 信息面板: 仅第一行第三格(第二特征行第三格是 width, 不可占用)
        ax_info = axes[0, 2]
        ax_info.axis("off")
        ax_info.text(
            0.5, 0.5,
            f"split: {split}\nclass: {cname}\nboxes: {len(rows_c)}\n"
            f"images: {len({b['image_stem'] for b in rows_c})}",
            ha="center", va="center", fontsize=12, color="#444444",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F2F2F2", "edgecolor": "#BBBBBB"},
        )
        if ref_rows:
            axes[1, 2].axis("off")

        fig.suptitle(f"{split} / {cname} / feature grid", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(split_dir, f"{cname}__grid.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return os.path.relpath(path, out_dir)

    written: list[str] = []
    by_split: dict[str, list[dict]] = {}
    for b in box_rows:
        by_split.setdefault(b["split"], []).append(b)

    plot_metrics = ("length_px", "width_px", "aspect_ratio", "angle_deg", "area_px")
    for split in sorted(by_split):
        rows = by_split[split]
        split_dir = os.path.join(out_dir, "plots", split)
        os.makedirs(split_dir, exist_ok=True)
        classes = ["__ALL__"] + sorted({b["class_name"] for b in rows})
        for cname in classes:
            grp = rows if cname == "__ALL__" else [b for b in rows if b["class_name"] == cname]
            for metric in plot_metrics:
                vals = [b[metric] for b in grp]
                ref_vals = _ref_vals(split, cname, metric)
                base_title = f"{split} / {cname} / {METRIC_LABELS[metric]}"
                path = os.path.join(split_dir, f"{cname}__{metric}.png")

                if ref_vals:
                    # 上下堆叠: 上=本组(dst), 下=src 参考。
                    # 区间数量按本维度联合数据自适应(跨维度各不相同),
                    # 同维度 src/dst 区间数量保持一致;
                    # 无量纲度量: 同域, 共享 x 轴与联合 bins, y 轴同刻度, 逐柱可比;
                    # 像素量纲度量: 边界按各自范围划分, y 轴(计数刻度)各自自适应——
                    #   共享会把同批框的最高柱画得一样满, 弱化分布形状差异。
                    dimless = metric in DIMENSIONLESS_METRICS
                    n_bins = _auto_bin_count([vals, ref_vals], metric)
                    if dimless:
                        lo = min(min(vals), min(ref_vals))
                        hi = max(max(vals), max(ref_vals))
                        if lo == hi:
                            hi = lo + 1.0
                        bins = np.linspace(lo, hi, n_bins + 1)
                        bins_d = bins_s = bins
                    else:
                        lo_d, hi_d = min(vals), max(vals)
                        if lo_d == hi_d:
                            hi_d = lo_d + 1.0
                        lo_s, hi_s = min(ref_vals), max(ref_vals)
                        if lo_s == hi_s:
                            hi_s = lo_s + 1.0
                        bins_d = np.linspace(lo_d, hi_d, n_bins + 1)
                        bins_s = np.linspace(lo_s, hi_s, n_bins + 1)
                    fig, (ax_d, ax_s) = plt.subplots(
                        2, 1, figsize=(9, 8), sharex=dimless, sharey=dimless,
                    )
                    bars_d = ax_d.hist(
                        vals, bins=bins_d, color=DST_COLOR, edgecolor="white",
                    )[2]
                    ax_d.margins(y=0.15)
                    _annotate_bars(ax_d, bars_d)
                    ax_d.set_title(f"this group ({len(vals)} boxes)")
                    ax_d.set_ylabel("box count")

                    bars_s = ax_s.hist(
                        ref_vals, bins=bins_s, color=REF_COLOR, edgecolor="white",
                    )[2]
                    ax_s.margins(y=0.15)
                    _annotate_bars(ax_s, bars_s)
                    ax_s.set_title(f"{ref_label} ref ({len(ref_vals)} boxes)")
                    ax_s.set_ylabel("box count")
                    ax_s.set_xlabel(f"{METRIC_LABELS[metric]} ({n_bins} bins)")
                    fig.suptitle(base_title)
                else:
                    n_bins = _auto_bin_count([vals], metric)
                    fig, ax = plt.subplots(figsize=(9, 5))
                    bars = ax.hist(vals, bins=n_bins, color=DST_COLOR, edgecolor="white")[2]
                    ax.margins(y=0.15)
                    _annotate_bars(ax, bars)
                    ax.set_title(base_title)
                    ax.set_xlabel(f"{METRIC_LABELS[metric]} ({n_bins} bins)")
                    ax.set_ylabel("box count")

                fig.tight_layout()
                fig.savefig(path, dpi=120)
                plt.close(fig)
                written.append(os.path.relpath(path, out_dir))
            # 本类 2x3 特征聚合图
            written.append(_grid_plot(split, cname))
        # 每类框数柱状图(src 与 dst 为同一批框, 计数恒相同, 无需对比)
        counts = {}
        for b in rows:
            counts[b["class_name"]] = counts.get(b["class_name"], 0) + 1
        fig, ax = plt.subplots(figsize=(10, 5))
        names = sorted(counts)
        bars = ax.bar(names, [counts[n] for n in names], width=0.8, color="#55A868")
        ax.margins(y=0.15)
        _annotate_bars(ax, bars, fontsize=11)
        ax.set_title(f"{split} / boxes per class")
        ax.set_ylabel("box count")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = os.path.join(split_dir, "__class_box_count__.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(os.path.relpath(path, out_dir))

    # train vs 其他 split 的 ALL 级对比(组内 split 间对比, 不涉及 src/dst)
    splits = sorted(by_split)
    for ref in [s for s in splits if s != "train" and "train" in by_split]:
        fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4.5 * len(plot_metrics), 4))
        for ax, metric in zip(axes, plot_metrics):
            for s, color in (("train", "#4C72B0"), (ref, "#C44E52")):
                vals = [b[metric] for b in by_split[s]]
                ax.hist(vals, bins=20, alpha=0.55, density=True, label=s, color=color, edgecolor="white")
            ax.set_title(METRIC_LABELS[metric], fontsize=10)
            ax.legend(fontsize=8)
        fig.suptitle(f"train vs {ref} (all classes, density)")
        fig.tight_layout()
        path = os.path.join(out_dir, "plots", f"train_vs_{ref}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(os.path.relpath(path, out_dir))
    return written


def summary_json_path(*parts: str) -> str:
    return ".".join(parts)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _rmtree_retry(path: str, attempts: int = 3):
    """WSL/网络盘句柄延迟释放会让 rmtree 偶发 PermissionError, 重试解决。"""
    import shutil
    import time

    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if i == attempts - 1:
                # 最后一次仍失败: 空目录留给下次清理, 非空目录尽力删
                if not os.listdir(path):
                    return
                shutil.rmtree(path, ignore_errors=True)
                return
            time.sleep(0.5)


def _clean_output_dir(out_dir: str):
    """清理旧产物: src/ dst_*/ plots/ 与顶层旧版输出文件(重跑不留残留)。"""
    if not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        if (name in ("plots", "src") or name.startswith("dst_")
                or name in ("feature_boxes.csv", "feature_image_counts.csv",
                            "feature_class_summary.csv", "feature_summary.json",
                            "errors.jsonl")):
            path = os.path.join(out_dir, name)
            if os.path.isdir(path):
                _rmtree_retry(path)
            else:
                os.remove(path)


def run(args: argparse.Namespace) -> dict:
    classes = load_classes(args.classes_file)
    class_map = load_class_map(args.class_map) if args.class_map else {}
    manifest = load_manifest(args.manifest) if args.manifest else {}

    split_dirs: list[tuple[str, str, str]] = []  # (split, images_dir, labels_dir)
    if args.images_dir and args.labels_dir:
        name = args.split_name or "custom"
        split_dirs.append((name, args.images_dir, args.labels_dir))
    else:
        splits = args.splits or discover_splits(args.data_root)
        if not splits:
            raise SystemExit(
                f"未在 {args.data_root}/images 下发现任何 split(需存在对应 labels/<split>)，"
                "或用 --images-dir/--labels-dir 显式指定"
            )
        for s in splits:
            split_dirs.append((
                s,
                os.path.join(args.data_root, "images", s),
                os.path.join(args.data_root, "labels", s),
            ))
    for split, img_dir, lbl_dir in split_dirs:
        if not os.path.isdir(img_dir):
            raise SystemExit(f"图片目录不存在: {img_dir}")
        if not os.path.isdir(lbl_dir):
            raise SystemExit(f"标注目录不存在: {lbl_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    _clean_output_dir(args.output_dir)

    groups: list[tuple[str, int | None]] = [("src", None), (f"dst_{args.dst_sz}", args.dst_sz)]
    summaries: dict[str, dict] = {}
    src_boxes: list[dict] | None = None  # dst 组绘图的 src 参考
    for group_name, dst_size in groups:
        group_dir = os.path.join(args.output_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        print(f"[dataset_feature_stats] ===== 组 {group_name} "
              f"(dst_size={dst_size if dst_size else '原图'}) =====")
        all_boxes: list[dict] = []
        all_errors: list[dict] = []
        inventories: dict[str, dict] = {}
        for split, img_dir, lbl_dir in split_dirs:
            print(f"[dataset_feature_stats] 分析 split={split}: {img_dir}")
            boxes, errors, inv = analyze_split(
                split, img_dir, lbl_dir, classes, args.format, class_map, manifest,
                args.num_workers, dst_size=dst_size,
            )
            all_boxes.extend(boxes)
            all_errors.extend(errors)
            inventories[split] = inv
            print(
                f"[dataset_feature_stats]   images={inv['image_count']} labels={inv['label_count']} "
                f"boxes={inv['box_count']} errors={len(errors)}"
            )

        class_summary = build_class_summary(all_boxes, classes)
        image_counts = build_image_counts(all_boxes)
        scene_summary = build_scene_summary(all_boxes)
        comparison = compare_splits(all_boxes)

        write_csv(
            os.path.join(group_dir, "feature_boxes.csv"),
            BOX_COLUMNS, all_boxes,
        )
        write_csv(
            os.path.join(group_dir, "feature_image_counts.csv"),
            ["split", "image_stem", "image_path", "case", "line", "tower", "cid",
             "class_name", "box_count", "image_total_boxes"],
            image_counts,
        )
        sum_cols = ["split", "class_name", "cid", "box_count", "image_count"]
        for m in BOX_METRICS:
            sum_cols += [f"{m}_{k}" for k in STAT_FIELDS]
        write_csv(
            os.path.join(group_dir, "feature_class_summary.csv"),
            sum_cols, class_summary,
        )
        write_errors_jsonl(os.path.join(group_dir, "errors.jsonl"), all_errors)

        plots: list[str] = []
        if not args.no_plots and all_boxes:
            plots = write_plots(
                group_dir, all_boxes,
                ref_rows=src_boxes if dst_size is not None else None,
                ref_label="src",
            )

        error_counts = {t: 0 for t in ERROR_TYPES}
        for e in all_errors:
            error_counts[e["error_type"]] = error_counts.get(e["error_type"], 0) + 1

        total_label_lines = sum(inv["label_count"] for inv in inventories.values())
        summary = {
            "tool_version": TOOL_VERSION,
            "group": group_name,
            "dst_size": dst_size,
            "group_note": (
                "原图几何统计" if dst_size is None else
                f"训练预处理后统计: 按 OBBResize 口径非等比拉伸到 {dst_size}x{dst_size} "
                f"(sx={dst_size}/w, sy={dst_size}/h), 角度/长宽比为拉伸后的训练输入分布"
            ),
            "params": {
                "data_root": args.data_root,
                "splits": [s for s, _, _ in split_dirs],
                "classes_file": args.classes_file,
                "format": args.format,
                "class_map": args.class_map,
                "manifest": args.manifest,
                "num_workers": args.num_workers,
                "no_plots": bool(args.no_plots),
                "dst_sz": args.dst_sz,
            },
            "geometry_definitions": GEOMETRY_DEFINITIONS,
            "class_names": classes,
            "class_map": {k: {"business_class": v[0], "cid": v[1]} for k, v in class_map.items()},
            "inputs": {
                "classes_file_sha256": file_sha256(args.classes_file),
                "splits": inventories,
            },
            "split_summary": {
                s: {
                    "image_count": inv["image_count"],
                    "labeled_image_count": inv["labeled_image_count"],
                    "box_count": inv["box_count"],
                    "objects_per_image": inv["objects_per_image"],
                    "empty_label_images": inv["empty_label_images"],
                    "single_object_images": inv["single_object_images"],
                    "multi_object_images": inv["multi_object_images"],
                }
                for s, inv in inventories.items()
            },
            "class_summary": {
                f"{r['split']}|{r['class_name']}": r for r in class_summary
            },
            "scene_summary": scene_summary,
            "scene_summary_note": (
                "按 (split, case) 聚合; 未提供 --manifest 时为空对象"
            ),
            "split_distribution_comparison": comparison,
            "errors": {
                "total": len(all_errors),
                "by_type": error_counts,
                "error_lines_file": "errors.jsonl",
                "error_ratio_vs_label_files": (
                    len(all_errors) / total_label_lines if total_label_lines else None
                ),
            },
            "outputs": {
                "feature_boxes.csv": len(all_boxes),
                "feature_image_counts.csv": len(image_counts),
                "feature_class_summary.csv": len(class_summary),
                "plots": plots,
            },
        }
        with open(os.path.join(group_dir, "feature_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        summaries[group_name] = summary
        if dst_size is None:
            src_boxes = all_boxes
        print(
            f"[dataset_feature_stats] 组 {group_name} 输出: {os.path.abspath(group_dir)} "
            f"(boxes={len(all_boxes)} errors={len(all_errors)} plots={len(plots)})"
        )

    print(f"[dataset_feature_stats] 输出目录: {os.path.abspath(args.output_dir)}")
    return summaries


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OBB 数据集特征统计工具(训练前数据分析)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=None,
                        help="YOLO-OBB 布局根目录(images/<split> 与 labels/<split>)")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="要统计的 split, 默认自动发现 images/ 下子目录")
    parser.add_argument("--classes-file", default=None,
                        help="类别文件(每行一个类名), 默认 <data-root>/classes.txt")
    parser.add_argument("--format", default="YOLO-OBB", choices=["YOLO-OBB", "DOTA"],
                        help="标注格式")
    parser.add_argument("--class-map", default=None,
                        help="可选 CSV(raw_name,business_name,cid): raw 类名 -> 业务类/cid")
    parser.add_argument("--manifest", default=None,
                        help="可选 CSV(image_stem,case,line,tower): 图片 -> 场景/线路/杆塔")
    parser.add_argument("--output-dir", default="outputs/dataset_feature_stats",
                        help="输出目录(自动创建)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="图片尺寸读取线程数")
    parser.add_argument("--dst_sz", type=int, default=640,
                        help="预处理目标尺寸 N(NxN 非等比拉伸, 与训练 OBBResize 同口径), "
                             "预处理后统计输出到 <output-dir>/dst_<N>/")
    parser.add_argument("--no-plots", action="store_true", help="跳过分布图生成")
    parser.add_argument("--images-dir", default=None,
                        help="自定义图片目录(与 --labels-dir 连用)")
    parser.add_argument("--labels-dir", default=None,
                        help="自定义标注目录(与 --images-dir 连用)")
    parser.add_argument("--split-name", default=None,
                        help="自定义目录模式的 split 名, 默认 'custom'")
    args = parser.parse_args(argv)

    if args.images_dir or args.labels_dir:
        if not (args.images_dir and args.labels_dir):
            raise SystemExit("--images-dir 与 --labels-dir 必须同时提供")
    else:
        if not args.data_root:
            raise SystemExit("需要 --data-root 或 (--images-dir + --labels-dir)")
        if not args.classes_file:
            args.classes_file = os.path.join(args.data_root, "classes.txt")
        if not os.path.isfile(args.classes_file):
            raise SystemExit(f"classes 文件不存在: {args.classes_file}")
    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
