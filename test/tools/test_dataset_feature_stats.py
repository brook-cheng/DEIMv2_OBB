"""tools/dataset_feature_stats.py 的单元测试.

覆盖:
    1. obb_features 几何口径与训练实现(engine/deim/obb_geometry.py)的对拍;
    2. YOLO-OBB / DOTA 标注行解析的正常与异常路径;
    3. 端到端 run(): 合成数据集的输出文件、错误分类计数、汇总统计、关联键列、
       manifest/class_map 注入、相同输入输出确定性。

Run:
    pytest test/tools/test_dataset_feature_stats.py -v
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_feature_stats import (  # noqa: E402
    AnnotationError,
    analyze_split,
    obb_features,
    parse_dota_line,
    parse_yolo_obb_line,
    parse_args,
    percentile_stats,
    run,
    write_plots,
)


def test_percentile_stats_multisample():
    """多样本分位数映射回归: median 与 p25 曾互换(单样本测试测不出)。"""
    vals = list(range(1, 101))  # 1..100
    s = percentile_stats(vals)
    assert s["count"] == 100
    assert s["mean"] == pytest.approx(50.5)
    assert s["median"] == pytest.approx(50.5)
    assert s["p5"] == pytest.approx(5.95)
    assert s["p25"] == pytest.approx(25.75)
    assert s["p75"] == pytest.approx(75.25)
    assert s["p95"] == pytest.approx(95.05)
    assert s["min"] == 1 and s["max"] == 100
    assert s["p25"] < s["median"] < s["p75"]
    empty = percentile_stats([])
    assert empty["count"] == 0 and empty["median"] is None

# --------------------------------------------------------------------------- #
# 几何口径
# --------------------------------------------------------------------------- #
def test_obb_features_axis_aligned():
    # 200x100 图内 100x50 轴对齐矩形 -> length=100, width=50, ar=2, angle=0
    poly = np.array([[50, 20], [150, 20], [150, 70], [50, 70]], dtype=np.float64)
    f = obb_features(poly)
    assert f["length"] == pytest.approx(100.0)
    assert f["width"] == pytest.approx(50.0)
    assert f["cx"] == pytest.approx(100.0)
    assert f["cy"] == pytest.approx(45.0)
    assert f["area_px"] == pytest.approx(5000.0)
    assert f["angle_rad"] == pytest.approx(0.0)


def test_obb_features_rotated_45deg():
    # 对角放置的正方形: 边长 100*sqrt(2), 长边方向 45deg
    poly = np.array([[150, 50], [250, 150], [150, 250], [50, 150]], dtype=np.float64)
    f = obb_features(poly)
    assert f["length"] == pytest.approx(100 * math.sqrt(2))
    assert f["width"] == pytest.approx(100 * math.sqrt(2))
    assert math.degrees(f["angle_rad"]) == pytest.approx(45.0)
    assert f["area_px"] == pytest.approx(20000.0)


def test_obb_features_matches_training_impl():
    """与 engine/deim/obb_geometry.py::xyxyxyxy_to_xywhr 逐框对拍(随机旋转矩形)。"""
    torch = pytest.importorskip("torch")
    from engine.deim.obb_geometry import xyxyxyxy_to_xywhr

    rng = np.random.default_rng(0)
    for _ in range(50):
        cx, cy = rng.uniform(50, 400, 2)
        length, width = rng.uniform(10, 200, 2)
        theta = rng.uniform(0, math.pi)
        ux, uy = math.cos(theta), math.sin(theta)
        vx, vy = -math.sin(theta), math.cos(theta)
        pts = [
            (cx + ux * length / 2 + vx * width / 2, cy + uy * length / 2 + vy * width / 2),
            (cx - ux * length / 2 + vx * width / 2, cy - uy * length / 2 + vy * width / 2),
            (cx - ux * length / 2 - vx * width / 2, cy - uy * length / 2 - vy * width / 2),
            (cx + ux * length / 2 - vx * width / 2, cy + uy * length / 2 - vy * width / 2),
        ]
        poly = np.array(pts, dtype=np.float64)
        f = obb_features(poly)
        ref = xyxyxyxy_to_xywhr(torch.from_numpy(poly).float())
        assert f["length"] == pytest.approx(float(ref[2]), abs=1e-3), "length 与训练实现不一致"
        assert f["width"] == pytest.approx(float(ref[3]), abs=1e-3), "width 与训练实现不一致"
        assert f["angle_rad"] == pytest.approx(float(ref[4]), abs=1e-4), "angle 与训练实现不一致"


# --------------------------------------------------------------------------- #
# 标注解析
# --------------------------------------------------------------------------- #
def test_parse_yolo_obb_line_valid():
    cls, pts = parse_yolo_obb_line("1 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4", num_classes=3)
    assert cls == 1
    assert pts.shape == (4, 2)
    assert pts[0, 0] == pytest.approx(0.1)


def test_parse_yolo_obb_line_errors():
    with pytest.raises(AnnotationError) as e:
        parse_yolo_obb_line("1 0.1 0.2 0.3", num_classes=3)
    assert e.value.error_type == "parse_error"
    with pytest.raises(AnnotationError) as e:
        parse_yolo_obb_line("1 a b c d e f g h", num_classes=3)
    assert e.value.error_type == "parse_error"
    with pytest.raises(AnnotationError) as e:
        parse_yolo_obb_line("9 0 0 0.1 0 0.1 0.1 0 0.1", num_classes=3)
    assert e.value.error_type == "class_unmapped"


def test_parse_dota_line_valid_and_unmapped():
    cat2id = {"plane": 0, "ship": 1}
    cls, pts = parse_dota_line("10 20 110 20 110 70 10 70 plane 0", cat2id)
    assert cls == 0 and pts.shape == (4, 2)
    with pytest.raises(AnnotationError) as e:
        parse_dota_line("10 20 110 20 110 70 10 70 unknown", cat2id)
    assert e.value.error_type == "class_unmapped"


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synthetic_dataset(tmp_path):
    """2 类 / 3 split 内构造: 正常框 + 4 类异常 + 空标注负样本图 + 无图标注。"""
    from PIL import Image

    root = tmp_path / "ds"
    for sub in ("images/train", "labels/train"):
        (root / sub).mkdir(parents=True)
    (root / "classes.txt").write_text("cls_a\ncls_b\n", encoding="utf-8")

    def save_img(name, size):
        Image.new("RGB", size, (128, 128, 128)).save(root / "images" / "train" / name)

    # img_a: 200x100, 一个 100x50 轴对齐框(cls 0) + 一个 40x20 轴对齐框(cls 1)
    save_img("img_a.jpg", (200, 100))
    (root / "labels" / "train" / "img_a.txt").write_text(
        "0 0.25 0.2 0.75 0.2 0.75 0.7 0.25 0.7\n"
        "1 0.05 0.1 0.25 0.1 0.25 0.3 0.05 0.3\n"
        "0 0.1 0.2 0.3\n"                       # parse_error: 字段不足
        "9 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",  # class_unmapped
        encoding="utf-8",
    )
    # img_b: 300x300, 一个 45 度旋转框(cls 1) + 一个退化零宽框
    save_img("img_b.jpg", (300, 300))
    a, b = 50 / 300, 250 / 300
    (root / "labels" / "train" / "img_b.txt").write_text(
        f"1 {0.5:.6f} {a:.6f} {b:.6f} {0.5:.6f} {0.5:.6f} {b:.6f} {a:.6f} {0.5:.6f}\n"
        + "0 0.3333 0.3333 0.3333 0.3333 0.3333 0.3333 0.3333 0.3333\n",  # zero_width
        encoding="utf-8",
    )
    # img_empty: 空标注负样本
    save_img("img_empty.jpg", (100, 100))
    (root / "labels" / "train" / "img_empty.txt").write_text("", encoding="utf-8")
    # img_bad: 图片内容损坏 -> image_unreadable
    (root / "images" / "train" / "img_bad.jpg").write_bytes(b"\x00\x01 not an image")
    (root / "labels" / "train" / "img_bad.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8",
    )
    # 无图标注 -> image_missing
    (root / "labels" / "train" / "ghost.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8",
    )
    return root


def _read_csv(path):
    import csv
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_run_end_to_end(synthetic_dataset, tmp_path):
    out = tmp_path / "out"
    args = parse_args([
        "--data-root", str(synthetic_dataset),
        "--splits", "train",
        "--output-dir", str(out),
    ])
    summaries = run(args)
    assert set(summaries) == {"src", "dst_640"}
    summary = summaries["src"]

    # ---- 输出文件齐全(src 组与 dst_640 组结构相同)
    for group in ("src", "dst_640"):
        for name in ("feature_boxes.csv", "feature_image_counts.csv",
                     "feature_class_summary.csv", "feature_summary.json", "errors.jsonl"):
            assert (out / group / name).is_file(), f"缺少输出 {group}/{name}"

    boxes = _read_csv(out / "src" / "feature_boxes.csv")
    # img_a 2 框 + img_b 1 框 = 3 个有效框
    assert len(boxes) == 3

    # ---- 关联键列恒存在
    for key in ("split", "case", "line", "tower", "image_path", "cid"):
        assert key in boxes[0], f"关联键列缺失: {key}"

    # ---- 几何数值正确
    box_a = next(b for b in boxes if b["image_stem"] == "img_a" and b["raw_class_id"] == "0")
    assert box_a["length_px"] == pytest.approx("100.000000")
    assert box_a["width_px"] == pytest.approx("50.000000")
    assert box_a["aspect_ratio"] == pytest.approx("2.000000")
    assert float(box_a["angle_deg"]) == pytest.approx(0.0, abs=1e-4)
    assert float(box_a["area_px"]) == pytest.approx(5000.0, abs=1e-3)
    assert float(box_a["objects_in_image"]) == 2  # 同图另一有效框计入, 错误框不计入

    box_b = next(b for b in boxes if b["image_stem"] == "img_b")
    assert float(box_b["angle_deg"]) == pytest.approx(45.0, abs=1e-3)
    assert float(box_b["objects_in_image"]) == 1  # zero_width 框已入 errors, 不计数

    # ---- dst_640 组: 非等比拉伸后的几何(YOLO 归一化坐标 * 640)
    dst_boxes = _read_csv(out / "dst_640" / "feature_boxes.csv")
    dbox_a = next(b for b in dst_boxes if b["image_stem"] == "img_a" and b["raw_class_id"] == "0")
    # 200x100 图的 100x50 轴对齐框: x*3.2=320, y*6.4=320 -> 拉伸后为正方形
    assert float(dbox_a["image_width"]) == 640
    assert float(dbox_a["orig_image_width"]) == 200
    assert float(dbox_a["length_px"]) == pytest.approx(320.0, abs=1e-4)
    assert float(dbox_a["width_px"]) == pytest.approx(320.0, abs=1e-4)
    assert float(dbox_a["aspect_ratio"]) == pytest.approx(1.0, abs=1e-6)
    assert float(dbox_a["area_ratio"]) == pytest.approx(320 * 320 / (640 * 640), abs=1e-6)
    dbox_b = next(b for b in dst_boxes if b["image_stem"] == "img_b")
    # 300x300 等比图: 框按 640/300 缩放, 45 度角与长宽比保持
    assert float(dbox_b["angle_deg"]) == pytest.approx(45.0, abs=1e-3)
    assert float(dbox_b["aspect_ratio"]) == pytest.approx(1.0, abs=1e-6)
    assert summaries["dst_640"]["dst_size"] == 640
    assert "拉伸" in summaries["dst_640"]["group_note"]

    # ---- 错误分类与计数(不静默; src 与 dst_640 两组各自完整记录)
    with open(out / "src" / "errors.jsonl", "r", encoding="utf-8") as f:
        errors = [json.loads(line) for line in f]
    by_type = {}
    for e in errors:
        by_type.setdefault(e["error_type"], []).append(e)
    assert set(by_type) == {"parse_error", "class_unmapped", "zero_width",
                            "image_missing", "image_unreadable"}
    assert summary["errors"]["by_type"] == {t: len(by_type[t]) for t in by_type}
    assert summary["errors"]["total"] == len(errors) == 5

    # ---- 汇总统计
    inv = summary["inputs"]["splits"]["train"]
    assert inv["image_count"] == 4
    assert inv["label_count"] == 5  # img_a/img_b/img_empty/img_bad/ghost
    assert inv["box_count"] == 3
    assert inv["orphan_image_count"] == 0
    s = summary["split_summary"]["train"]
    assert s["empty_label_images"] == 1      # img_empty
    assert s["single_object_images"] == 1    # img_a
    assert s["multi_object_images"] == 1     # img_b

    cs = _read_csv(out / "src" / "feature_class_summary.csv")
    row_a = next(r for r in cs if r["class_name"] == "cls_a")
    assert int(row_a["box_count"]) == 1
    assert float(row_a["length_px_mean"]) == pytest.approx(100.0)
    assert float(row_a["length_px_p5"]) == pytest.approx(100.0)  # 单样本分位数 == 本身

    # 每图每类长表
    ic = _read_csv(out / "src" / "feature_image_counts.csv")
    assert {(r["image_stem"], r["class_name"]) for r in ic} == {
        ("img_a", "cls_a"), ("img_a", "cls_b"), ("img_b", "cls_b"),
    }


def test_run_manifest_and_class_map(synthetic_dataset, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_stem,case,line,tower\nimg_a,C01,L01,T01\n", encoding="utf-8",
    )
    class_map = tmp_path / "cmap.csv"
    class_map.write_text(
        "raw_name,business_name,cid\ncls_a,insulator,100\n", encoding="utf-8",
    )
    out = tmp_path / "out2"
    args = parse_args([
        "--data-root", str(synthetic_dataset),
        "--splits", "train",
        "--manifest", str(manifest),
        "--class-map", str(class_map),
        "--output-dir", str(out),
        "--no-plots",
    ])
    run(args)
    boxes = _read_csv(out / "src" / "feature_boxes.csv")
    box_a = next(b for b in boxes if b["image_stem"] == "img_a" and b["class_name"] == "cls_a")
    assert (box_a["case"], box_a["line"], box_a["tower"]) == ("C01", "L01", "T01")
    assert (box_a["business_class"], box_a["cid"]) == ("insulator", "100")
    # 未提供映射的类别回退为 identity
    box_b = next(b for b in boxes if b["image_stem"] == "img_b")
    assert box_b["business_class"] == "cls_b"
    assert box_b["cid"] == "1"


def test_run_deterministic(synthetic_dataset, tmp_path):
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    for out in (out1, out2):
        run(parse_args([
            "--data-root", str(synthetic_dataset), "--splits", "train",
            "--output-dir", str(out), "--no-plots",
        ]))
    for group in ("src", "dst_640"):
        for name in ("feature_boxes.csv", "feature_image_counts.csv",
                     "feature_class_summary.csv", "feature_summary.json", "errors.jsonl"):
            p1, p2 = out1 / group / name, out2 / group / name
            assert p1.read_bytes() == p2.read_bytes(), f"{group}/{name} 不确定"


def test_write_plots_with_src_ref(tmp_path):
    """dst 组绘图附加 src 参考: 无量纲同轴叠加 / 像素量纲双面板, 文件名保持不变。"""
    def _row(split, metric_vals, scale=1.0):
        return [{
            "split": split, "class_name": "cls_a", "image_stem": f"img_{i}",
            "length_px": v * scale, "width_px": v * scale * 0.5,
            "aspect_ratio": v / (v * 0.5), "angle_deg": v % 180.0,
            "area_px": (v * scale) ** 2,
        } for i, v in enumerate(metric_vals)]

    src_rows = _row("train", [10.0, 20.0, 30.0, 40.0], scale=3.0)
    dst_rows = _row("train", [10.0, 20.0, 30.0, 40.0], scale=1.0)
    out = tmp_path / "plots_out"
    written = write_plots(str(out), dst_rows, ref_rows=src_rows, ref_label="src")
    names = {os.path.basename(p) for p in written}
    # (每类含 __ALL__) x (5 metric + 1 聚合网格) + 类别柱状图; 单 split 无 train_vs 图
    expected = {"__class_box_count__.png"}
    for prefix in ("__ALL__", "cls_a"):
        for metric in ("length_px", "width_px", "aspect_ratio", "angle_deg", "area_px"):
            expected.add(f"{prefix}__{metric}.png")
        expected.add(f"{prefix}__grid.png")
    assert names == expected
    for p in written:
        assert (out / p).is_file() and (out / p).stat().st_size > 0


def test_analyze_split_direct(synthetic_dataset):
    """不经 CLI 直接调用 analyze_split, 校验返回结构。"""
    classes = ["cls_a", "cls_b"]
    boxes, errors, inv = analyze_split(
        "train",
        str(synthetic_dataset / "images" / "train"),
        str(synthetic_dataset / "labels" / "train"),
        classes, "YOLO-OBB", {}, {}, num_workers=2,
    )
    assert len(boxes) == 3
    assert len(errors) == 5
    assert inv["labeled_image_count"] == 4
    assert inv["class_map_unmapped_names"] == []
    # manifest 未提供 -> 成功解析标注的图记入 missing(img_bad 属 image_unreadable, 不在此列)
    assert set(inv["manifest_missing_stems"]) == {"img_a", "img_b", "img_empty"}
