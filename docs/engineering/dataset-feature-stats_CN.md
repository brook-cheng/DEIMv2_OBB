# 数据集特征统计工具（dataset_feature_stats）

训练开始前的数据分析工具：统计 train/val(/test) 各 split 的 OBB 标注特征，
输出按框、按图、按类别可关联的数据与分布图，作为后续模型差异分析的输入。

- 工具：`tools/dataset_feature_stats.py`
- 测试：`test/tools/test_dataset_feature_stats.py`（`pytest test/tools/test_dataset_feature_stats.py -v`）

## 使用方法

```bash
# 标准 YOLO-OBB 布局（<root>/images/<split> + <root>/labels/<split> + <root>/classes.txt）
python tools/dataset_feature_stats.py \
    --data-root /path/to/yolo_obb_root \
    --splits train val \
    --dst_sz 640 \
    --output-dir outputs/dataset_feature_stats

# 不指定 --splits 时自动发现 images/ 下有对应 labels/ 的子目录

# 带业务类别映射与场景清单（case/线路/杆塔）
python tools/dataset_feature_stats.py \
    --data-root /path/to/root --splits train test \
    --class-map class_map.csv \
    --manifest manifest.csv \
    --output-dir outputs/dataset_feature_stats

# 自定义单目录（非 data-root 布局）
python tools/dataset_feature_stats.py \
    --images-dir .../test_images --labels-dir .../test_labels \
    --classes-file classes.txt --split-name test \
    --format DOTA \
    --output-dir outputs/dataset_feature_stats
```

### 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--data-root` | YOLO-OBB 布局根目录 | 无（与 `--images-dir/--labels-dir` 二选一） |
| `--splits` | 要统计的 split 列表 | 自动发现 |
| `--classes-file` | 类别文件（每行一个类名） | `<data-root>/classes.txt` |
| `--format` | `YOLO-OBB` 或 `DOTA` | `YOLO-OBB` |
| `--dst_sz` | 预处理目标尺寸 N：额外统计"非等比拉伸到 N×N 后"的特征，输出到 `dst_<N>/` 组 | 640 |
| `--class-map` | CSV：`raw_name,business_name,cid` | 无（business 回退为 raw 类名） |
| `--manifest` | CSV：`image_stem,case,line,tower` | 无（case/line/tower 输出空串） |
| `--output-dir` | 输出目录（自动创建） | `outputs/dataset_feature_stats` |
| `--num-workers` | 图片尺寸读取线程数 | 8 |
| `--no-plots` | 跳过分布图 | 关 |

## 输出目录结构（两组，结构完全相同）

```
<output-dir>/
├── src/        # 原图几何统计
└── dst_640/    # --dst_sz 640: 训练预处理后统计
```

**dst 组的预处理口径**：与训练侧 `OBBResize`（`engine/data/transforms/obb_transforms.py`）
一致的非等比拉伸 `sx = N/w, sy = N/h`。YOLO-OBB 归一化坐标直接乘 N，
DOTA 像素坐标按 `(N/orig_w, N/orig_h)` 缩放。非等比拉伸会改变目标的角度与
长宽比（例如 1302×3264 的竖图缩到 640×640 后，y 方向压缩更狠），dst 组反映的
是模型训练输入中的真实目标分布。dst 组的 `feature_boxes.csv` 额外保留
`orig_image_width/orig_image_height` 两列，`image_width/height` 为目标尺寸。

## 标注格式（与 DotaDataset 解析口径一致）

- **YOLO-OBB**：`cls_id x1 y1 x2 y2 x3 y3 x4 y4`，坐标为归一化值（除以图宽/图高）。
- **DOTA**：`x1 y1 x2 y2 x3 y3 x4 y4 cls_name [difficult]`，坐标为绝对像素；
  末列为数字（difficult 标志）时类名取第 9 列起的非数字部分。

## 几何口径（固定）

与训练侧 `engine/deim/obb_geometry.py::xyxyxyxy_to_xywhr` 同口径，
单元测试中有与该实现的逐框对拍（随机旋转矩形上 length/width/angle 一致）。

| 字段 | 定义 |
|---|---|
| `length` | max(两条邻边长) |
| `width` | min(两条邻边长) |
| `aspect_ratio` | length / width（≥ 1） |
| `angle` | 长边方向与 x 轴夹角，atan2 后归一到 [0, π) rad，即 [0, 180) deg |
| `area_px` | 四点多边形鞋带面积（绝对值） |
| `area_ratio` | area_px / (image_width × image_height) |
| `cx_norm / cy_norm` | 中心点像素坐标 / 图宽 / 图高（YOLO 口径） |
| `length_norm / width_norm` | 边长像素值 / 图像对角线长 sqrt(w²+h²) |

## 输出文件（src/ 与 dst_<N>/ 两组内结构相同）

| 文件 | 内容 |
|---|---|
| `feature_boxes.csv` | 每个有效标注框一行（28 列，见下） |
| `feature_image_counts.csv` | 每图每类一行（长表）：split/image_stem/image_path/case/line/tower/cid/class_name/box_count/image_total_boxes |
| `feature_class_summary.csv` | 按 split × 类别：box_count、image_count 及 length/width/aspect_ratio/angle_deg/area_px/area_ratio 各 8 项分位统计（mean/median/p5/p25/p75/p95/min/max） |
| `feature_summary.json` | 组信息（group/dst_size/口径说明）、运行参数、几何口径、类别映射、输入文件身份指纹、每 split 样本统计（每图框数 mean/median/p95/max、单/多目标图数、空标注图数）、train vs 其他 split 分布差异、测试场景级(scene)汇总、错误统计 |
| `errors.jsonl` | 异常明细，每行一个错误 |
| `plots/<split>/*.png` | 每类（含 `__ALL__`）× 5 个度量的直方图（bins=20，柱顶标 count）；每类一张 **2×3 特征聚合网格图** `<class>__grid.png`（第一行 `angle, area`+信息面板，第二行 `aspect, length, width`）；每类框数柱状图（柱顶标数值）；`plots/train_vs_<ref>.png` 为 train 与对比 split 的密度叠加图 |

**聚合网格图**（`<class>__grid.png`，src 与 dst 两组均生成）：

- src 组：2×3 网格，每个格子是该类的单个特征直方图，第一行第三格为样本数信息面板；
- dst 组：4×3 网格，保持"src 贴在下方"的对比形式——每个特征格的下一行是同特征的
  src 参考直方图；angle/aspect 共享 x 轴与 y 刻度，像素量纲度量（area/length/width）
  区间数量相同（20 格）而边界与 y 刻度各自自适应。格子内不放数值标签（细节看单特征图）。

**dst 组图中的 src 对比参考**（预处理影响对比）：src 的柱状图直接贴在 dst 柱状图下方（同文件上下两面板，均带数值标签）：

- 无量纲度量（`aspect_ratio`、`angle_deg`）：两组值域相同，**共享 x 轴与联合 bins，y 轴共享同一刻度区间**，逐柱直接可比；
- 像素量纲度量（`length_px`、`width_px`、`area_px`）：x 轴**区间数量相同但边界按各自范围划分**，y 轴（计数刻度）**各自自适应**——共享 y 会把同一批框的最高柱画得一样满，弱化分布形状差异；各自刻度下集中度、偏态、次峰的差异更直观。

**区间数量按维度自适应**：bins 数由该维度数据经 numpy `auto` 规则（Freedman-Diaconis/Sturges）确定，限制在 10~40 之间；angle 值域固定 [0, 180) 取 20 格。跨维度各不相同（如 area 26 格、length 33 格），同一维度 src/dst 的区间数量保持一致；各图 x 轴标签标注实际 bins 数（如 `length (px) (33 bins)`）。

src 组的图保持纯 src（无叠加）。类别框数柱状图两组恒相同（同一批框），不做对比。

### feature_boxes.csv 列

`split, image_path, image_stem, image_width, image_height, orig_image_width,
orig_image_height, raw_class_id, class_name, business_class, cid, cx, cy,
cx_norm, cy_norm, length_px, width_px, length_norm, width_norm, aspect_ratio,
angle_deg, angle_rad, area_px, area_ratio, objects_in_image, case, line, tower`

**关联键**：`split / case / line / tower / image_path / cid` 恒存在；
未提供 `--manifest` 时 case/line/tower 为空串。

## 异常处理（不静默跳过）

| error_type | 触发条件 |
|---|---|
| `parse_error` | 标注行字段数不足或数值解析失败 |
| `class_unmapped` | 类别 id 越界 / DOTA 类名不在 classes 文件内 |
| `zero_width` | 退化框（短边为 0） |
| `image_missing` | 标注文件存在但对应图片不存在 |
| `image_unreadable` | 图片尺寸读取失败（文件损坏等） |

所有错误写入 `errors.jsonl` 并在 `feature_summary.json.errors` 中按类型计数
（含 `error_ratio_vs_label_files`）。图片存在但无标注文件的图记为 orphan
（计入 inventory，不进 errors）。空标注文件视为负样本图，计入
`empty_label_images`。

## 可重复性

- 不修改任何原始标注与图片；
- 输出目录自动创建，重复运行覆盖旧输出（plots 目录先清理）；
- 输入身份：classes 文件内容 sha256；每 split 的图片按 (文件名, 大小) 列表
  sha256、标注文件按内容 sha256 记入 `feature_summary.json.inputs`；
- 所有输出按固定精度（%.6f）与固定排序写出，相同输入两次运行输出逐字节一致
  （单测覆盖）；
- `feature_summary.json` 不含时间戳。

## 实际输出示例（jyz 样本数据）

```
python tools/dataset_feature_stats.py \
    --data-root /mnt/d/project_data/model_test/deimv2_obb_train_data/jyz \
    --splits train val \
    --dst_sz 640 \
    --output-dir outputs/dataset_feature_stats_jyz

# src:     train images=137 boxes=1160 / val images=65 boxes=534, errors=0
# dst_640: 同一批框的预处理后分布(每组 68 张图)
# 目标均值变化: length 111.6px -> 33.0px, aspect 1.69 -> 1.83, angle 89.3° -> 78.8°
```

## 尚未覆盖的情况

- DOTA 格式的 `difficult` 标志只用于剥离类名，不参与统计过滤（按有效框计入）；
- 类别 id 依据 `--classes-file` 校验；同一份标注换用不同 classes 文件需要重新运行；
- manifest 仅支持 `image_stem` 精确匹配，不支持按文件名模式推断 case/line/tower
  （不做猜测，字段缺失时输出空串并在 summary 记录 missing 列表）。
