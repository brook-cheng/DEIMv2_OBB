# DEIMv2-OBB 中文文档

[English](README.md) | 简体中文

`DEIMv2-OBB` 是基于 [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) 的 fork，添加**残差注意力机制，旋转框检测（OBB）与统一的应用层（`deim_app`）**——训练、评估与推理。

---

## 快速开始

`deim_app` 通过每个数据集一个 YAML 配置运行训练、评估与推理。完整 CLI 与 Python API 见 [推理 API 与 CLI 指南](docs/engineering/inference-api_CN.md)；YAML 结构见 [应用配置指南](docs/engineering/application-config_CN.md)。

### 环境安装

```shell
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```

### 选择并修改配置

`configs/app/examples/` 下有三个可直接使用的示例：

| 示例配置 | 用途 | 标注格式 |
| :--- | :--- | :--- |
| `hbb_coco.yml` | HBB + COCO | COCO JSON |
| `obb_dota.yml` | OBB + DOTA | DOTA txt |
| `obb_yolo.yml` | OBB + YOLO-OBB | YOLO-OBB txt |

复制一份示例，将 `data` 下的路径替换为本地数据集路径，再按需修改 `runtime`、`train`、`inference` 等公共字段。字段说明见 [应用配置指南](docs/engineering/application-config_CN.md)。

注意事项：

- OBB 数据必须提供 `classes_file`（每行一个类别名，行序即标签编号），`num_classes` 由标注元数据自动推导，无需手动填写。
- COCO 需 `cache_images: none`（示例已配置）；OBB 支持 `none` / `disk` / `ram`。
- 从零训练可用 `train.pretrained` 初始化骨干权重；断点续训用 `train.resume` 或 CLI `--resume`，两者互斥。

### 训练

```bash
python -m deim_app train \
  -c configs/app/examples/hbb_coco.yml \
  --device cuda \
  [--resume /path/to/checkpoint.pth] \
  [--output-dir ./outputs/my-run]
```

### 评估

```bash
python -m deim_app eval \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  [--device cuda]
```

### 推理

HBB（JSON + 可视化）：

```bash
python -m deim_app infer \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json visualization
```

OBB（JSON + DOTA + 可视化）：

```bash
python -m deim_app infer \
  -c configs/app/examples/obb_yolo.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json dota visualization
```

`json` 与 `visualization` 同时支持 HBB 与 OBB；`dota` 仅用于 OBB，每张图一个 txt，每行格式为 `x1 y1 x2 y2 x3 y3 x4 y4 class score`。

### Python API

```python
from deim_app import DetectionModel

model = DetectionModel.from_config("configs/app/examples/obb_yolo.yml")
model.load("outputs/model.pth")
results = model.predict_filtered("images/", score_threshold=0.25)
results.export_json("outputs/predictions.json")
results.export_dota("outputs/dota")
results.save_images("outputs/visualization")
```

`from_config` 不会自动加载权重，需显式调用 `load`（返回 `self`，可链式调用）。`predict_filtered` 默认使用配置中的 `score_threshold`、`class_filter` 与 `top_k`，也可在调用时覆盖筛选参数。

### 模型导出

应用层导出（`python -m deim_app export`）为预留接口，v1 版本尚未启用，调用会抛出 `ExportError`。需要 ONNX / TensorRT 请使用上游
[DEIMv2 仓库](https://github.com/Intellindust-AI-Lab/DEIMv2) 的部署工具（`tools/deployment/export_onnx.py` 与 `trtexec`）。
