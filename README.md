# DEIMv2-OBB

English | [简体中文](./README_CN.md)

`DEIMv2-OBB` is a fork of [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) that adds **residual attention mechanism, oriented object detection (OBB), and a unified application layer (`deim_app`)** — training, evaluation, and inference.

---

## Quick Start

`deim_app` runs training, evaluation, and inference from one YAML config per dataset. For the full CLI and Python API, see the [Inference API & CLI Guide](docs/engineering/inference-api.md); for the YAML schema, see the [Application Config Guide](docs/engineering/application-config.md).

### Setup

```shell
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```

### Choose a config

Three examples cover the supported formats:

| Example | Format | Box mode |
| :--- | :--- | :---: |
| `hbb_coco.yml` | COCO | HBB |
| `obb_dota.yml` | DOTA | OBB |
| `obb_yolo.yml` | YOLO-OBB | OBB |

Copy the example that matches your dataset and edit its `data` section: point `train_images`, `train_annotations`, `val_images`, and `val_annotations` at your paths. OBB examples also require `classes_file` (one class name per line); COCO class names come from the annotation JSON instead.

Notes:

- OBB data must provide a `classes_file` (one class name per line; line order is the label index). `num_classes` is derived automatically from annotation metadata — no manual entry needed.
- COCO requires `cache_images: none` (already set in the examples); OBB supports `none` / `disk` / `ram`.
- Train from scratch with `train.pretrained` to initialize backbone weights; resume with `train.resume` or the CLI `--resume` — the two are mutually exclusive.

### Train

```bash
python -m deim_app train \
  -c configs/app/examples/hbb_coco.yml \
  --device cuda \
  [--resume /path/to/checkpoint.pth] \
  [--output-dir ./outputs/my-run]
```

### Evaluate

```bash
python -m deim_app eval \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  [--device cuda]
```

### Infer

HBB (JSON + visualization):

```bash
python -m deim_app infer \
  -c configs/app/examples/hbb_coco.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json visualization
```

OBB (JSON + DOTA + visualization):

```bash
python -m deim_app infer \
  -c configs/app/examples/obb_yolo.yml \
  -r /path/to/model.pth \
  -i /path/to/images \
  -o /tmp/deim-app-output \
  --score-threshold 0.25 \
  --format json dota visualization
```

`json` and `visualization` work for both HBB and OBB; `dota` is OBB-only — one txt per image, each line `x1 y1 x2 y2 x3 y3 x4 y4 class score`.

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

`from_config` does not load weights; call `load` next (returns `self`, chainable). `predict_filtered` applies `score_threshold`, `class_filter`, and `top_k` from the config by default, and accepts overrides at call time.

### Export

The `deim_app export` subcommand is a reserved interface and is not enabled in v1 — calling it raises `ExportError`. For ONNX / TensorRT, use the upstream [DEIMv2 repository](https://github.com/Intellindust-AI-Lab/DEIMv2) deployment tools (`tools/deployment/export_onnx.py` and `trtexec`).
