"""一次性构建 DOTA15 切图的 disk npy 缓存 (train + val).

solver 只对 train dataset 自动 precache; val 需在此手动构建一次。
已缓存的图片自动跳过, 可安全重复执行。
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from engine.core import YAMLConfig  # noqa: E402

CFG = os.path.join(REPO_ROOT, "configs/custom_obb/dota15/sp_fz_rep3.yml")


def main() -> None:
    cfg = YAMLConfig(CFG)
    for name in ("train_dataloader", "val_dataloader"):
        ds_cfg = cfg.yaml_cfg[name]["dataset"]
        assert ds_cfg.get("cache_images") == "disk", f"{name} 未启用 disk 缓存"
    train_ds = cfg.train_dataloader.dataset
    val_ds = cfg.val_dataloader.dataset
    print("[precache] building train cache...", flush=True)
    train_ds.precache_images(num_workers=8)
    print("[precache] building val cache...", flush=True)
    val_ds.precache_images(num_workers=8)
    print("[precache] ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
