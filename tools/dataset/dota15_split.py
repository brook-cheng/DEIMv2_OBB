"""DOTA-v1.5 官方协议切图包装脚本.

调用第三方 DOTA_devkit 的 splitbase(官方 ImgSplit_multi_process),
不修改 devkit 源码,仅在此集中管理路径与协议参数。

输入目录约定(devkit 硬编码):
    <root>/<split>/images      原始大图
    <root>/<split>/labelTxt    原始标注(labelTxt-v1.5 的 symlink)
输出目录约定:
    <root>/split/<split>_<subsize>/{images,labelTxt}

协议参数与主流论文/DOTA 官方一致: subsize=1024, gap=200, thresh=0.7
(被切断实例按 overlap 比例保留或标记 difficult=2,由 devkit 决定)。

用法:
    python tools/dataset/dota15_split.py                     # train + val
    python tools/dataset/dota15_split.py --splits val        # 仅 val
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEVKIT_PATH = os.path.join(REPO_ROOT, "third_party", "DOTA_devkit")
sys.path.insert(0, DEVKIT_PATH)

from ImgSplit_multi_process import splitbase  # noqa: E402  (devkit 模块)

DEFAULT_ROOT = "/mnt/d/data/datasets/public/DOTA-V1.5"


def main() -> None:
    parser = argparse.ArgumentParser(description="DOTA-v1.5 官方协议切图")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="DOTA-V1.5 数据根目录")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="要切图的子集, 默认 train val")
    parser.add_argument("--subsize", type=int, default=1024, help="patch 边长")
    parser.add_argument("--gap", type=int, default=200, help="相邻 patch 重叠像素")
    parser.add_argument("--num-process", type=int, default=8, help="并行进程数")
    args = parser.parse_args()

    for split in args.splits:
        base = os.path.join(args.root, split)
        out = os.path.join(args.root, "split", f"{split}_{args.subsize}")
        assert os.path.isdir(os.path.join(base, "images")), f"缺少 {base}/images"
        assert os.path.isdir(os.path.join(base, "labelTxt")), (
            f"缺少 {base}/labelTxt (可 symlink 到 labelTxt-v1.5)"
        )
        print(f"[dota15_split] {base} -> {out} "
              f"(subsize={args.subsize}, gap={args.gap})", flush=True)
        # devkit 只用单层 os.mkdir, 多级输出目录需预先创建
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, "labelTxt"), exist_ok=True)
        splitter = splitbase(
            base,
            out,
            gap=args.gap,
            subsize=args.subsize,
            thresh=0.7,
            num_process=args.num_process,
        )
        splitter.splitdata(1)
        print(f"[dota15_split] {split} 完成", flush=True)


if __name__ == "__main__":
    main()
