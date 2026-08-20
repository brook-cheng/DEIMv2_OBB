#!/bin/bash
set -e
# 脚本相对自身定位仓库根（目录改名无关）
cd "$(dirname "$(readlink -f "$0")")/../../.."

# COMET_API_KEY 必须来自环境（密钥已轮换，禁止写回脚本）
# export COMET_API_KEY="..."

# 020 密度族已在 OBB 消融清理中删除
DENSITIES=(001 002 005 010 050 100)
TOTAL=${#DENSITIES[@]}
CURRENT=0
ERRORS=()
START_TIME=$(date)

echo "============================================================"
echo "  Synthetic Ellipse Training Batch"
echo "  Start: $START_TIME"
echo "  Total experiments: $TOTAL"
echo "  Densities: ${DENSITIES[*]}"
echo "============================================================"

for density in "${DENSITIES[@]}"; do
    CURRENT=$((CURRENT + 1))
    EXP_START=$(date +%s)

    echo ""
    echo "============================================================"
    echo "  [$CURRENT/$TOTAL] Starting density_${density} at $(date)"
    echo "  Config: configs/custom_obb/synthetic_configs/synthetic_exp_${density}.yml"
    echo "  Output: outputs/synthetic_exp_${density}/last.pth"
    echo "============================================================"

    if python tools/train/train.py --config "configs/custom_obb/synthetic_configs/synthetic_exp_${density}.yml"; then
        EXP_END=$(date +%s)
        ELAPSED=$((EXP_END - EXP_START))
        ELAPSED_MIN=$((ELAPSED / 60))
        echo "  [$CURRENT/$TOTAL] ✓ density_${density} COMPLETED in ${ELAPSED_MIN}m at $(date)"
    else
        EXP_END=$(date +%s)
        ELAPSED=$((EXP_END - EXP_START))
        echo "  [$CURRENT/$TOTAL] ✗ density_${density} FAILED after ${ELAPSED}s at $(date)"
        ERRORS+=("density_${density}")
    fi
done

END_TIME=$(date)

echo ""
echo "============================================================"
echo "  BATCH COMPLETE"
echo "  End: $END_TIME"
echo "============================================================"

# Summary
echo ""
echo "============================================================"
echo "  CHECKPOINT SUMMARY"
echo "============================================================"

ALL_OK=true
for density in "${DENSITIES[@]}"; do
    PTH="outputs/synthetic_exp_${density}/last.pth"
    if [ -f "$PTH" ]; then
        SIZE=$(du -h "$PTH" | cut -f1)
        echo "  ✓ density_${density}: $PTH ($SIZE)"
    else
        echo "  ✗ density_${density}: $PTH NOT FOUND"
        ALL_OK=false
    fi
done

echo ""
echo "============================================================"
echo "  TOTAL EXPERIMENTS: $TOTAL"
echo "  FAILURES: ${#ERRORS[@]}"
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo "  Failed densities: ${ERRORS[*]}"
fi
if $ALL_OK; then
    echo "  STATUS: ALL CHECKPOINTS PRESENT ✓"
else
    echo "  STATUS: SOME CHECKPOINTS MISSING ✗"
fi
echo "============================================================"
