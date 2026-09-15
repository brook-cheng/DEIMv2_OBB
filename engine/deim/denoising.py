"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .obb_angle_contract import physical_rad_to_shifted_norm


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    box_mode="hbb",
):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None
    # ground truth number in each batch
    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device
    # max ground truth number in all batch
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    _num_box_dof = 5 if box_mode == "obb" else 4
    # devide groups by num_denoising
    # DN 查询预算钳位: max_gt 超过 num_denoising 预算时(如 DOTA 超密集 patch
    # 单图 2449 实例), num_group 的 0→1 钳位会让 dn_total = 2*max_gt 随实例数
    # 无界膨胀, 注意力矩阵 (num_queries+dn_total)^2 瞬态达十几 GB 直接 OOM。
    # 钳位后每图只取前 num_denoising 个 GT 参与 DN, dn_total ≤ 2*num_denoising,
    # 符合 num_denoising 的设计预算; 对 max_gt ≤ num_denoising 的普通批次零影响。
    max_gt_num = min(max(num_gts), num_denoising)
    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)  # batch size
    # use num_classes to init input_query_class
    input_query_class = torch.full(
        [bs, max_gt_num], num_classes, dtype=torch.int32, device=device
    )
    input_query_bbox = torch.zeros([bs, max_gt_num, _num_box_dof], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
    # put gt info into the container
    for i in range(bs):
        # 超出 DN 预算的 GT 不参与 denoising(见上方 max_gt_num 钳位注释)
        num_gt = min(num_gts[i], max_gt_num)
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"][:num_gt]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"][:num_gt]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    # (bs,2*num_group*max_gt_num)
    input_query_class = input_query_class.tile([1, 2 * num_group])
    # (bs,2*num_group*max_gt_num,4)
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    # (bs,2*num_group*max_gt_num)
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    # (bs,2*max_gt_num,1)
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    # (bs,2*num_group*max_gt_num,1)
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    # (bs,2*num_group*max_gt_num)
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    # (bs, num_group*num_gt) —— split 尺寸与钳位后的每图 DN GT 数一致
    dn_gt_nums = [min(n, max_gt_num) for n in num_gts]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in dn_gt_nums])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    ### add label noise: random change label value, simulate mistaken label ###
    if label_noise_ratio > 0:
        # create random label in ratio=label_noise_ratio * 0.5
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (
            label_noise_ratio * 0.5
        )
        # randomly put a new one here
        new_label = torch.randint_like(
            mask, 0, num_classes, dtype=input_query_class.dtype
        )
        # use new frontground label to replace old frontground labels
        input_query_class = torch.where(
            mask & pad_gt_mask, new_label, input_query_class
        )

    ### add boxes noise: random box bias, simulate mistake label ###
    spatial_bbox = input_query_bbox[
        ..., :4
    ]  # (x,y,w,h),不对角度加噪声，所有可以共用hbb的操作
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(spatial_bbox)
        diff = torch.tile(spatial_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(spatial_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(spatial_bbox)
        # negative_gt will add more rand bias
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (
            1 - negative_gt_mask
        )
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        noise_spatial = box_xyxy_to_cxcywh(known_bbox)
        noise_spatial[noise_spatial < 0] *= -1
    else:
        noise_spatial = spatial_bbox

    if box_mode == "hbb":
        input_query_bbox = noise_spatial
    elif box_mode == "obb":
        # [0,pi) → decoder 私有 shifted 编码 [0,1)
        input_query_bbox[..., 4] = physical_rad_to_shifted_norm(
            input_query_bbox[..., 4]
        )
        input_query_bbox = torch.cat([noise_spatial, input_query_bbox[..., 4:]], dim=-1)

    input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
        if i == num_group - 1:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2
            ] = True
        else:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising,
            ] = True
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i
            ] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
        # 每图实际参与 DN 的 GT 数(经预算钳位), 供 criterion 匹配端保持一致
        "dn_gt_nums": dn_gt_nums,
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
