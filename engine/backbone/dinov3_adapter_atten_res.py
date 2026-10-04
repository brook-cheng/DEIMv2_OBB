"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer
from .dinov3 import DinoVisionTransformer
from collections import OrderedDict

from tools.analysis.vis_feature import save_pca_features
from .atten_res import BlockAttnRes
from ..ultralytics.blocks import C3k2
from ..ultralytics.conv import Conv


class C3K2Module(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()
        # 1/4
        self.stem = nn.Sequential(
            *[
                Conv(3, inplanes, k=3, s=2, p=1, act=nn.GELU()),
                Conv(inplanes, inplanes, k=3, s=2, p=1, act=nn.GELU()),
                C3k2(inplanes, inplanes, n=1),
            ]
        )
        # 1/8
        self.c3k2_1 = nn.Sequential(
            *[
                Conv(inplanes, inplanes, k=3, s=2, p=1, act=nn.GELU()),
                C3k2(inplanes, inplanes * 2, n=1),
            ]
        )
        # 1/16
        self.c3k2_2 = nn.Sequential(
            *[
                Conv(inplanes * 2, inplanes * 2, k=3, s=2, p=1, act=nn.GELU()),
                C3k2(inplanes * 2, inplanes * 4, n=1),
            ]
        )
        # 1/32
        self.c3k2_3 = nn.Sequential(
            *[
                Conv(inplanes * 4, inplanes * 4, k=3, s=2, p=1, act=nn.GELU()),
                C3k2(inplanes * 4, inplanes * 4, n=1),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.c3k2_1(c1)  # 1/8
        c3 = self.c3k2_2(c2)  # 1/16
        c4 = self.c3k2_3(c3)  # 1/32
        return c2, c3, c4


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(
                    4 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)  # 1/8
        c3 = self.conv3(c2)  # 1/16
        c4 = self.conv4(c3)  # 1/32

        return c2, c3, c4


@register()
class DINOv3STAsResAtten(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        adapter_type="c3k2",
        conv_inplane=16,
        hidden_dim=None,
        gradient_checkpoint=False,
    ):
        super(DINOv3STAsResAtten, self).__init__()
        if "dinov3" in name:
            self.dinov3 = DinoVisionTransformer(
                name=name, gradient_checkpointing=gradient_checkpoint
            )
            if weights_path is not None and os.path.exists(weights_path):
                print(f"Loading ckpt from {weights_path}...")
                weights = torch.load(weights_path, weights_only=True, map_location="cpu")
                dinov3_weight = OrderedDict()
                if "model" in weights:
                    model_weight = weights["model"]
                    dinov3_weight_flag = False
                    for layer_weigt in model_weight:
                        if "backbone.dinov3." in layer_weigt:
                            dinov3_weight[
                                layer_weigt.replace("backbone.dinov3.", "")
                            ] = model_weight[layer_weigt]
                            dinov3_weight_flag = True
                    if not dinov3_weight_flag:
                        dinov3_weight = weights
                else:
                    dinov3_weight = weights
                self.dinov3.load_state_dict(dinov3_weight)
            else:
                print("Training DINOv3 from scratch...")
        else:
            self.dinov3 = VisionTransformer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                return_layers=interaction_indexes,
            )
            if weights_path is not None and os.path.exists(weights_path):
                print(f"Loading ckpt from {weights_path}...")
                self.dinov3._model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            else:
                print("Training ViT-Tiny from scratch...")

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # init the feature pyramid
        self.adapter_type = adapter_type
        if adapter_type == "sta":
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        elif adapter_type == "c3k2":
            print(f"Using C3K2 Module with inplanes={conv_inplane}")
            self.sta = C3K2Module(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    embed_dim + conv_inplane * 2,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Conv2d(
                    embed_dim + conv_inplane * 4,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Conv2d(
                    embed_dim + conv_inplane * 4,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            ]
        )
        # norm
        self.norms = nn.ModuleList(
            [
                nn.SyncBatchNorm(hidden_dim),
                nn.SyncBatchNorm(hidden_dim),
                nn.SyncBatchNorm(hidden_dim),
            ]
        )
        # attention residuals
        self.atten_res = nn.ModuleList(
            [
                BlockAttnRes(embed_dim),
                BlockAttnRes(embed_dim),
                BlockAttnRes(embed_dim),
            ]
        )

    def forward(self, x):
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(
            self.dinov3, VisionTransformer
        ):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        elif len(self.interaction_indexes) == 0:
            self.interaction_indexes = range(self.dinov3.n_blocks)
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:  # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        num_scales = len(self.atten_res) - 2

        for i, atten_res in enumerate(self.atten_res):
            tmp_sem_feats = []
            resize_H, resize_W = int(H_c * 2 ** (num_scales - i)), int(
                W_c * 2 ** (num_scales - i)
            )
            for j, tmp_sem_feat in enumerate(all_layers):
                feat, _ = tmp_sem_feat
                tmp_sem_feat = (
                    feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
                )  # [B, D, H, W]

                # tmp_sem_feat = F.interpolate(
                #     tmp_sem_feat,
                #     size=[resize_H, resize_W],
                #     mode="bilinear",
                #     align_corners=False,
                # )
                tmp_sem_feats.append(tmp_sem_feat)
            sem_feat = atten_res(tmp_sem_feats)
            sem_feat = F.interpolate(
                sem_feat,
                size=[resize_H, resize_W],
                mode="bilinear",
                align_corners=False,
            )
            sem_feats.append(sem_feat)

        # for i, sem_feat in enumerate(all_layers):
        #     feat, _ = sem_feat
        #     sem_feat = (
        #         feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
        #     )  # [B, D, H, W]
        #     resize_H, resize_W = int(H_c * 2 ** (num_scales - i)), int(
        #         W_c * 2 ** (num_scales - i)
        #     )
        #     sem_feat = F.interpolate(
        #         sem_feat,
        #         size=[resize_H, resize_W],
        #         mode="bilinear",
        #         align_corners=False,
        #     )
        #     sem_feats.append(sem_feat)

        # fusion
        fused_feats = []
        if self.adapter_type in ["sta", "c3k2"]:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        # save_pca_features(x, [c2, sem_feats[0], c3, sem_feats[1], c4, sem_feats[2]])

        return c2, c3, c4
