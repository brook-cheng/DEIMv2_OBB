"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.checkpoint
from typing import List

from ..core import register
from .denoising import get_contrastive_denoising_training_group
from .utils import (
    deformable_attention_core_func_v2,
    get_activation,
    inverse_sigmoid,
    bias_init_with_prob,
)

from .dfine_decoder import MSDeformableAttention, LQE, Integral
from .dfine_utils import weighting_function, distance2bbox, distance2bbox_obb
from .deim_utils import RMSNorm, SwiGLUFFN, Gate, MLP
from .obb_angle_contract import (
    shifted_norm_to_physical_rad,
    physical_rad_to_shifted_norm,
)

__all__ = ["DEIMTransformer"]


_VALID_ANGLE_REPS = (0, 3)


def _assert_obb_angle_rep(angle_rep) -> None:
    """Assert ``angle_rep`` is exactly ``int`` ``0`` or ``3``.

    Python numeric equality lets ``False == 0``, ``0.0 == 0``, etc., so a
    plain ``angle_rep not in (0, 3)`` check accepts booleans and floats
    that are semantically invalid.  This guard rejects them at the
    boundary so the error surfaces at construction, not deep in forward.
    """
    if isinstance(angle_rep, bool) or type(angle_rep) is not int:
        raise TypeError(
            f"angle_rep must be int (0 or 3), "
            f"got {type(angle_rep).__name__}: {angle_rep!r}"
        )
    if angle_rep not in _VALID_ANGLE_REPS:
        raise ValueError(
            f"angle_rep must be 0 or 3, got {angle_rep}"
        )


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
        use_gateway=False,
    ):
        super(TransformerDecoderLayer, self).__init__()

        if layer_scale is not None:
            print(f"     --- Wide Layer@{layer_scale} ---")
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        # ffn
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        target,
        reference_points,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos_embed=None,
    ):

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )

        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)

        # ffn
        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        num_reg_dist,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        act="relu",
        box_mode="hbb",
        angle_rep=0,
        gradient_checkpoint=False,
    ):
        super(TransformerDecoder, self).__init__()
        self.gradient_checkpoint = gradient_checkpoint
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.box_mode = box_mode
        self.angle_rep = angle_rep
        if self.box_mode == "obb":
            _assert_obb_angle_rep(self.angle_rep)
        self.num_decouple_layers = num_layers

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [
                copy.deepcopy(decoder_layer_wide)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        self.lqe_layers = nn.ModuleList(
            [
                copy.deepcopy(
                    LQE(4, 64, 2, reg_max, act=act, num_reg_dist=num_reg_dist)
                )
                for _ in range(num_layers)
            ]
        )

        if self.angle_rep == 3:
            decouple_layer_template = self.layers[-1]
            self.decouple_angle_layers = nn.ModuleList(
                [
                    copy.deepcopy(decouple_layer_template)
                    for _ in range(self.num_decouple_layers)
                ]
            )

    def value_op(
        self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes
    ):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = (
            F.interpolate(memory, size=value_scale)
            if value_scale is not None
            else value
        )
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(
            self.reg_max, self.up, self.reg_scale, deploy=True
        )
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        dec_bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
        pre_angle_head=None,
        query_angle_head=None,
        dec_angle_head=None,
    ):

        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []

        # TODO: 需要针对obb请款下的project
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        if self.box_mode == "obb":
            if self.angle_rep == 0:
                ref_points_detach = F.sigmoid(ref_points_unact)
                query_pos_embed = query_pos_head(ref_points_detach).clamp(
                    min=-10, max=10
                )
            elif self.angle_rep == 3:
                dec_angle_output = target
                dec_angle_output_detach = 0
                dec_angle_pred_corners_undetach = 0
                ref_points_detach = F.sigmoid(ref_points_unact[..., :4])
                ref_dec_angle_detach = F.sigmoid(ref_points_unact)
                query_dec_angle_embed = query_angle_head(
                    F.sigmoid(ref_points_unact[..., 4:])
                )
                query_pos_embed = query_pos_head(ref_points_detach).clamp(
                    min=-10, max=10
                )
        elif self.box_mode == "hbb":
            ref_points_detach = F.sigmoid(ref_points_unact)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for layer_idx, layer in enumerate(self.layers):

            ref_points_input = ref_points_detach.unsqueeze(2)

            if layer_idx >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(
                    query_pos_embed, scale_factor=self.layer_scale
                )
                value = self.value_op(
                    memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes
                )
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

                if self.box_mode == "obb":
                    if self.angle_rep == 3:
                        query_dec_angle_embed = F.interpolate(
                            query_dec_angle_embed, scale_factor=self.layer_scale
                        )
                        dec_angle_output = F.interpolate(
                            dec_angle_output, size=query_dec_angle_embed.shape[-1]
                        )
                        dec_angle_output_detach = dec_angle_output.detach()

            if self.gradient_checkpoint and self.training:
                # 激活重计算: 反向重算层内注意力中间量(层输出本身是 aux 边界
                # 必须保留); use_reentrant=False 兼容 value 列表等非叶张量输入
                output = torch.utils.checkpoint.checkpoint(
                    layer,
                    output,
                    ref_points_input,
                    value,
                    spatial_shapes,
                    attn_mask,
                    query_pos_embed,
                    use_reentrant=False,
                )
            else:
                output = layer(
                    output,
                    ref_points_input,
                    value,
                    spatial_shapes,
                    attn_mask,
                    query_pos_embed,
                )

            if layer_idx == 0:
                pre_bboxes = F.sigmoid(
                    pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = score_head[layer_idx](output)
                ref_points_initial = pre_bboxes.detach()
            pred_corners = (
                dec_bbox_head[layer_idx](output + output_detach) + pred_corners_undetach
            )
            output_detach = output.detach()
            pred_corners_undetach = pred_corners

            if self.box_mode == "obb":
                if self.angle_rep == 3:
                    # ref_dec_angle_input 首次为(x,y,w,h,offset_w,offset_h)，后续为(x,y,w,h,r)
                    ref_dec_angle_input = ref_dec_angle_detach.unsqueeze(2)
                    if self.gradient_checkpoint and self.training:
                        dec_angle_output = torch.utils.checkpoint.checkpoint(
                            self.decouple_angle_layers[layer_idx],
                            dec_angle_output,
                            ref_dec_angle_input,
                            value,
                            spatial_shapes,
                            attn_mask,
                            query_dec_angle_embed,
                            use_reentrant=False,
                        )
                    else:
                        dec_angle_output = self.decouple_angle_layers[layer_idx](
                            dec_angle_output,
                            ref_dec_angle_input,
                            value,
                            spatial_shapes,
                            attn_mask,
                            query_dec_angle_embed,
                        )
                    if layer_idx == 0:
                        dec_angle_initial = torch.sigmoid(
                            pre_angle_head(dec_angle_output)
                            + inverse_sigmoid(ref_dec_angle_detach)[..., 4:]
                        )
                        if self.angle_rep == 3:
                            pre_bboxes = torch.concat(
                                [pre_bboxes, dec_angle_initial], dim=-1
                            )
                        ref_points_initial = pre_bboxes.detach()

                    dec_angle_pred_corners = (
                        dec_angle_head[layer_idx](
                            dec_angle_output + dec_angle_output_detach
                        )
                        + dec_angle_pred_corners_undetach
                    )
                    dec_angle_output_detach = dec_angle_output.detach()
                    dec_angle_pred_corners_undetach = dec_angle_pred_corners

                    # 1:(α,β,γ,δ)(ε,η)->(α,β,γ,δ,ε,η) 2:(α,β,γ,δ)(deta_theta)->(α,β,γ,δ,deta_theta)
                    pred_corners = torch.concat(
                        [pred_corners, dec_angle_pred_corners], dim=-1
                    )

            if self.box_mode == "hbb":
                inter_ref_bbox = distance2bbox(
                    ref_points_initial, integral(pred_corners, project), reg_scale
                )
            elif self.box_mode == "obb":
                # [0,1)→[-pi/4,3*pi/4)
                ref_phys = torch.cat(
                    [
                        ref_points_initial[..., :4],
                        shifted_norm_to_physical_rad(ref_points_initial[..., 4:5]),
                    ],
                    dim=-1,
                )
                distance = integral(pred_corners, project)
                inter_ref_bbox = distance2bbox_obb(
                    ref_phys,
                    distance,
                    reg_scale,
                )
                # [-pi/4,3*pi/4)→[0,1)
                inter_ref_bbox = torch.cat(
                    [
                        inter_ref_bbox[..., :4],
                        physical_rad_to_shifted_norm(inter_ref_bbox[..., 4:]),
                    ],
                    dim=-1,
                )

            if self.training or layer_idx == self.eval_idx:
                scores = score_head[layer_idx](output)
                scores = self.lqe_layers[layer_idx](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            if self.angle_rep == 3:
                ref_points_detach = inter_ref_bbox[..., :4].detach()
                ref_dec_angle_detach = inter_ref_bbox.detach()
            else:
                ref_points_detach = inter_ref_bbox.detach()
        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


@register()
class DEIMTransformer(nn.Module):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="hbb",
        angle_rep=0,
        gradient_checkpoint=False,
    ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        self.box_mode = box_mode
        self.angle_rep = angle_rep
        self.num_r_layers = num_layers

        # num_reg_dist: vertex bias for refienment, used in LQE
        # _num_box_dof: bbox representation, used to define headers
        if self.box_mode == "obb":
            _assert_obb_angle_rep(self.angle_rep)
            if self.angle_rep == 0:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 3:
                self._num_box_dof = 5  # (cx,cy,w,h,θ)
                self.num_reg_dist = 5  # (α,β,γ,δ,deta_θ)
        elif self.box_mode == "hbb":
            self._num_box_dof = 4  # (cx,cy,w,h)
            self.num_reg_dist = 4  # (α,β,γ,δ)

        assert query_select_method in ("default", "one2many", "agnostic"), ""
        assert cross_attn_method in ("default", "discrete"), ""
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        # -- print the parameters
        print(f"     --- Use Gateway@{use_gateway} ---")
        print(f"     --- Use Share Bbox Head@{share_bbox_head} ---")
        print(f"     --- Use Share Score Head@{share_score_head} ---")

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            use_gateway=use_gateway,
        )

        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            use_gateway=use_gateway,
        )

        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            self.num_reg_dist,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=activation,
            box_mode=self.box_mode,
            angle_rep=self.angle_rep,
            gradient_checkpoint=gradient_checkpoint,
        )
        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(
            hidden_dim, hidden_dim, self._num_box_dof, 3, act=mlp_act
        )

        # decoder head
        self.integral = Integral(self.reg_max, self.num_reg_dist)
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        dec_score_head = nn.Linear(hidden_dim, num_classes)
        self.dec_score_head = nn.ModuleList(
            [
                dec_score_head if share_score_head else copy.deepcopy(dec_score_head)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                copy.deepcopy(dec_score_head)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )

        self.pre_angle_head = None
        self.query_angle_head = None
        self.dec_angle_head = None
        if self.box_mode == "obb":
            if self.angle_rep == 0:
                pre_bbox_head_out_dim = 5  # (cx,cy,w,h,θ)
                num_query_pos_in = 5
                num_reg_dist_xywh = 6  # (α,β,γ,δ,ε,η)
            elif self.angle_rep == 3:
                pre_bbox_head_out_dim = 4
                num_query_pos_in = 4
                num_reg_dist_xywh = 4
                num_angle_describer = 1

            ## bbox head
            self.pre_bbox_head = MLP(
                hidden_dim, hidden_dim, pre_bbox_head_out_dim, 3, act=mlp_act
            )
            self.query_pos_head = MLP(
                num_query_pos_in, hidden_dim, hidden_dim, 3, act=mlp_act
            )
            dec_bbox_head_hide_mlp = MLP(
                hidden_dim,
                hidden_dim,
                num_reg_dist_xywh * (self.reg_max + 1),
                3,
                act=mlp_act,
            )
            dec_bbox_head_scaled_mlp = MLP(
                scaled_dim,
                scaled_dim,
                num_reg_dist_xywh * (self.reg_max + 1),
                3,
                act=mlp_act,
            )
            self.dec_bbox_head = nn.ModuleList(
                [
                    (
                        dec_bbox_head_hide_mlp
                        if share_bbox_head
                        else copy.deepcopy(dec_bbox_head_hide_mlp)
                    )
                    for _ in range(self.eval_idx + 1)
                ]
                + [
                    copy.deepcopy(dec_bbox_head_scaled_mlp)
                    for _ in range(num_layers - self.eval_idx - 1)
                ]
            )

            ## angle head
            if self.angle_rep == 3:
                self.pre_angle_head = MLP(
                    hidden_dim, hidden_dim, num_angle_describer, 3, act=mlp_act
                )
                self.query_angle_head = MLP(
                    num_angle_describer, hidden_dim, hidden_dim, 3, act=mlp_act
                )
                self.dec_angle_head = nn.ModuleList(
                    [
                        MLP(
                            hidden_dim,
                            hidden_dim,
                            num_angle_describer * (self.reg_max + 1),
                            3,
                            act=mlp_act,
                        )
                        for _ in range(self.num_layers)
                    ]
                )

        elif self.box_mode == "hbb":
            self.pre_bbox_head = MLP(
                hidden_dim, hidden_dim, self._num_box_dof, 3, act=mlp_act
            )
            dec_bbox_head = MLP(
                hidden_dim,
                hidden_dim,
                self.num_reg_dist * (self.reg_max + 1),
                3,
                act=mlp_act,
            )
            self.dec_bbox_head = nn.ModuleList(
                [
                    dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head)
                    for _ in range(self.eval_idx + 1)
                ]
                + [
                    MLP(
                        scaled_dim,
                        scaled_dim,
                        self.num_reg_dist * (self.reg_max + 1),
                        3,
                        act=mlp_act,
                    )
                    for _ in range(num_layers - self.eval_idx - 1)
                ]
            )
            self.query_pos_head = MLP(
                self._num_box_dof, hidden_dim, hidden_dim, 3, act=mlp_act
            )
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

        if self.angle_rep == 3:
            init.constant_(self.pre_angle_head.layers[-1].weight, 0)
            init.constant_(self.pre_angle_head.layers[-1].bias, 0)
            for dec_angle_h in self.dec_angle_head:
                init.constant_(dec_angle_h.layers[-1].weight, 0)
                init.constant_(dec_angle_h.layers[-1].bias, 0)
            init.xavier_uniform_(self.query_angle_head.layers[0].weight)
            init.xavier_uniform_(self.query_angle_head.layers[1].weight)
            init.xavier_uniform_(self.query_angle_head.layers[-1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels, self.hidden_dim, 1, bias=False
                                    ),
                                ),
                                (
                                    "norm",
                                    nn.BatchNorm2d(
                                        self.hidden_dim,
                                    ),
                                ),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels,
                                        self.hidden_dim,
                                        3,
                                        2,
                                        padding=1,
                                        bias=False,
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(
        self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"
    ):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        if self.box_mode == "hbb":
            for lvl, (h, w) in enumerate(spatial_shapes):
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(h), torch.arange(w), indexing="ij"
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1)
                grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor(
                    [w, h], dtype=dtype
                )
                wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
                lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(
                    -1, h * w, self._num_box_dof
                )
                anchors.append(lvl_anchors)
        elif self.box_mode == "obb":
            for lvl, (h, w) in enumerate(spatial_shapes):
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(h), torch.arange(w), indexing="ij"
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1)
                grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor(
                    [w, h], dtype=dtype
                )
                wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)

                default_r = 0.5
                r = default_r * torch.ones(
                    *grid_xy.shape[:-1],
                    1,
                    dtype=grid_xy.dtype,
                    device=grid_xy.device,
                )
                lvl_anchors = torch.concat([grid_xy, wh, r], dim=-1).reshape(
                    -1, h * w, self._num_box_dof
                )
                anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        denoising_logits=None,
        denoising_bbox_unact=None,
    ):
        """Prepare initial decoder queries from encoder memory.

        Pipeline:
            1. Generate (or reuse cached) grid anchors for every feature level.
            2. Mask invalid anchor positions out of encoder memory.
            3. Score every memory token with ``enc_score_head`` and select the
               top-``num_queries`` tokens as initial content queries.
            4. Regress anchor-box logits with ``enc_bbox_head`` to obtain
               unactivated reference points for the decoder.
            5. During training, produce sigmoid boxes for the encoder auxiliary
               output. In OBB mode the angle dimension is rescaled
               ``[0,1] -> [0, pi]`` unless ``angle_rep`` is active, in
               which case the 6-D ADR representation is kept as-is.
            6. Prepend denoising queries when ``denoising_bbox_unact`` is given.
               For the decoupled-angle path the denoising boxes are converted
               from 5-D OBB to the 6-D ``(external_rect, epsilon, eta)``
               layout so that the decoder sees a consistent representation.

        Args:
            memory (torch.Tensor): Encoder memory of shape ``(B, sum(H_l*W_l), hidden_dim)``.
            spatial_shapes: List of ``(H_l, W_l)`` per feature level.
            denoising_logits: Classification logits for denoising queries,
                shape ``(B, num_dn_queries, num_classes)``. Only provided in
                training with contrastive denoising enabled.
            denoising_bbox_unact: Unactivated denoising reference boxes. For
                ``box_mode="hbb"`` these are 4-D; for OBB ``angle_rep``
                they are 5-D ``(cx, cy, w, h, theta)`` and are converted to the
                6-D ADR layout before concatenation.

        Returns:
            content: Initial content queries for the decoder,
                shape ``(B, num_queries (+num_dn_queries), hidden_dim)`` .
            enc_topk_bbox_unact: Unactivated reference points for the decoder,
                shape ``(B, num_queries (+num_dn_queries), num_box_dof)``.
            enc_topk_bboxes_list: Auxiliary sigmoid boxes for the encoder
                output (training only); empty list in eval mode.
            enc_topk_logits_list: Auxiliary logits for the encoder output
                (training only); empty list in eval mode.
        """
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory

        enc_outputs_logits: torch.Tensor = self.enc_score_head(memory)

        # select topk queries
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            memory, enc_outputs_logits, anchors, self.num_queries
        )

        enc_topk_bbox_unact: torch.Tensor = (
            self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors
        )

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            if self.box_mode == "obb":
                # 内部 θ_shift 还原为物理角 [0, π)
                enc_topk_bboxes = torch.cat(
                    [
                        enc_topk_bboxes[..., :4],
                        shifted_norm_to_physical_rad(enc_topk_bboxes[..., 4:]),
                    ],
                    dim=-1,
                )

            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, enc_topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int,
    ):
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]),
        )

        topk_logits = (
            outputs_logits.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]),
            )
            if self.training
            else None
        )

        topk_memory = memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    box_mode=self.box_mode,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        ) = self._get_decoder_input(
            memory, spatial_shapes, denoising_logits, denoising_bbox_unact
        )
        # decoder
        # out_bboxes: 每层预测框; out_logits: 每层输出logits; out_corners: 每层输出refinement 值
        # out_refs: 每层初始参考位置，是pre_bboxes组成的list; pre_bboxes:初始预测位置; pre_logits: 初始输出logits
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = (
            self.decoder(
                target=init_ref_contents,
                ref_points_unact=init_ref_points_unact,
                memory=memory,
                spatial_shapes=spatial_shapes,
                dec_bbox_head=self.dec_bbox_head,
                score_head=self.dec_score_head,
                query_pos_head=self.query_pos_head,
                pre_bbox_head=self.pre_bbox_head,
                integral=self.integral,
                up=self.up,
                reg_scale=self.reg_scale,
                attn_mask=attn_mask,
                dn_meta=dn_meta,
                pre_angle_head=self.pre_angle_head,
                query_angle_head=self.query_angle_head,
                dec_angle_head=self.dec_angle_head,
            )
        )

        # criterion/matcher/postprocessor 中 theta 量纲为 [0, pi)
        # decoder 内部 [0,1) → 外部 [0, pi)
        if self.box_mode == "obb":
            theta_decode = shifted_norm_to_physical_rad
            out_bboxes = torch.cat(
                [
                    out_bboxes[..., :4],
                    theta_decode(out_bboxes[..., 4:]),
                ],
                dim=-1,
            )
            out_refs = torch.cat(
                [
                    out_refs[..., :4],
                    theta_decode(out_refs[..., 4:]),
                ],
                dim=-1,
            )
            pre_bboxes = torch.cat(
                [
                    pre_bboxes[..., :4],
                    theta_decode(pre_bboxes[..., 4:]),
                ],
                dim=-1,
            )
        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(
                pre_logits, dn_meta["dn_num_split"], dim=1
            )
            dn_pre_bboxes, pre_bboxes = torch.split(
                pre_bboxes, dn_meta["dn_num_split"], dim=1
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_corners, out_corners = torch.split(
                out_corners, dn_meta["dn_num_split"], dim=2
            )
            dn_out_refs, out_refs = torch.split(
                out_refs, dn_meta["dn_num_split"], dim=2
            )

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.aux_loss:
            # Layer-wise Supervision
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )

            out["enc_aux_outputs"] = self._set_aux_loss(
                enc_topk_logits_list, enc_topk_bboxes_list
            )
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                }
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class, outputs_coord)
        ]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(
                outputs_class, outputs_coord, outputs_corners, outputs_ref
            )
        ]
