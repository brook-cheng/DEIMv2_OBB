"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .dfine_utils import bbox2distance, bbox2distance_obb
from .obb_geometry import periodic_angle_distance, oriented_box_to_external_xyxy_rect
from .box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_iou,
    generalized_box_iou,
    diou,
    ciou,
    eiou,
    siou,
)
from .obb_ops import probiou, kld_loss
from .yolo_obb_loss import (
    canonical_side_l1_loss,
    yolo_angle_loss,
    yolo_probiou_loss,
)
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register
from .obb_angle_contract import physical_rad_to_norm


@register()
class DEIMCriterion(nn.Module):
    """This class computes the loss for DEIM."""

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        mal_iou_type=None,
        local_iou_type=None,
        box_mode="hbb",
        lambda_angle=1.0,
        obbox_rep_dim=6,
        periodic_angle_flag=True,
        use_yolo_probiou=False,
        use_yolo_angle=False,
        keep_kld=True,
        angle_lambda=3.0,
        adr_loss=False,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.mal_iou_type = mal_iou_type
        self.local_iou_type = local_iou_type
        self.box_mode = box_mode
        self.num_reg_dist = 4 if self.box_mode == "hbb" else obbox_rep_dim
        self.lambda_angle = lambda_angle
        self.obbox_rep_dim = obbox_rep_dim
        self.periodic_angle_flag = periodic_angle_flag
        self.use_yolo_probiou = use_yolo_probiou
        self.use_yolo_angle = use_yolo_angle
        self.keep_kld = keep_kld
        self.angle_lambda = angle_lambda
        self.adr_loss = adr_loss

        if self.box_mode == "obb" and self.adr_loss:
            required_keys = ["loss_extrect_l1", "loss_extrect_giou", "loss_offset_l1"]
            if self.keep_kld:
                required_keys.append("loss_kld")
            missing_keys = [key for key in required_keys if key not in weight_dict]
            if missing_keys:
                raise ValueError(
                    "OBB ADR-loss mode weight_dict is missing required keys: "
                    + ", ".join(missing_keys)
                )

        if self.box_mode == "obb" and (self.use_yolo_probiou or self.use_yolo_angle):
            required_keys = ["loss_bbox"]
            if self.use_yolo_probiou:
                required_keys.append("loss_probiou")
            if self.use_yolo_angle:
                required_keys.append("loss_angle")
            if self.keep_kld:
                required_keys.append("loss_kld")
            missing_keys = [key for key in required_keys if key not in weight_dict]
            if missing_keys:
                raise ValueError(
                    "OBB new-mode weight_dict is missing required keys: "
                    + ", ".join(missing_keys)
                )

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            if self.box_mode == "hbb":
                ious, _ = box_iou(
                    box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                )
                ious = torch.diag(ious).detach()
            elif self.box_mode == "obb":
                ious = probiou(target_boxes, src_boxes).squeeze(-1).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            if self.box_mode == "hbb":
                if self.local_iou_type == None or self.local_iou_type == "iou":
                    ious, _ = box_iou(
                        box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                    )
                elif self.local_iou_type == "giou":
                    gious = generalized_box_iou(
                        box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                    )
                    ious = (1 + gious) * 0.5
                elif self.local_iou_type == "dious":
                    dious = diou(
                        box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                    )
                    ious = (1 + dious) * 0.5
                elif self.local_iou_type == "ciou":
                    cious = ciou(
                        box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                    )
                    ious = (1 + cious) * 0.5
                elif self.local_iou_type == "eiou":
                    eious = eiou(
                        box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                    )
                    ious = (1 + eious) * 0.5
                else:
                    raise ValueError(f"{self.local_iou_type} is not supported")
                ious = torch.diag(ious).detach()
            elif self.box_mode == "obb":
                ious = probiou(target_boxes, src_boxes).squeeze(-1).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        # convert gt lables to one-hot code
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        # p
        pred_score = F.sigmoid(src_logits).detach()
        # q^{\gamma}
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            # weight=p^{\gamma}*(1-y)+y, y\in{0,1}
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        # loss(x,y)=L=\{l_1,l_2,...,l_N\}^T, l_n=-w_n*(y_n*log(sigmoid(x_n))+(1-y_n)log(1-sigmoid(x_n)))
        # l=-w(q^{\gamma}*log(p)+(1-q^{\gamma})*log(1-p)))
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )

        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        losses = {}
        if self.box_mode == "hbb":
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
            losses["loss_bbox"] = loss_bbox.sum() / num_boxes
            loss_giou = 1 - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                )
            )
            loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
            losses["loss_giou"] = loss_giou.sum() / num_boxes
        elif self.box_mode == "obb":
            if self.adr_loss:
                if src_boxes.shape[0] == 0:
                    zero = torch.zeros(
                        (), device=src_boxes.device, dtype=src_boxes.dtype
                    )
                    losses["loss_extrect_l1"] = zero
                    losses["loss_extrect_giou"] = zero
                    losses["loss_offset_l1"] = zero
                    if self.keep_kld:
                        losses["loss_kld"] = zero
                    return losses

                ext_rect_src, offsets_src = oriented_box_to_external_xyxy_rect(
                    src_boxes
                )
                ext_rect_tgt, offsets_tgt = oriented_box_to_external_xyxy_rect(
                    target_boxes
                )

                ext_src_cxcywh = box_xyxy_to_cxcywh(ext_rect_src)
                ext_tgt_cxcywh = box_xyxy_to_cxcywh(ext_rect_tgt)
                loss_extrect_l1 = F.l1_loss(
                    ext_src_cxcywh, ext_tgt_cxcywh, reduction="none"
                ).sum(-1)
                loss_extrect_giou = 1 - torch.diag(
                    generalized_box_iou(ext_rect_src, ext_rect_tgt)
                )
                loss_offset_l1 = F.l1_loss(
                    offsets_src, offsets_tgt, reduction="none"
                ).sum(-1)

                losses["loss_extrect_l1"] = loss_extrect_l1.sum() / num_boxes
                losses["loss_extrect_giou"] = loss_extrect_giou.sum() / num_boxes
                losses["loss_offset_l1"] = loss_offset_l1.sum() / num_boxes
                if self.keep_kld:
                    losses["loss_kld"] = (
                        kld_loss(src_boxes, target_boxes, reduction="none").sum()
                        / num_boxes
                    )
            elif self.use_yolo_probiou or self.use_yolo_angle:
                losses["loss_bbox"] = canonical_side_l1_loss(
                    src_boxes, target_boxes, num_boxes
                )
                if self.use_yolo_probiou:
                    losses["loss_probiou"] = yolo_probiou_loss(
                        src_boxes, target_boxes, num_boxes
                    )
                if self.use_yolo_angle:
                    losses["loss_angle"] = yolo_angle_loss(
                        src_boxes,
                        target_boxes,
                        num_boxes,
                        lambda_val=self.angle_lambda,
                    )
                if self.keep_kld:
                    loss_kld = kld_loss(src_boxes, target_boxes, reduction="none")
                    losses["loss_kld"] = loss_kld.sum() / num_boxes
            else:
                if self.periodic_angle_flag:
                    spatial_l1 = F.l1_loss(
                        src_boxes[..., :4], target_boxes[..., :4], reduction="none"
                    )
                    angle_term = (
                        self.lambda_angle
                        * periodic_angle_distance(
                            src_boxes[..., 4:], target_boxes[..., 4:]
                        )
                        / torch.pi
                    )
                    loss_bbox = torch.cat([spatial_l1, angle_term], dim=-1)
                else:
                    src_boxes_l1 = torch.cat(
                        [
                            src_boxes[..., :4],
                            physical_rad_to_norm(src_boxes[..., 4:]),
                        ],
                        dim=-1,
                    )
                    target_boxes_l1 = torch.cat(
                        [
                            target_boxes[..., :4],
                            physical_rad_to_norm(target_boxes[..., 4:]),
                        ],
                        dim=-1,
                    )
                    loss_bbox = F.l1_loss(
                        src_boxes_l1, target_boxes_l1, reduction="none"
                    )
                losses["loss_bbox"] = loss_bbox.sum() / num_boxes
                loss_kld = kld_loss(src_boxes, target_boxes, reduction="none")
                losses["loss_kld"] = loss_kld.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    if self.box_mode == "hbb":
                        self.fgl_targets_dn = bbox2distance(
                            ref_points,
                            box_cxcywh_to_xyxy(target_boxes),
                            self.reg_max,
                            outputs["reg_scale"],
                            outputs["up"],
                        )
                    elif self.box_mode == "obb":
                        self.fgl_targets_dn = bbox2distance_obb(
                            ref_points,
                            target_boxes,
                            self.reg_max,
                            outputs["reg_scale"],
                            outputs["up"],
                            obbox_rep_dim=self.obbox_rep_dim,
                        )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    if self.box_mode == "hbb":
                        self.fgl_targets = bbox2distance(
                            ref_points,
                            box_cxcywh_to_xyxy(target_boxes),
                            self.reg_max,
                            outputs["reg_scale"],
                            outputs["up"],
                        )
                    elif self.box_mode == "obb":
                        self.fgl_targets = bbox2distance_obb(
                            ref_points,
                            target_boxes,
                            self.reg_max,
                            outputs["reg_scale"],
                            outputs["up"],
                            obbox_rep_dim=self.obbox_rep_dim,
                        )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )
            if self.box_mode == "hbb":
                if self.local_iou_type == None or self.local_iou_type == "iou":
                    box_ious, _ = box_iou(
                        box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]),
                        box_cxcywh_to_xyxy(target_boxes),
                    )
                    ious = torch.diag(box_ious)
                elif self.local_iou_type == "giou":
                    gious = generalized_box_iou(
                        box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]),
                        box_cxcywh_to_xyxy(target_boxes),
                    )
                    gious = (1 + gious) * 0.5
                    ious = torch.diag(gious)
                elif self.local_iou_type == "diou":
                    dious = diou(
                        box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]),
                        box_cxcywh_to_xyxy(target_boxes),
                    )
                    dious = (1 + dious) * 0.5
                    ious = torch.diag(dious)
                elif self.local_iou_type == "ciou":
                    cious = ciou(
                        box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]),
                        box_cxcywh_to_xyxy(target_boxes),
                    )
                    cious = (1 + cious) * 0.5
                    ious = torch.diag(cious)
                elif self.local_iou_type == "eiou":
                    eious = eiou(
                        box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]),
                        box_cxcywh_to_xyxy(target_boxes),
                    )
                    eious = (1 + eious) * 0.5
                    ious = torch.diag(eious)
                else:
                    raise ValueError(f"undefined iou type:{self.local_iou_type}")
            elif self.box_mode == "obb":
                ious = (
                    probiou(target_boxes, outputs["pred_boxes"][idx])
                    .squeeze(-1)
                    .detach()
                )

            weight_targets = (
                ious.unsqueeze(-1).repeat(1, 1, self.num_reg_dist).reshape(-1).detach()
            )

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(
                    -1, (self.reg_max + 1)
                )
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = (
                        outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
                    )

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = (
                        mask.unsqueeze(-1).repeat(1, 1, self.num_reg_dist).reshape(-1)
                    )

                    weight_targets_local[idx] = ious.reshape_as(
                        weight_targets_local[idx]
                    ).to(weight_targets_local.dtype)
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1)
                        .repeat(1, 1, self.num_reg_dist)
                        .reshape(-1)
                        .detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            mask.sum() * batch_scale
                        ) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = (
                        loss_match_local[mask].mean() if mask.any() else 0
                    )
                    loss_match_local2 = (
                        loss_match_local[~mask].mean() if (~mask).any() else 0
                    )

                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos
                        + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [
            torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices
        ]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # TODO:针对rep0/rep2去除关于角度loss的计算
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, epoch=0, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)["indices"]
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs["aux_outputs"]
            if "pre_outputs" in outputs:
                aux_outputs_list = outputs["aux_outputs"] + [outputs["pre_outputs"]]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets, epoch=epoch)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                indices_enc = self.matcher(aux_outputs, targets, epoch=epoch)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go],
                dtype=torch.float,
                device=next(iter(outputs.values())).device,
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert "aux_outputs" in outputs, ""

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(
                loss, outputs, targets, indices_in, num_boxes_in, **meta
            )
            l_dict = {
                k: l_dict[k] * self.weight_dict[k]
                for k in l_dict
                if k in self.weight_dict
            }
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if "local" in self.losses:  # only work for local loss
                    aux_outputs["up"], aux_outputs["reg_scale"] = (
                        outputs["up"],
                        outputs["reg_scale"],
                    )
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_in
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(
                    loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                )

                l_dict = {
                    k: l_dict[k] * self.weight_dict[k]
                    for k in l_dict
                    if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss == "boxes")
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, enc_targets, indices_in
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                if "local" in self.losses:  # only work for local loss
                    aux_outputs["is_dn"] = True
                    aux_outputs["up"], aux_outputs["reg_scale"] = (
                        outputs["up"],
                        outputs["reg_scale"],
                    )
                for loss in self.losses:
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_dn
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_dn
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat(
            [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0
        )
        if self.box_mode == "obb":
            iou = batch_probiou(src_boxes.detach(), target_boxes)
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "iou":
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "giou":
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_cxcywh_to_xyxy(target_boxes),
                )
            )
        else:
            raise AttributeError()

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ("vfl", "mal"):
            meta = {"values": iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = (
            dn_meta["dn_positive_idx"],
            dn_meta["dn_num_group"],
        )
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        # 每图实际参与 DN 的 GT 数(denoising 端做过预算钳位, 见 denoising.py);
        # 无该键时退回完整 num_gt, 兼容旧调用方。
        dn_gt_nums = dn_meta.get("dn_gt_nums")
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                n = dn_gt_nums[i] if dn_gt_nums is not None else num_gt
                gt_idx = torch.arange(n, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self,
        pred,
        label,
        weight_right,
        weight_left,
        weight=None,
        reduction="sum",
        avg_factor=None,
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(
            -1
        )

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = (
            [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        )
        return opt_list
