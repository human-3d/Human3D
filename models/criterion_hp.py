# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.utils.comm import get_world_size
from torch import nn

from models.misc import (
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterionHumanParts(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        class_weights,
        num_human_queries,
        num_parts_per_human_queries,
        num_parts,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()

        self.num_human_queries = num_human_queries
        self.num_parts_per_human_queries = num_parts_per_human_queries
        self.num_parts = num_parts

        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        if self.class_weights != -1:
            assert (
                len(self.class_weights) == self.num_classes
            ), "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight", empty_weight)

        empty_weight_parts = torch.ones(self.num_parts)
        empty_weight_parts[-1] = self.eos_coef

        if self.class_weights != -1:
            assert (
                len(self.class_weights) == self.num_classes
            ), "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)

        self.register_buffer("empty_weight_parts", empty_weight_parts)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(
        self, outputs, targets, indices, num_masks, mask_type, parts
    ):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        if parts:
            target_classes = torch.full(
                src_logits.shape[:2],
                self.num_parts - 1,
                dtype=torch.int64,
                device=src_logits.device,
            )
        else:
            target_classes = torch.full(
                src_logits.shape[:2],
                self.num_classes,
                dtype=torch.int64,
                device=src_logits.device,
            )
        target_classes[idx] = target_classes_o

        if parts:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                self.empty_weight_parts,
                ignore_index=253,
            )
        else:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                self.empty_weight,
                ignore_index=253,
            )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(
        self, outputs, targets, indices, num_masks, mask_type, parts
    ):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id][mask_type][target_id]

            if self.num_points != -1:
                point_idx = torch.randperm(
                    target_mask.shape[1], device=target_mask.device
                )[: int(self.num_points * target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(
                    target_mask.shape[1], device=target_mask.device
                )

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()

            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))
        # del target_mask
        return {
            "loss_mask": torch.sum(torch.stack(loss_masks)),
            "loss_dice": torch.sum(torch.stack(loss_dices)),
        }

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t[mask_type] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(
                point_logits, point_labels, num_masks, mask_type
            ),
            "loss_dice": dice_loss_jit(
                point_logits, point_labels, num_masks, mask_type
            ),
        }

        del src_masks
        del target_masks
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

    def get_loss(
        self,
        loss,
        outputs,
        targets,
        indices,
        num_masks,
        mask_type,
        parts=False,
    ):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](
            outputs, targets, indices, num_masks, mask_type, parts
        )

    def forward(self, outputs, targets, mask_type):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        human_pred = {
            "pred_logits": outputs_without_aux["pred_human_logits"],
            "pred_masks": [
                pred_masks[:, : self.num_human_queries]
                for pred_masks in outputs_without_aux["pred_masks"]
            ],
        }

        human_target = [
            {"labels": t["human_labels"], "masks": t["human_masks"]}
            for t in targets
        ]

        indices = self.matcher(human_pred, human_target, mask_type)

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets, mask_type)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in human_target)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(
                loss, human_pred, human_target, indices, num_masks, mask_type
            )
            losses.update({f"human_{k}": v for k, v in l_dict.items()})

            # for k, v in l_dict.items():
            #    if v.isnan().cpu().item():
            #        print("jkjk")

        human_parts_targets = []
        human_parts_pred = {"pred_masks": list(), "pred_logits": list()}

        for batch_id in range(len(indices)):
            for match_id in range(len(indices[batch_id][0])):
                part_pred_logits = outputs_without_aux["pred_part_logits"][
                    batch_id,
                    range(
                        indices[batch_id][0][match_id],
                        self.num_human_queries
                        * self.num_parts_per_human_queries,
                        self.num_human_queries,
                    ),
                    :,
                ]
                part_pred_masks = outputs_without_aux["pred_masks"][batch_id][
                    :,
                    range(
                        indices[batch_id][0][match_id]
                        + self.num_human_queries,
                        self.num_human_queries
                        * (1 + self.num_parts_per_human_queries),
                        self.num_human_queries,
                    ),
                ]

                human_parts_pred["pred_masks"].append(part_pred_masks)

                human_parts_pred["pred_logits"].append(part_pred_logits)

                human_idx = (
                    targets[batch_id]["full_ids"] % 1000
                    == indices[batch_id][1][match_id] + 1
                )
                human_parts_targets.append(
                    {
                        "labels": targets[batch_id]["labels"][human_idx],
                        "masks": targets[batch_id]["masks"][human_idx],
                    }
                )

        human_parts_pred["pred_logits"] = torch.stack(
            human_parts_pred["pred_logits"]
        )

        part_indices = self.matcher(
            human_parts_pred, human_parts_targets, mask_type
        )

        for loss in self.losses:
            l_dict = self.get_loss(
                loss,
                human_parts_pred,
                human_parts_targets,
                part_indices,
                num_masks,
                mask_type,
                parts=True,
            )
            losses.update({f"parts_{k}": v for k, v in l_dict.items()})
            # for k, v in l_dict.items():
            #    if v.isnan().cpu().item():
            #        print("jkjk")

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                human_pred = {
                    "pred_logits": aux_outputs["pred_human_logits"],
                    "pred_masks": [
                        pred_masks[:, : self.num_human_queries]
                        for pred_masks in aux_outputs["pred_masks"]
                    ],
                }

                indices = self.matcher(human_pred, human_target, mask_type)

                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss,
                        human_pred,
                        human_target,
                        indices,
                        num_masks,
                        mask_type,
                    )
                    losses.update(
                        {f"human_{k}_{i}": v for k, v in l_dict.items()}
                    )
                    # for k, v in l_dict.items():
                    #    if v.isnan().cpu().item():
                    #        print("jkjk")

                human_parts_targets = []
                human_parts_pred = {
                    "pred_masks": list(),
                    "pred_logits": list(),
                }

                for batch_id in range(len(indices)):
                    for match_id in range(len(indices[batch_id][0])):
                        part_pred_logits = aux_outputs["pred_part_logits"][
                            batch_id,
                            range(
                                indices[batch_id][0][match_id],
                                self.num_human_queries
                                * self.num_parts_per_human_queries,
                                self.num_human_queries,
                            ),
                            :,
                        ]
                        part_pred_masks = aux_outputs["pred_masks"][batch_id][
                            :,
                            range(
                                indices[batch_id][0][match_id]
                                + self.num_human_queries,
                                self.num_human_queries
                                * (1 + self.num_parts_per_human_queries),
                                self.num_human_queries,
                            ),
                        ]

                        human_parts_pred["pred_masks"].append(part_pred_masks)
                        human_parts_pred["pred_logits"].append(
                            part_pred_logits
                        )
                        human_idx = (
                            targets[batch_id]["full_ids"] % 1000
                            == indices[batch_id][1][match_id] + 1
                        )
                        human_parts_targets.append(
                            {
                                "labels": targets[batch_id]["labels"][
                                    human_idx
                                ],
                                "masks": targets[batch_id]["masks"][human_idx],
                            }
                        )

                human_parts_pred["pred_logits"] = torch.stack(
                    human_parts_pred["pred_logits"]
                )

                part_indices = self.matcher(
                    human_parts_pred, human_parts_targets, mask_type
                )

                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss,
                        human_parts_pred,
                        human_parts_targets,
                        part_indices,
                        num_masks,
                        mask_type,
                        parts=True,
                    )
                    losses.update(
                        {f"parts_{k}_{i}": v for k, v in l_dict.items()}
                    )
                    # for k, v in l_dict.items():
                    #    if v.isnan().cpu().item():
                    #        print("jkjk")

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
