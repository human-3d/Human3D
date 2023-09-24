import colorsys
import functools
import gc
import math
import os
import random
import shutil
import statistics
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import pyviz3d.visualizer as vis
import torch
from torch_scatter import scatter_mean
from tqdm import tqdm

import models.metrics
from benchmark.evaluate_human_instances import evaluate as evaluate_human
from benchmark.evaluate_mhp import evaluate as evaluate_mhp
from models.metrics import IoU


def get_occlusion_subset(preds):
    occ_preds = []
    for split in ["low", "mid", "high"]:
        with open(
            f"occlusion_subsets/split_test_occlusion_{split}.txt", "r"
        ) as f:
            tmp = f.read().splitlines()

        occ_preds.append(
            {
                f"egobody_validation_{t.replace('.ply', '')}": preds[
                    f"egobody_validation_{t.replace('.ply', '')}"
                ]
                for t in tmp
                if f"egobody_validation_{t.replace('.ply', '')}"
                in preds.keys()
            }
        )

    return occ_preds[0], occ_preds[1], occ_preds[2]


def evaluate_bodysemseg(preds, gt_path):
    confusion = models.metrics.ConfusionMatrix(16, 255)
    iou = IoU()

    for (k, v) in tqdm(
        preds.items(), desc="evaluate sem seg from human instances"
    ):
        gt_file = os.path.join(gt_path, k + ".txt")
        assert os.path.isfile(
            gt_file
        ), "Scan {} does not match any gt file".format(k)

        gt_semseg = open(gt_file).read().splitlines()
        gt_semseg = (np.array(gt_semseg, dtype=np.int64) // 1000).astype(
            np.int64
        )

        scores, indices = v["pred_human_scores"].sort()

        final_bodysemseg = np.zeros_like(gt_semseg)

        for mask_id in range(len(scores)):
            score = scores[mask_id]
            index = indices[mask_id]

            if score > 0.5:
                final_bodysemseg[v["body_semseg"][index] > 0] = v[
                    "body_semseg"
                ][index][v["body_semseg"][index] > 0]

        confusion.add(final_bodysemseg, gt_semseg)
    return iou.value(confusion.value())


def evaluate_semseg_from_instseg(preds, gt_path):
    confusion = models.metrics.ConfusionMatrix(2, 255)
    iou = IoU()

    for (k, v) in tqdm(
        preds.items(), desc="evaluate sem seg from human instances"
    ):
        gt_file = os.path.join(gt_path, k + ".txt")
        assert os.path.isfile(
            gt_file
        ), "Scan {} does not match any gt file".format(k)

        gt_semseg = open(gt_file).read().splitlines()
        gt_semseg = (np.array(gt_semseg, dtype=np.int64) > 0).astype(np.int64)

        if type(v["pred_human_scores"]) == np.ndarray:
            human_score = v["pred_human_scores"]
        else:
            human_score = v["pred_human_scores"].numpy()

        human_semseg = (
            ((v["pred_human_masks"] * human_score) > 0.5)
            .any(axis=1)
            .astype(np.int64)
        )

        confusion.add(human_semseg, gt_semseg)
    return iou.value(confusion.value())


@functools.lru_cache(20)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(
                np.uint8
            ),
            HSV_tuples,
        )
    )


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/last-epoch.ckpt")
        print("Checkpoint created")


class InstanceSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.decoder_id = config.general.decoder_id

        if config.model.train_on_segments:
            self.mask_type = "segment_mask"
        else:
            self.mask_type = "masks"

        self.eval_on_segments = config.general.eval_on_segments

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext
        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad
        # loss
        self.ignore_label = config.data.ignore_label

        matcher = hydra.utils.instantiate(config.matcher)
        if not self.config.data.part2human:
            weight_dict = {
                "human_loss_ce": matcher.cost_class,
                "human_loss_mask": matcher.cost_mask,
                "human_loss_dice": matcher.cost_dice,
                "parts_loss_ce": matcher.cost_class,
                "parts_loss_mask": matcher.cost_mask,
                "parts_loss_dice": matcher.cost_dice,
            }
        else:
            weight_dict = {
                "loss_ce": matcher.cost_class,
                "loss_mask": matcher.cost_mask,
                "loss_dice": matcher.cost_dice,
            }
        if self.config.general.body_part_segmentation:
            weight_dict = {
                "loss_ce": matcher.cost_class,
                "loss_mask": matcher.cost_mask,
                "loss_dice": matcher.cost_dice,
            }

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            if i not in self.config.general.ignore_mask_idx:
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            else:
                aux_weight_dict.update(
                    {k + f"_{i}": 0.0 for k, v in weight_dict.items()}
                )
        weight_dict.update(aux_weight_dict)

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )

        # metrics
        self.confusion = hydra.utils.instantiate(config.metrics)
        self.confusion_human = models.metrics.ConfusionMatrix(2, 255)
        self.iou = IoU()
        # misc
        self.labels_info = dict()

    def forward(
        self,
        x,
        point2segment=None,
        raw_coordinates=None,
        is_eval=False,
        clip_feat=None,
        clip_pos=None,
    ):
        with self.optional_freeze():
            x = self.model(
                x,
                point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
                clip_feat=clip_feat,
                clip_pos=clip_pos,
            )
        return x

    def training_step(self, batch, batch_idx):
        data, target, file_names, clip_feat, clip_pos = batch

        if self.config.data.part2human:
            for b_id in range(len(target)):
                target[b_id]["labels"] = target[b_id].pop("human_labels")
                target[b_id]["masks"] = target[b_id].pop("human_masks")

        if data.features.shape[0] > self.config.general.max_batch_size:
            print("data exceeds threshold")
            raise RuntimeError("BATCH TOO BIG")

        if len(target) == 0:
            print("no targets")
            return None

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            output = self.forward(
                data,
                point2segment=[
                    target[i]["point2segment"] for i in range(len(target))
                ],
                raw_coordinates=raw_coordinates,
                clip_feat=clip_feat,
                clip_pos=clip_pos,
            )
        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err

        try:
            losses = self.criterion(output, target, mask_type=self.mask_type)
        except ValueError as val_err:
            print(f"ValueError: {val_err}")
            print(f"data shape: {data.shape}")
            print(f"data feat shape:  {data.features.shape}")
            print(f"data feat nans:   {data.features.isnan().sum()}")
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"filenames: {file_names}")
            raise val_err

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        logs = {
            f"train_{k}": v.detach().cpu().item() for k, v in losses.items()
        }

        logs["train_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_ce" in k]]
        )

        logs["train_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_mask" in k]]
        )

        logs["train_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_dice" in k]]
        )

        self.log_dict(logs)

        return sum(losses.values())

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def export(self, pred_masks, scores, pred_classes, file_names, decoder_id):
        root_path = "eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}/decoder_{decoder_id}"
        pred_mask_path = f"{base_path}/pred_mask"

        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = file_names
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > self.config.general.export_threshold:
                    # reduce the export size a bit. I guess no performance difference
                    np.savetxt(
                        f"{pred_mask_path}/{file_name}_{real_id}.txt",
                        mask,
                        fmt="%d",
                    )
                    fout.write(
                        f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n"
                    )

    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(
            outputs
        )
        results = {"train_loss_mean": train_loss}
        self.log_dict(results)

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)

    def save_visualizations(
        self,
        target_full,
        full_res_coords,
        sorted_masks,
        sort_classes,
        file_name,
        original_colors,
        original_normals,
        sort_scores_values,
        point_size=20,
        sorted_heatmaps=None,
        query_pos=None,
        backbone_features=None,
    ):

        full_res_coords -= full_res_coords.mean(axis=0)
        original_colors[:, :] = 120.0

        v = vis.Visualizer()

        v.add_points(
            "Input (no colors)",
            full_res_coords,
            colors=original_colors,
            normals=original_normals,
            visible=True,
            point_size=point_size,
        )

        if backbone_features is not None:
            v.add_points(
                "PCA",
                full_res_coords,
                colors=backbone_features,
                normals=original_normals,
                visible=False,
                point_size=point_size,
            )

        pred_coords = []
        pred_normals = []
        pred_sem_color = []
        pred_inst_color = []

        for did in range(len(sorted_masks)):
            for i in reversed(range(sorted_masks[did].shape[1])):
                if sort_scores_values[did][i] > 0.5:
                    mask_coords = full_res_coords[
                        sorted_masks[did][:, i].astype(bool), :
                    ]
                    mask_normals = original_normals[
                        sorted_masks[did][:, i].astype(bool), :
                    ]

                    label = sort_classes[did][i]

                    if len(mask_coords) == 0:
                        continue

                    pred_coords.append(mask_coords)
                    pred_normals.append(mask_normals)

                    pred_sem_color.append(
                        self.validation_dataset.map2color([label]).repeat(
                            mask_coords.shape[0], 1
                        )
                    )

                    pred_inst_color.append(
                        self.validation_dataset.map2color([i + 1])
                        .detach()
                        .cpu()
                        .unsqueeze(0)
                        .numpy()
                        .repeat(mask_coords.shape[0], 1)
                    )

            if len(pred_coords) > 0:
                pred_coords = np.concatenate(pred_coords)
                pred_normals = np.concatenate(pred_normals)
                pred_sem_color = np.concatenate(pred_sem_color)
                pred_inst_color = np.hstack(pred_inst_color)[0]

                v.add_points(
                    "Instances (Mask3D)",
                    pred_coords,
                    colors=pred_inst_color,
                    normals=pred_normals,
                    visible=True,
                    alpha=1.0,
                    point_size=point_size,
                )

        v.save(
            f"{self.config['general']['save_dir']}/visualizations/{file_name}"
        )

    def save_visualizations2(
        self,
        target_full,
        full_res_coords,
        sorted_masks,
        sort_classes,
        file_name,
        original_colors,
        original_normals,
        sort_scores_values,
        point_size=20,
        backbone_features=None,
    ):
        import open3d

        print(file_name)
        export_files = False
        threshold = 0.5

        full_res_coords -= full_res_coords.mean(axis=0)

        if not os.path.exists(
            f"{self.config['general']['save_dir']}/export/{file_name}"
        ):
            os.makedirs(
                f"{self.config['general']['save_dir']}/export/{file_name}"
            )

        v = vis.Visualizer()

        original_colors[:, :] = 120.0

        v.add_points(
            "Input (no colors)",
            full_res_coords,
            colors=original_colors,
            normals=original_normals,
            visible=True,
            point_size=point_size,
        )

        if backbone_features is not None:
            v.add_points(
                "PCA",
                full_res_coords,
                colors=backbone_features,
                normals=original_normals,
                visible=False,
                point_size=point_size,
            )

        humans = {
            k.item(): {"pos": [], "color": [], "part_color": [], "normals": []}
            for k in torch.unique(target_full["full_ids"] % 1000)
        }

        target_output = np.ones_like(full_res_coords, dtype=np.int16) * 226
        target_human = np.ones_like(full_res_coords, dtype=np.int16) * 226

        pcd_output = open3d.geometry.PointCloud()
        pcd_output.points = open3d.utility.Vector3dVector(full_res_coords)
        pcd_output.colors = open3d.utility.Vector3dVector(target_human / 255.0)
        pcd_output.normals = open3d.utility.Vector3dVector(original_normals)

        if export_files:
            open3d.io.write_point_cloud(
                f"{self.config['general']['save_dir']}/export/{file_name}/raw.ply",
                pcd_output,
            )

        pcd_output = open3d.geometry.PointCloud()
        pcd_output.points = open3d.utility.Vector3dVector(full_res_coords)
        pcd_output.colors = open3d.utility.Vector3dVector(
            original_colors / 255.0
        )
        pcd_output.normals = open3d.utility.Vector3dVector(original_normals)

        if export_files:
            open3d.io.write_point_cloud(
                f"{self.config['general']['save_dir']}/export/{file_name}/raw.ply",
                pcd_output,
            )

        if "labels" in target_full:
            instances_colors = torch.from_numpy(
                np.vstack(
                    get_evenly_distributed_colors(
                        target_full["labels"].shape[0]
                    )
                )
            )
            for instance_counter, (label, mask, full_id) in enumerate(
                zip(
                    target_full["labels"],
                    target_full["masks"],
                    target_full["full_ids"],
                )
            ):
                if label == 255:
                    continue

                human_id = (full_id % 1000).item()
                human_part = (full_id // 1000).item()

                mask_tmp = mask.detach().cpu().numpy()
                mask_coords = full_res_coords[mask_tmp.astype(bool), :]

                if len(mask_coords) == 0:
                    continue

                humans[human_id]["pos"].append(mask_coords)

                humans[human_id]["color"].append(
                    self.validation_dataset.map2color([human_part]).repeat(
                        humans[human_id]["pos"][-1].shape[0], 1
                    )
                )
                humans[human_id]["part_color"].append(
                    instances_colors[instance_counter % len(instances_colors)]
                    .unsqueeze(0)
                    .repeat(humans[human_id]["pos"][-1].shape[0], 1)
                )

                humans[human_id]["normals"].append(
                    original_normals[mask_tmp.astype(bool), :]
                )

                target_output[mask_tmp] = self.validation_dataset.map2color(
                    [human_part]
                ).repeat(humans[human_id]["pos"][-1].shape[0], 1)
                target_human[mask_tmp] = self.validation_dataset.map2color(
                    [human_id]
                ).repeat(humans[human_id]["pos"][-1].shape[0], 1)

            pcd_output = open3d.geometry.PointCloud()
            pcd_output.points = open3d.utility.Vector3dVector(full_res_coords)
            pcd_output.colors = open3d.utility.Vector3dVector(
                target_output / 255.0
            )
            pcd_output.normals = open3d.utility.Vector3dVector(
                original_normals
            )

            open3d.io.write_point_cloud(
                f"{self.config['general']['save_dir']}/export/{file_name}/targets_mhbps.ply",
                pcd_output,
            )

            pcd_output = open3d.geometry.PointCloud()
            pcd_output.points = open3d.utility.Vector3dVector(full_res_coords)
            pcd_output.colors = open3d.utility.Vector3dVector(
                target_human / 255.0
            )
            pcd_output.normals = open3d.utility.Vector3dVector(
                original_normals
            )

            if export_files:
                open3d.io.write_point_cloud(
                    f"{self.config['general']['save_dir']}/export/{file_name}/targets_human_instance.ply",
                    pcd_output,
                )

            for human_id, human_prop in humans.items():
                pos = np.concatenate(human_prop["pos"])
                # part_color = np.concatenate(human_prop['part_color'])
                color = np.concatenate(humans[human_id]["color"])
                normals = np.concatenate(human_prop["normals"])

                v.add_points(
                    f"{human_id}_target",
                    pos,
                    colors=color,
                    # colors=part_color,
                    normals=normals,
                    alpha=0.8,
                    visible=False,
                    point_size=point_size,
                )

        pred_output = np.ones_like(full_res_coords, dtype=np.int16) * 226
        pred_human = np.ones_like(full_res_coords, dtype=np.int16) * 226

        for did in range(len(sorted_masks)):
            for i in range(sorted_masks[did]["human"].shape[1]):
                human_mask = sorted_masks[did]["human"][:, i].astype(bool)

                if sort_scores_values[did]["human"][i] > threshold:
                    pred_human[human_mask] = (
                        self.validation_dataset.map2color([i + 1])
                        .detach()
                        .cpu()
                        .numpy()
                    )

                parts_mask = sorted_masks[did]["parts"][
                    :, range(i, 80, 5)
                ].astype(bool)
                restricted_mask = np.logical_and(
                    human_mask[..., None], parts_mask
                )

                part_classes = sort_classes[did]["parts"][range(i, 80, 5)]
                parts_scores = sort_scores_values[did]["parts"][
                    range(i, 80, 5)
                ]

                part_semseg = np.zeros(
                    restricted_mask.shape[0], dtype=np.int64
                )

                score_values, score_indx = parts_scores.sort()
                for score_index, score_value in zip(score_indx, score_values):
                    if score_value < 0.1:
                        continue

                    part_semseg[
                        restricted_mask[:, score_index]
                    ] = part_classes[score_index]

                if (part_semseg > 0).sum() > 0:
                    v.add_points(
                        f"{i}_pred_mhbps_score_{sort_scores_values[did]['human'][i].item():.2f}",
                        full_res_coords[part_semseg > 0],
                        visible=sort_scores_values[did]["human"][i].item()
                        > 0.01,
                        alpha=1.0,
                        normals=original_normals[part_semseg > 0, :],
                        colors=self.validation_dataset.map2color(
                            part_semseg[part_semseg > 0]
                        )
                        .detach()
                        .cpu()
                        .numpy(),
                        point_size=point_size,
                    )

                    if sort_scores_values[did]["human"][i] > threshold:
                        pred_output[part_semseg > 0] = (
                            self.validation_dataset.map2color(
                                part_semseg[part_semseg > 0]
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

        pcd_output = open3d.geometry.PointCloud()
        pcd_output.points = open3d.utility.Vector3dVector(full_res_coords)
        pcd_output.colors = open3d.utility.Vector3dVector(pred_output / 255.0)
        pcd_output.normals = open3d.utility.Vector3dVector(original_normals)

        if export_files:
            open3d.io.write_point_cloud(
                f"{self.config['general']['save_dir']}/export/{file_name}/predictions_mhbps.ply",
                pcd_output,
            )

        pcd_output = open3d.geometry.PointCloud()
        pcd_output.points = open3d.utility.Vector3dVector(full_res_coords)
        pcd_output.colors = open3d.utility.Vector3dVector(pred_human / 255.0)
        pcd_output.normals = open3d.utility.Vector3dVector(original_normals)

        if export_files:
            open3d.io.write_point_cloud(
                f"{self.config['general']['save_dir']}/export/{file_name}/predictions_human_instance.ply",
                pcd_output,
            )
        v.save(
            f"{self.config['general']['save_dir']}/visualizations/{file_name}"
        )

    def eval_step(self, batch, batch_idx):
        data, target, file_names, clip_feat, clip_pos = batch
        inverse_maps = data.inverse_maps
        target_full = data.target_full
        original_colors = data.original_colors
        data_idx = data.idx
        original_normals = data.original_normals
        original_coordinates = data.original_coordinates

        if self.config.data.part2human:
            for b_id in range(len(target)):
                target[b_id]["labels"] = target[b_id]["human_labels"]
                target[b_id]["masks"] = target[b_id]["human_masks"]

                target_full[b_id]["labels"] = target_full[b_id]["human_labels"]
                target_full[b_id]["masks"] = target_full[b_id]["human_masks"]

        if len(data.coordinates) == 0:
            return 0.0

        raw_coordinates = None
        if self.config.data.add_raw_coordinates:
            raw_coordinates = data.features[:, -3:]
            data.features = data.features[:, :-3]

        if raw_coordinates.shape[0] == 0:
            return 0.0

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device,
        )

        try:
            output = self.forward(
                data,
                point2segment=[
                    target[i]["point2segment"] for i in range(len(target))
                ],
                raw_coordinates=raw_coordinates,
                is_eval=True,
                clip_feat=clip_feat,
                clip_pos=clip_pos,
            )
        except RuntimeError as run_err:
            print(run_err)
            if (
                "only a single point gives nans in cross-attention"
                == run_err.args[0]
            ):
                return None
            else:
                raise run_err

        if (
            self.config.data.test_mode != "test"
        ):  # and self.validation_dataset.dataset_name != "human_segmentation":
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(False)

            try:
                losses = self.criterion(
                    output, target, mask_type=self.mask_type
                )
            except ValueError as val_err:
                print(f"ValueError: {val_err}")
                print(f"data shape: {data.shape}")
                print(f"data feat shape:  {data.features.shape}")
                print(f"data feat nans:   {data.features.isnan().sum()}")
                print(f"output: {output}")
                print(f"target: {target}")
                print(f"filenames: {file_names}")
                raise val_err

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            if self.config.trainer.deterministic:
                torch.use_deterministic_algorithms(True)

        if self.config.general.save_visualizations:
            backbone_features = (
                output["backbone_features"].F.detach().cpu().numpy()
            )
            from sklearn import decomposition

            pca = decomposition.PCA(n_components=3)
            pca.fit(backbone_features)
            pca_features = pca.transform(backbone_features)
            rescaled_pca = (
                255
                * (pca_features - pca_features.min())
                / (pca_features.max() - pca_features.min())
            )

        if (
            self.config.data.part2human
            or "pred_human_logits" not in output.keys()
        ):
            self.eval_instance_step_instance_segmentation(
                output,
                target,
                target_full,
                inverse_maps,
                file_names,
                original_coordinates,
                original_colors,
                original_normals,
                raw_coordinates,
                data_idx,
                backbone_features=rescaled_pca
                if self.config.general.save_visualizations
                else None,
            )
        else:
            self.eval_instance_step(
                output,
                target,
                target_full,
                inverse_maps,
                file_names,
                original_coordinates,
                original_colors,
                original_normals,
                raw_coordinates,
                data_idx,
                backbone_features=rescaled_pca
                if self.config.general.save_visualizations
                else None,
            )

        if self.config.data.test_mode != "test":
            return {
                f"val_{k}": v.detach().cpu().item() for k, v in losses.items()
            }
        else:
            return 0.0

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def get_full_res_mask(
        self, mask, inverse_map, point2segment_full, is_heatmap=False
    ):
        mask = mask.detach().cpu()[inverse_map]  # full res

        if self.eval_on_segments and is_heatmap == False:
            mask = scatter_mean(
                mask, point2segment_full, dim=0
            )  # full res segments
            mask = (mask > 0.5).float()
            mask = mask.detach().cpu()[
                point2segment_full.cpu()
            ]  # full res points

        return mask

    def get_mask_and_scores(
        self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None
    ):
        if device is None:
            device = self.device
        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        scores_per_query, labels_per_query = mask_cls.max(dim=1)

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6
        )
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap

    def eval_instance_step(
        self,
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords,
        original_colors,
        original_normals,
        raw_coords,
        idx,
        first_full_res=False,
        backbone_features=None,
    ):
        label_offset = self.validation_dataset.label_offset
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_human_logits": output["pred_human_logits"],
                "pred_part_logits": output["pred_part_logits"],
                "pred_masks": output["pred_masks"],
            }
        )

        prediction[self.decoder_id][
            "pred_human_logits"
        ] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_human_logits"], dim=-1
        )[
            ..., :-1
        ]

        prediction[self.decoder_id][
            "pred_part_logits"
        ] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_part_logits"], dim=-1
        )[
            ..., :-1
        ]

        all_human_pred_classes = list()
        all_human_pred_masks = list()
        all_human_pred_scores = list()
        all_human_heatmaps = list()
        all_human_query_pos = list()

        all_parts_pred_classes = list()
        all_parts_pred_masks = list()
        all_parts_pred_scores = list()
        all_parts_heatmaps = list()
        all_parts_query_pos = list()

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            if self.model.train_on_segments:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()[target_low_res[bid]["point2segment"].cpu()]
                )
            else:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()
                )

            (
                human_scores,
                human_masks,
                human_classes,
                human_heatmap,
            ) = self.get_mask_and_scores(
                prediction[self.decoder_id]["pred_human_logits"][bid]
                .detach()
                .cpu(),
                masks[:, : self.config.model.num_human_queries],
                prediction[self.decoder_id]["pred_human_logits"][bid].shape[0],
                2 - 1,
            )

            (
                parts_scores,
                parts_masks,
                parts_classes,
                parts_heatmap,
            ) = self.get_mask_and_scores(
                prediction[self.decoder_id]["pred_part_logits"][bid]
                .detach()
                .cpu(),
                masks[:, self.config.model.num_human_queries :],
                prediction[self.decoder_id]["pred_part_logits"][bid].shape[0],
                self.model.num_classes - 1,
            )

            human_masks = self.get_full_res_mask(
                human_masks,
                inverse_maps[bid],
                target_full_res[bid]["point2segment"]
                if self.validation_dataset.dataset_name != "human_segmentation"
                else torch.ones_like(inverse_maps[bid]),
            )

            parts_masks = self.get_full_res_mask(
                parts_masks,
                inverse_maps[bid],
                target_full_res[bid]["point2segment"]
                if self.validation_dataset.dataset_name != "human_segmentation"
                else torch.ones_like(inverse_maps[bid]),
            )

            human_heatmap = self.get_full_res_mask(
                human_heatmap,
                inverse_maps[bid],
                target_full_res[bid]["point2segment"]
                if self.validation_dataset.dataset_name != "human_segmentation"
                else torch.ones_like(inverse_maps[bid]),
                is_heatmap=True,
            )

            parts_heatmap = self.get_full_res_mask(
                parts_heatmap,
                inverse_maps[bid],
                target_full_res[bid]["point2segment"]
                if self.validation_dataset.dataset_name != "human_segmentation"
                else torch.ones_like(inverse_maps[bid]),
                is_heatmap=True,
            )

            if backbone_features is not None:
                backbone_features = self.get_full_res_mask(
                    torch.from_numpy(backbone_features),
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )
                backbone_features = backbone_features.numpy()

            human_masks = human_masks.numpy()
            parts_masks = parts_masks.numpy()
            human_heatmap = human_heatmap.numpy()
            parts_heatmap = parts_heatmap.numpy()

            sort_human_classes = human_classes
            sorted_human_masks = human_masks
            sorted_human_heatmap = human_heatmap
            sort_human_scores_values = human_scores

            sort_parts_classes = parts_classes
            sorted_parts_masks = parts_masks
            sorted_parts_heatmap = parts_heatmap
            sort_parts_scores_values = parts_scores

            all_human_pred_classes.append(sort_human_classes)
            all_human_pred_masks.append(sorted_human_masks)
            all_human_pred_scores.append(sort_human_scores_values)
            all_human_heatmaps.append(sorted_human_heatmap)

            all_parts_pred_classes.append(sort_parts_classes)
            all_parts_pred_masks.append(sorted_parts_masks)
            all_parts_pred_scores.append(sort_parts_scores_values)
            all_parts_heatmaps.append(sorted_parts_heatmap)

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            all_human_pred_classes[
                bid
            ] = self.validation_dataset._remap_model_output(
                all_human_pred_classes[bid].cpu() + label_offset
            )
            all_parts_pred_classes[
                bid
            ] = self.validation_dataset._remap_model_output(
                all_parts_pred_classes[bid].cpu() + label_offset
            )

            if (
                self.config.data.test_mode != "test"
                and len(target_full_res) != 0
            ):
                target_full_res[bid][
                    "labels"
                ] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]["labels"].cpu() + label_offset
                )
                target_full_res[bid][
                    "human_labels"
                ] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]["human_labels"].cpu() + label_offset
                )

            if self.config.general.eval_inner_core == -1:
                body_semseg = []
                for i in range(all_human_pred_masks[bid].shape[1]):
                    human_mask = all_human_pred_masks[bid][:, i].astype(bool)

                    parts_mask = all_parts_pred_masks[bid][
                        :, range(i, 80, 5)
                    ].astype(bool)
                    restricted_mask = np.logical_and(
                        human_mask[..., None], parts_mask
                    )

                    part_classes = all_parts_pred_classes[bid][range(i, 80, 5)]
                    parts_scores = all_parts_pred_scores[bid][range(i, 80, 5)]

                    part_semseg = np.zeros(
                        restricted_mask.shape[0], dtype=np.int64
                    )

                    score_values, score_indx = parts_scores.sort()
                    for score_index, score_value in zip(
                        score_indx, score_values
                    ):
                        if score_value < 0.1:
                            continue

                        part_semseg[
                            restricted_mask[:, score_index]
                        ] = part_classes[score_index]

                    body_semseg.append(part_semseg)

                if self.config.general.save_visualizations:
                    self.preds[file_names[bid]] = {
                        "pred_human_masks": all_human_pred_masks[bid].astype(
                            bool
                        ),
                        "pred_human_scores": all_human_pred_scores[bid],
                        "pred_human_classes": all_human_pred_classes[bid],
                        "pred_parts_masks": all_parts_pred_masks[bid],
                        "pred_parts_scores": all_parts_pred_scores[bid],
                        "pred_parts_classes": all_parts_pred_classes[bid],
                        "body_semseg": np.stack(body_semseg),
                    }
                else:
                    self.preds[file_names[bid]] = {
                        "pred_human_masks": all_human_pred_masks[bid].astype(
                            bool
                        ),
                        "pred_human_scores": all_human_pred_scores[bid],
                        "body_semseg": np.stack(body_semseg),
                    }
            else:
                # prev val_dataset
                self.preds[file_names[bid]] = {
                    "pred_masks": all_pred_masks[bid][
                        self.test_dataset.data[idx[bid]]["cond_inner"]
                    ],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                }

            if self.config.general.save_visualizations:
                if "cond_inner" in self.test_dataset.data[idx[bid]]:
                    target_full_res[bid]["masks"] = target_full_res[bid][
                        "masks"
                    ][:, self.test_dataset.data[idx[bid]]["cond_inner"]]
                    self.save_visualizations(
                        target_full_res[bid],
                        full_res_coords[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        [self.preds[file_names[bid]]["pred_masks"]],
                        [self.preds[file_names[bid]]["pred_classes"]],
                        file_names[bid],
                        original_colors[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        original_normals[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        [self.preds[file_names[bid]]["pred_scores"]],
                        sorted_heatmaps=[
                            all_heatmaps[bid][
                                self.test_dataset.data[idx[bid]]["cond_inner"]
                            ]
                        ],
                        query_pos=all_query_pos[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ]
                        if len(all_query_pos) > 0
                        else None,
                        backbone_features=backbone_features[
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        point_size=self.config.general.visualization_point_size,
                    )
                else:
                    self.save_visualizations2(
                        target_full_res[bid],
                        full_res_coords[bid],
                        [
                            {
                                "parts": self.preds[file_names[bid]][
                                    "pred_parts_masks"
                                ],
                                "human": self.preds[file_names[bid]][
                                    "pred_human_masks"
                                ],
                            }
                        ],
                        [
                            {
                                "parts": self.preds[file_names[bid]][
                                    "pred_parts_classes"
                                ],
                                "human": self.preds[file_names[bid]][
                                    "pred_human_classes"
                                ],
                            }
                        ],
                        file_names[bid].replace(
                            "egobody_validation_",
                            "",
                        ),
                        original_colors[bid],
                        original_normals[bid],
                        [
                            {
                                "parts": self.preds[file_names[bid]][
                                    "pred_parts_scores"
                                ],
                                "human": self.preds[file_names[bid]][
                                    "pred_human_scores"
                                ],
                            }
                        ],
                        backbone_features=backbone_features,
                        point_size=self.config.general.visualization_point_size,
                    )

    def eval_instance_step_instance_segmentation(
        self,
        output,
        target_low_res,
        target_full_res,
        inverse_maps,
        file_names,
        full_res_coords,
        original_colors,
        original_normals,
        raw_coords,
        idx,
        first_full_res=False,
        backbone_features=None,
    ):
        label_offset = self.validation_dataset.label_offset
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )

        prediction[self.decoder_id][
            "pred_logits"
        ] = torch.functional.F.softmax(
            prediction[self.decoder_id]["pred_logits"], dim=-1
        )[
            ..., :-1
        ]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            if self.model.train_on_segments:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()[target_low_res[bid]["point2segment"].cpu()]
                )
            else:
                masks = (
                    prediction[self.decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()
                )

            scores, masks, classes, heatmap = self.get_mask_and_scores(
                prediction[self.decoder_id]["pred_logits"][bid].detach().cpu(),
                masks,
                prediction[self.decoder_id]["pred_logits"][bid].shape[0],
                self.model.num_classes - 1,
            )

            masks = self.get_full_res_mask(
                masks,
                inverse_maps[bid],
                target_full_res[bid]["point2segment"],
            )

            heatmap = self.get_full_res_mask(
                heatmap,
                inverse_maps[bid],
                target_full_res[bid]["point2segment"],
                is_heatmap=True,
            )

            if backbone_features is not None:
                backbone_features = self.get_full_res_mask(
                    torch.from_numpy(backbone_features),
                    inverse_maps[bid],
                    target_full_res[bid]["point2segment"],
                    is_heatmap=True,
                )
                backbone_features = backbone_features.numpy()

            masks = masks.numpy()
            heatmap = heatmap.numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            if self.config.general.filter_out_instances:
                keep_instances = set()
                pairwise_overlap = sorted_masks.T @ sorted_masks
                normalization = pairwise_overlap.max(axis=0)
                norm_overlaps = pairwise_overlap / normalization

                for instance_id in range(norm_overlaps.shape[0]):
                    if not (
                        sort_scores_values[instance_id]
                        < self.config.general.scores_threshold
                    ):
                        # check if mask != empty
                        if not sorted_masks[:, instance_id].sum() == 0.0:
                            overlap_ids = set(
                                np.nonzero(
                                    norm_overlaps[instance_id, :]
                                    > self.config.general.iou_threshold
                                )[0]
                            )

                            if len(overlap_ids) == 0:
                                keep_instances.add(instance_id)
                            else:
                                if instance_id == min(overlap_ids):
                                    keep_instances.add(instance_id)

                keep_instances = sorted(list(keep_instances))
                all_pred_classes.append(sort_classes[keep_instances])
                all_pred_masks.append(sorted_masks[:, keep_instances])
                all_pred_scores.append(sort_scores_values[keep_instances])
                all_heatmaps.append(sorted_heatmap[:, keep_instances])
            else:
                all_pred_classes.append(sort_classes)
                all_pred_masks.append(sorted_masks)
                all_pred_scores.append(sort_scores_values)
                all_heatmaps.append(sorted_heatmap)

        for bid in range(len(prediction[self.decoder_id]["pred_masks"])):
            all_pred_classes[
                bid
            ] = self.validation_dataset._remap_model_output(
                all_pred_classes[bid].cpu() + label_offset
            )

            if (
                self.config.data.test_mode != "test"
                and len(target_full_res) != 0
            ):
                target_full_res[bid][
                    "labels"
                ] = self.validation_dataset._remap_model_output(
                    target_full_res[bid]["labels"].cpu() + label_offset
                )

                # PREDICTION BOX
                bbox_data = []
                for query_id in range(
                    all_pred_masks[bid].shape[1]
                ):  # self.model.num_queries
                    obj_coords = full_res_coords[bid][
                        all_pred_masks[bid][:, query_id].astype(bool), :
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))

                        bbox_data.append(
                            (
                                all_pred_classes[bid][query_id].item(),
                                bbox,
                                all_pred_scores[bid][query_id],
                            )
                        )
                self.bbox_preds[file_names[bid]] = bbox_data

                # GT BOX
                bbox_data = []
                for obj_id in range(target_full_res[bid]["masks"].shape[0]):
                    if target_full_res[bid]["labels"][obj_id].item() == 255:
                        continue

                    obj_coords = full_res_coords[bid][
                        target_full_res[bid]["masks"][obj_id, :]
                        .cpu()
                        .detach()
                        .numpy()
                        .astype(bool),
                        :,
                    ]
                    if obj_coords.shape[0] > 0:
                        obj_center = obj_coords.mean(axis=0)
                        obj_axis_length = obj_coords.max(
                            axis=0
                        ) - obj_coords.min(axis=0)

                        bbox = np.concatenate((obj_center, obj_axis_length))
                        bbox_data.append(
                            (
                                target_full_res[bid]["labels"][obj_id].item(),
                                bbox,
                            )
                        )

                self.bbox_gt[file_names[bid]] = bbox_data

            if self.config.general.eval_inner_core == -1:
                self.preds[file_names[bid]] = {
                    "pred_masks": all_pred_masks[bid],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                }
            else:
                # prev val_dataset
                self.preds[file_names[bid]] = {
                    "pred_masks": all_pred_masks[bid][
                        self.test_dataset.data[idx[bid]]["cond_inner"]
                    ],
                    "pred_scores": all_pred_scores[bid],
                    "pred_classes": all_pred_classes[bid],
                }

            if self.config.general.save_visualizations:
                if "cond_inner" in self.test_dataset.data[idx[bid]]:
                    target_full_res[bid]["masks"] = target_full_res[bid][
                        "masks"
                    ][:, self.test_dataset.data[idx[bid]]["cond_inner"]]
                    self.save_visualizations(
                        target_full_res[bid],
                        full_res_coords[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        [self.preds[file_names[bid]]["pred_masks"]],
                        [self.preds[file_names[bid]]["pred_classes"]],
                        file_names[bid],
                        original_colors[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        original_normals[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        [self.preds[file_names[bid]]["pred_scores"]],
                        sorted_heatmaps=[
                            all_heatmaps[bid][
                                self.test_dataset.data[idx[bid]]["cond_inner"]
                            ]
                        ],
                        query_pos=all_query_pos[bid][
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ]
                        if len(all_query_pos) > 0
                        else None,
                        backbone_features=backbone_features[
                            self.test_dataset.data[idx[bid]]["cond_inner"]
                        ],
                        point_size=self.config.general.visualization_point_size,
                    )
                else:
                    self.save_visualizations(
                        target_full_res[bid],
                        full_res_coords[bid],
                        [self.preds[file_names[bid]]["pred_masks"]],
                        [self.preds[file_names[bid]]["pred_classes"]],
                        file_names[bid].replace(
                            "egobody_validation_",
                            "",
                        ),
                        original_colors[bid],
                        original_normals[bid],
                        [self.preds[file_names[bid]]["pred_scores"]],
                        sorted_heatmaps=[all_heatmaps[bid]],
                        query_pos=all_query_pos[bid]
                        if len(all_query_pos) > 0
                        else None,
                        backbone_features=backbone_features,
                        point_size=self.config.general.visualization_point_size,
                    )

            if self.config.general.export:
                self.export(
                    self.preds[file_names[bid]]["pred_masks"],
                    self.preds[file_names[bid]]["pred_scores"],
                    self.preds[file_names[bid]]["pred_classes"],
                    file_names[bid],
                    self.decoder_id,
                )

    def eval_instance_epoch_end(self):
        low_pred, mid_pred, high_pred = get_occlusion_subset(self.preds)
        ap_results = {}

        root_path = "eval_output"
        base_path = f"{root_path}/instance_evaluation_{self.config.general.experiment_name}_{self.current_epoch}"

        gt_data_path = f"{self.test_dataset.data_dir[0]}/gt_part/validation"

        pred_path = f"{base_path}/tmp_output.txt"

        log_prefix = "val"

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        try:
            if self.config.general.body_part_segmentation:
                from benchmark.evaluate_semantic_instance_original import (
                    evaluate as evaluate_original,
                )

                part_APs = evaluate_original(
                    self.preds,
                    gt_data_path,
                    pred_path,
                    dataset="human_part_segmentation",
                )

                low_part_APs = evaluate_original(
                    low_pred,
                    gt_data_path,
                    pred_path,
                    dataset="human_part_segmentation",
                )
                mid_part_APs = evaluate_original(
                    mid_pred,
                    gt_data_path,
                    pred_path,
                    dataset="human_part_segmentation",
                )
                high_part_APs = evaluate_original(
                    high_pred,
                    gt_data_path,
                    pred_path,
                    dataset="human_part_segmentation",
                )

                for part_name in part_APs["classes"]:
                    for score_name in part_APs["classes"][part_name]:
                        ap_results[
                            f"{log_prefix}_{part_name}_{score_name}"
                        ] = float(part_APs["classes"][part_name][score_name])

                ap_results[f"{log_prefix}_mean_AP"] = float(part_APs["all_ap"])
                ap_results[f"{log_prefix}_mean_AP_50"] = float(
                    part_APs["all_ap_50%"]
                )
                ap_results[f"{log_prefix}_mean_AP_25"] = float(
                    part_APs["all_ap_25%"]
                )

                for part_name in low_part_APs["classes"]:
                    for score_name in low_part_APs["classes"][part_name]:
                        ap_results[
                            f"{log_prefix}_low_{part_name}_{score_name}"
                        ] = float(
                            low_part_APs["classes"][part_name][score_name]
                        )

                ap_results[f"{log_prefix}_low_mean_AP"] = float(
                    low_part_APs["all_ap"]
                )
                ap_results[f"{log_prefix}_low_mean_AP_50"] = float(
                    low_part_APs["all_ap_50%"]
                )
                ap_results[f"{log_prefix}_low_mean_AP_25"] = float(
                    low_part_APs["all_ap_25%"]
                )

                for part_name in mid_part_APs["classes"]:
                    for score_name in mid_part_APs["classes"][part_name]:
                        ap_results[
                            f"{log_prefix}_mid_{part_name}_{score_name}"
                        ] = float(
                            mid_part_APs["classes"][part_name][score_name]
                        )

                ap_results[f"{log_prefix}_mid_mean_AP"] = float(
                    mid_part_APs["all_ap"]
                )
                ap_results[f"{log_prefix}_mid_mean_AP_50"] = float(
                    mid_part_APs["all_ap_50%"]
                )
                ap_results[f"{log_prefix}_mid_mean_AP_25"] = float(
                    mid_part_APs["all_ap_25%"]
                )

                for part_name in high_part_APs["classes"]:
                    for score_name in high_part_APs["classes"][part_name]:
                        ap_results[
                            f"{log_prefix}_high_{part_name}_{score_name}"
                        ] = float(
                            high_part_APs["classes"][part_name][score_name]
                        )

                ap_results[f"{log_prefix}_high_mean_AP"] = float(
                    high_part_APs["all_ap"]
                )
                ap_results[f"{log_prefix}_high_mean_AP_50"] = float(
                    high_part_APs["all_ap_50%"]
                )
                ap_results[f"{log_prefix}_high_mean_AP_25"] = float(
                    high_part_APs["all_ap_25%"]
                )

                self.log_dict(ap_results)

                if not self.config.general.export:
                    shutil.rmtree(base_path)

                del self.preds
                del self.bbox_preds
                del self.bbox_gt

                gc.collect()

                self.preds = dict()
                self.bbox_preds = dict()
                self.bbox_gt = dict()

                return
            elif self.validation_dataset.dataset_name == "human_segmentation":
                if not self.config.data.part2human:
                    human_iou = evaluate_semseg_from_instseg(
                        self.preds, gt_data_path.replace("gt_part", "gt_human")
                    )

                    low_human_iou = evaluate_semseg_from_instseg(
                        low_pred, gt_data_path.replace("gt_part", "gt_human")
                    )
                    mid_human_iou = evaluate_semseg_from_instseg(
                        mid_pred, gt_data_path.replace("gt_part", "gt_human")
                    )
                    high_human_iou = evaluate_semseg_from_instseg(
                        high_pred, gt_data_path.replace("gt_part", "gt_human")
                    )

                    for scene_name in self.preds:
                        self.preds[scene_name]["pred_human_classes"] = np.ones(
                            self.preds[scene_name]["pred_human_masks"].shape[
                                1
                            ],
                            dtype=np.int64,
                        )
                    human_AP = evaluate_human(
                        self.preds, gt_data_path.replace("gt_part", "gt_human")
                    )

                    low_human_AP = evaluate_human(
                        low_pred, gt_data_path.replace("gt_part", "gt_human")
                    )
                    mid_human_AP = evaluate_human(
                        mid_pred, gt_data_path.replace("gt_part", "gt_human")
                    )
                    high_human_AP = evaluate_human(
                        high_pred, gt_data_path.replace("gt_part", "gt_human")
                    )

                else:
                    for scene_name in self.preds:
                        self.preds[scene_name]["pred_human_classes"] = np.ones(
                            self.preds[scene_name]["pred_masks"].shape[1],
                            dtype=np.int64,
                        )
                        self.preds[scene_name][
                            "pred_human_scores"
                        ] = self.preds[scene_name].pop("pred_scores")
                        self.preds[scene_name][
                            "pred_human_masks"
                        ] = self.preds[scene_name].pop("pred_masks")
                    human_iou = evaluate_semseg_from_instseg(
                        self.preds, gt_data_path.replace("gt_part", "gt_human")
                    )

                    low_human_iou = evaluate_semseg_from_instseg(
                        low_pred, gt_data_path.replace("gt_part", "gt_human")
                    )
                    mid_human_iou = evaluate_semseg_from_instseg(
                        mid_pred, gt_data_path.replace("gt_part", "gt_human")
                    )
                    high_human_iou = evaluate_semseg_from_instseg(
                        high_pred, gt_data_path.replace("gt_part", "gt_human")
                    )

                    human_AP = evaluate_human(
                        self.preds, gt_data_path.replace("gt_part", "gt_human")
                    )

                    low_human_AP = evaluate_human(
                        low_pred, gt_data_path.replace("gt_part", "gt_human")
                    )
                    mid_human_AP = evaluate_human(
                        mid_pred, gt_data_path.replace("gt_part", "gt_human")
                    )
                    high_human_AP = evaluate_human(
                        high_pred, gt_data_path.replace("gt_part", "gt_human")
                    )

                ap_results[f"{log_prefix}_AP_50_human"] = float(
                    human_AP["all_ap_50%"]
                )
                ap_results[f"{log_prefix}_AP_25_human"] = float(
                    human_AP["all_ap_25%"]
                )
                ap_results[f"{log_prefix}_AP_human"] = float(
                    human_AP["all_ap"]
                )

                ap_results[f"{log_prefix}_low_AP_50_human"] = float(
                    low_human_AP["all_ap_50%"]
                )
                ap_results[f"{log_prefix}_low_AP_25_human"] = float(
                    low_human_AP["all_ap_25%"]
                )
                ap_results[f"{log_prefix}_low_AP_human"] = float(
                    low_human_AP["all_ap"]
                )

                ap_results[f"{log_prefix}_mid_AP_50_human"] = float(
                    mid_human_AP["all_ap_50%"]
                )
                ap_results[f"{log_prefix}_mid_AP_25_human"] = float(
                    mid_human_AP["all_ap_25%"]
                )
                ap_results[f"{log_prefix}_mid_AP_human"] = float(
                    mid_human_AP["all_ap"]
                )

                ap_results[f"{log_prefix}_high_AP_50_human"] = float(
                    high_human_AP["all_ap_50%"]
                )
                ap_results[f"{log_prefix}_high_AP_25_human"] = float(
                    high_human_AP["all_ap_25%"]
                )
                ap_results[f"{log_prefix}_high_AP_human"] = float(
                    high_human_AP["all_ap"]
                )

                ap_results[f"{log_prefix}_iou_background"] = float(
                    human_iou[0]
                )
                ap_results[f"{log_prefix}_iou_human"] = float(human_iou[1])
                ap_results[f"{log_prefix}_mean_iou"] = float(human_iou.mean())

                ap_results[f"{log_prefix}_low_iou_background"] = float(
                    low_human_iou[0]
                )
                ap_results[f"{log_prefix}_low_iou_human"] = float(
                    low_human_iou[1]
                )
                ap_results[f"{log_prefix}_low_mean_iou"] = float(
                    low_human_iou.mean()
                )

                ap_results[f"{log_prefix}_mid_iou_background"] = float(
                    mid_human_iou[0]
                )
                ap_results[f"{log_prefix}_mid_iou_human"] = float(
                    mid_human_iou[1]
                )
                ap_results[f"{log_prefix}_mid_mean_iou"] = float(
                    mid_human_iou.mean()
                )

                ap_results[f"{log_prefix}_high_iou_background"] = float(
                    high_human_iou[0]
                )
                ap_results[f"{log_prefix}_high_iou_human"] = float(
                    high_human_iou[1]
                )
                ap_results[f"{log_prefix}_high_mean_iou"] = float(
                    high_human_iou.mean()
                )

                if not self.config.data.part2human:
                    AP_P, PCP = evaluate_mhp(
                        self.preds,
                        gt_data_path,
                        pred_path,
                        dataset="human_segmentation",
                    )
                    part_iou = evaluate_bodysemseg(self.preds, gt_data_path)

                    part_iou_low = evaluate_bodysemseg(low_pred, gt_data_path)
                    part_iou_mid = evaluate_bodysemseg(mid_pred, gt_data_path)
                    part_iou_high = evaluate_bodysemseg(
                        high_pred, gt_data_path
                    )

                    ap_results[f"{log_prefix}_low_mean_part_iou"] = float(
                        part_iou_low.mean()
                    )
                    ap_results[f"{log_prefix}_mid_mean_part_iou"] = float(
                        part_iou_mid.mean()
                    )
                    ap_results[f"{log_prefix}_high_mean_part_iou"] = float(
                        part_iou_high.mean()
                    )

                    AP_P_low, PCP_low = evaluate_mhp(
                        low_pred,
                        gt_data_path,
                        pred_path,
                        dataset="human_segmentation",
                    )
                    AP_P_mid, PCP_mid = evaluate_mhp(
                        mid_pred,
                        gt_data_path,
                        pred_path,
                        dataset="human_segmentation",
                    )
                    AP_P_high, PCP_high = evaluate_mhp(
                        high_pred,
                        gt_data_path,
                        pred_path,
                        dataset="human_segmentation",
                    )

                    ap_results[f"{log_prefix}_low_AP_50_parts"] = float(
                        AP_P_low["all_ap_50%"]
                    )
                    ap_results[f"{log_prefix}_low_AP_25_parts"] = float(
                        AP_P_low["all_ap_25%"]
                    )
                    ap_results[f"{log_prefix}_low_AP_parts"] = float(
                        AP_P_low["all_ap"]
                    )

                    ap_results[f"{log_prefix}_low_PCP_50"] = float(
                        PCP_low["all_ap_50%"]
                    )
                    ap_results[f"{log_prefix}_low_PCP_25"] = float(
                        PCP_low["all_ap_25%"]
                    )
                    ap_results[f"{log_prefix}_low_PCP"] = float(
                        PCP_low["all_ap"]
                    )

                    ap_results[f"{log_prefix}_mid_AP_50_parts"] = float(
                        AP_P_mid["all_ap_50%"]
                    )
                    ap_results[f"{log_prefix}_mid_AP_25_parts"] = float(
                        AP_P_mid["all_ap_25%"]
                    )
                    ap_results[f"{log_prefix}_mid_AP_parts"] = float(
                        AP_P_mid["all_ap"]
                    )

                    ap_results[f"{log_prefix}_mid_PCP_50"] = float(
                        PCP_mid["all_ap_50%"]
                    )
                    ap_results[f"{log_prefix}_mid_PCP_25"] = float(
                        PCP_mid["all_ap_25%"]
                    )
                    ap_results[f"{log_prefix}_mid_PCP"] = float(
                        PCP_mid["all_ap"]
                    )

                    ap_results[f"{log_prefix}_high_AP_50_parts"] = float(
                        AP_P_high["all_ap_50%"]
                    )
                    ap_results[f"{log_prefix}_high_AP_25_parts"] = float(
                        AP_P_high["all_ap_25%"]
                    )
                    ap_results[f"{log_prefix}_high_AP_parts"] = float(
                        AP_P_high["all_ap"]
                    )

                    ap_results[f"{log_prefix}_high_PCP_50"] = float(
                        PCP_high["all_ap_50%"]
                    )
                    ap_results[f"{log_prefix}_high_PCP_25"] = float(
                        PCP_high["all_ap_25%"]
                    )
                    ap_results[f"{log_prefix}_high_PCP"] = float(
                        PCP_high["all_ap"]
                    )

                    ap_results[f"{log_prefix}_AP_50_parts"] = float(
                        AP_P["all_ap_50%"]
                    )
                    ap_results[f"{log_prefix}_AP_25_parts"] = float(
                        AP_P["all_ap_25%"]
                    )
                    ap_results[f"{log_prefix}_AP_parts"] = float(
                        AP_P["all_ap"]
                    )

                    ap_results[f"{log_prefix}_PCP_50"] = float(
                        PCP["all_ap_50%"]
                    )
                    ap_results[f"{log_prefix}_PCP_25"] = float(
                        PCP["all_ap_25%"]
                    )
                    ap_results[f"{log_prefix}_PCP"] = float(PCP["all_ap"])

                    ap_results[f"{log_prefix}_mean_part_iou"] = float(
                        part_iou.mean()
                    )

                print("EVALUATION RESULTS")
                print(ap_results)

                self.log_dict(ap_results)

                if not self.config.general.export:
                    shutil.rmtree(base_path)

                del self.preds
                del self.bbox_preds
                del self.bbox_gt

                gc.collect()

                self.preds = dict()
                self.bbox_preds = dict()
                self.bbox_gt = dict()

                return

            with open(pred_path, "r") as fin:
                for line_id, line in enumerate(fin):
                    if line_id == 0:
                        # ignore header
                        continue
                    class_name, _, ap, ap_50, ap_25 = line.strip().split(",")

                    ap_results[f"{log_prefix}_{class_name}_val_ap"] = float(ap)
                    ap_results[f"{log_prefix}_{class_name}_val_ap_50"] = float(
                        ap_50
                    )
                    ap_results[f"{log_prefix}_{class_name}_val_ap_25"] = float(
                        ap_25
                    )

            mean_ap = statistics.mean(
                [
                    item
                    for key, item in ap_results.items()
                    if key.endswith("val_ap")
                ]
            )
            mean_ap_50 = statistics.mean(
                [
                    item
                    for key, item in ap_results.items()
                    if key.endswith("val_ap_50")
                ]
            )
            mean_ap_25 = statistics.mean(
                [
                    item
                    for key, item in ap_results.items()
                    if key.endswith("val_ap_25")
                ]
            )

            ap_results[f"{log_prefix}_mean_ap"] = mean_ap
            ap_results[f"{log_prefix}_mean_ap_50"] = mean_ap_50
            ap_results[f"{log_prefix}_mean_ap_25"] = mean_ap_25

            ap_results = {
                key: 0.0 if math.isnan(score) else score
                for key, score in ap_results.items()
            }
        except (IndexError, OSError) as e:
            print(f"NO SCORES: {e}")
            ap_results[f"{log_prefix}_mean_ap"] = 0.0
            ap_results[f"{log_prefix}_mean_ap_50"] = 0.0
            ap_results[f"{log_prefix}_mean_ap_25"] = 0.0

        self.log_dict(ap_results)

        if not self.config.general.export:
            shutil.rmtree(base_path)

        del self.preds
        del self.bbox_preds
        del self.bbox_gt

        gc.collect()

        self.preds = dict()
        self.bbox_preds = dict()
        self.bbox_gt = dict()

    def test_epoch_end(self, outputs):
        if self.config.general.export:
            return

        self.eval_instance_epoch_end()

        dd = defaultdict(list)
        for output in outputs:
            for key, val in output.items():  # .items() in Python 3.
                dd[key].append(val)

        dd = {k: statistics.mean(v) for k, v in dd.items()}

        dd["val_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_ce" in k]]
        )
        dd["val_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_mask" in k]]
        )
        dd["val_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_dice" in k]]
        )

        self.log_dict(dd)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(
            self.config.data.train_dataset
        )
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(
            self.config.data.test_dataset
        )
        self.labels_info = self.train_dataset.label_info

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
