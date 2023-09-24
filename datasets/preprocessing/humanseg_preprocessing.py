import numpy as np
import pandas as pd
from base_preprocessing import BasePreprocessing
from fire import Fire
from natsort import natsorted
from plyfile import PlyData


class HumanSegmentationDataset(BasePreprocessing):
    def __init__(
        self,
        data_dir: str,
        save_dir: str,
        dataset: str,
        modes: tuple = ("train", "validation"),
        min_points: int = 0,
        min_instances: int = 0,
        n_jobs: int = -1,
    ):
        if type(modes) is not tuple:
            modes = (modes,)
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.min_points = min_points
        self.min_instances = min_instances

        self.dataset = dataset

        assert self.dataset in [
            "egobody",
            "synthetic",
        ], f"{self.dataset} is not supported."

        self.class_map = {
            "background": 0,
            "human": 1,
        }

        self.color_map = [[0, 0, 255], [0, 255, 0]]  # background  # human

        self.COLOR_MAP_W_BODY_PARTS = {
            -1: (255.0, 255.0, 255.0),
            0: (0.0, 0.0, 0.0),
            1: (174.0, 199.0, 232.0),
            2: (152.0, 223.0, 138.0),
            3: (31.0, 119.0, 180.0),
            4: (255.0, 187.0, 120.0),
            5: (188.0, 189.0, 34.0),
            6: (140.0, 86.0, 75.0),
            7: (255.0, 152.0, 150.0),
            8: (214.0, 39.0, 40.0),
            9: (197.0, 176.0, 213.0),
            10: (148.0, 103.0, 189.0),
            11: (196.0, 156.0, 148.0),
            12: (23.0, 190.0, 207.0),
            14: (247.0, 182.0, 210.0),
            15: (66.0, 188.0, 102.0),
            16: (219.0, 219.0, 141.0),
            17: (140.0, 57.0, 197.0),
            18: (202.0, 185.0, 52.0),
            19: (51.0, 176.0, 203.0),
            20: (200.0, 54.0, 131.0),
            21: (92.0, 193.0, 61.0),
            22: (78.0, 71.0, 183.0),
            23: (172.0, 114.0, 82.0),
            24: (255.0, 127.0, 14.0),
            25: (91.0, 163.0, 138.0),
            26: (153.0, 98.0, 156.0),
            27: (140.0, 153.0, 101.0),
            28: (158.0, 218.0, 229.0),
            29: (100.0, 125.0, 154.0),
            30: (178.0, 127.0, 135.0),
            31: (120.0, 185.0, 128.0),
            32: (146.0, 111.0, 194.0),
            33: (44.0, 160.0, 44.0),
            34: (112.0, 128.0, 144.0),
            35: (96.0, 207.0, 209.0),
            36: (227.0, 119.0, 194.0),
            37: (213.0, 92.0, 176.0),
            38: (94.0, 106.0, 211.0),
            39: (82.0, 84.0, 163.0),
            40: (100.0, 85.0, 144.0),
            41: (0.0, 0.0, 255.0),  # artificial human
            # body parts
            100: (35.0, 69.0, 100.0),  # rightHand
            101: (73.0, 196.0, 37.0),  # rightUpLeg
            102: (121.0, 25.0, 252.0),  # leftArm
            103: (96.0, 237.0, 31.0),  # head
            104: (55.0, 40.0, 93.0),  # leftEye
            105: (75.0, 180.0, 125.0),  # rightEye
            106: (165.0, 38.0, 65.0),  # leftLeg
            107: (63.0, 75.0, 77.0),  # leftToeBase
            108: (27.0, 255.0, 80.0),  # leftFoot
            109: (82.0, 110.0, 90.0),  # spine1
            110: (87.0, 54.0, 10.0),  # spine2
            111: (210.0, 200.0, 110.0),  # leftShoulder
            112: (217.0, 212.0, 76.0),  # rightShoulder
            113: (254.0, 176.0, 234.0),  # rightFoot
            114: (111.0, 140.0, 56.0),  # rightArm
            115: (83.0, 15.0, 157.0),  # leftHandIndex1
            116: (98.0, 255.0, 160.0),  # rightLeg
            117: (153.0, 170.0, 17.0),  # rightHandIndex1
            118: (54.0, 82.0, 122.0),  # leftForeArm
            119: (10.0, 19.0, 94.0),  # rightForeArm
            120: (1.0, 147.0, 72.0),  # neck
            121: (47.0, 210.0, 21.0),  # rightToeBase
            122: (174.0, 22.0, 133.0),  # spine
            123: (98.0, 58.0, 83.0),  # leftUpLeg
            124: (222.0, 25.0, 45.0),  # leftHand
            125: (75.0, 233.0, 65.0),  # hips
        }

        # in each scene there are at most 10 human instances
        self.COLOR_MAP_INSTANCES = {
            0: (0.0, 0.0, 0.0),
            1: (255.0, 0.0, 0.0),
            2: (0.0, 255.0, 0.0),
            3: (0.0, 0.0, 255.0),
            4: (255.0, 255.0, 0.0),
            5: (255.0, 0.0, 255.0),
            6: (0.0, 255.0, 255.0),
            7: (255.0, 204.0, 153.0),
            8: (255.0, 102.0, 0.0),
            9: (0.0, 128.0, 128.0),
            10: (153.0, 153.0, 255.0),
        }

        self.ORIG_BODY_PART_IDS = set(range(100, 126))

        self.LABEL_MAP = {
            0: "background",
            1: "rightHand",
            2: "rightUpLeg",
            3: "leftArm",
            4: "head",
            5: "leftLeg",
            6: "leftFoot",
            7: "torso",
            8: "rightFoot",
            9: "rightArm",
            10: "leftHand",
            11: "rightLeg",
            12: "leftForeArm",
            13: "rightForeArm",
            14: "leftUpLeg",
            15: "hips",
        }

        self.LABEL_MAPPER_FOR_BODY_PART_SEGM = {
            -1: 0,
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
            21: 0,
            22: 0,
            23: 0,
            24: 0,
            25: 0,
            26: 0,
            27: 0,
            28: 0,
            29: 0,
            30: 0,
            31: 0,
            32: 0,
            33: 0,
            34: 0,
            35: 0,
            36: 0,
            37: 0,
            38: 0,
            39: 0,
            40: 0,
            41: 0,  # background
            100: 1,  # rightHand
            101: 2,  # rightUpLeg
            102: 3,  # leftArm
            103: 4,  # head
            104: 4,  # head
            105: 4,  # head
            106: 5,  # leftLeg
            107: 6,  # leftFoot
            108: 6,  # leftFoot
            109: 7,  # torso
            110: 7,  # torso
            111: 7,  # torso
            112: 7,  # torso
            113: 8,  # rightFoot
            114: 9,  # rightArm
            115: 10,  # leftHand
            116: 11,  # rightLeg
            117: 1,  # rightHand
            118: 12,  # leftForeArm
            119: 13,  # rightForeArm
            120: 4,  # head
            121: 8,  # rightFoot
            122: 7,  # torso
            123: 14,  # leftUpLeg
            124: 10,  # leftHand
            125: 15,  # hips
        }

        self.create_label_database()

        for mode in self.modes:
            with open(f"{data_dir}/{mode}_list.txt") as file:
                if self.dataset == "egobody" and mode == "validation":
                    self.files[mode] = natsorted(
                        [
                            f"{self.data_dir}/validation/{line.strip()}"
                            for line in file
                        ]
                    )
                else:
                    self.files[mode] = natsorted(
                        [
                            f"{self.data_dir}/scenes/{line.strip()}"
                            for line in file
                        ]
                    )

    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                "color": self.color_map[class_id],
                "name": class_name,
                "validation": True,
            }
        self._save_yaml(self.save_dir / "label_database.yaml", label_database)

        part_database = dict()
        # part_map = {i: part_name for i, part_name in enumerate(self.LABEL_LIST)}
        inv_label_map = {
            v: k for k, v in self.LABEL_MAPPER_FOR_BODY_PART_SEGM.items()
        }
        inv_label_map[0] = 0
        for part_id, part_name in self.LABEL_MAP.items():
            part_database[part_id] = {
                "color": self.COLOR_MAP_W_BODY_PARTS[inv_label_map[part_id]],
                "name": part_name,
                "validation": True,
            }
        self._save_yaml(self.save_dir / "part_database.yaml", part_database)

    def read_plyfile(self, file_path):
        """Read ply file and return it as numpy array. Returns None if emtpy."""
        with open(file_path, "rb") as f:
            plydata = PlyData.read(f)
        if plydata.elements:
            return pd.DataFrame(plydata.elements[0].data).values

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        if self.dataset == "egobody":
            scene_name = "_".join(filepath.split("/")[-3:]).replace(".ply", "")
        else:
            scene_name = "_".join(filepath.split("/")[-2:]).replace(".ply", "")

        filebase = {
            "filepath": filepath,
            "scene": scene_name,
            "raw_filepath": str(filepath),
        }

        # reading both files and checking that they are fitting
        pcd = self.read_plyfile(filepath)
        coords = pcd[:, :3]

        # fix rotation bug
        coords = coords[:, [0, 2, 1]]
        coords[:, 2] = -coords[:, 2]

        rgb = pcd[:, 3:6]
        instance_id = pcd[:, 6][..., None]

        if (
            coords.shape[0] < self.min_points
            or np.unique(instance_id[:, 0]).shape[0] <= self.min_instances
        ):
            return scene_name

        part_id = pcd[:, 7][..., None]

        part_id = np.array(
            [
                self.LABEL_MAPPER_FOR_BODY_PART_SEGM[int(part_id[i, 0])]
                for i in range(part_id.shape[0])
            ]
        )[..., None].astype(np.float32)

        points = np.hstack((coords, rgb, part_id, instance_id))

        if np.isinf(points).sum() > 0:
            # some scenes (scene0573_01_frame_04) got nans
            return scene_name

        gt_part = part_id * 1000 + instance_id
        gt_human = (part_id > 0.0) * 1000 + instance_id

        processed_filepath = self.save_dir / mode / f"{scene_name}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = (
            self.save_dir / "gt_human" / mode / f"{scene_name}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_human.astype(np.int32), fmt="%d")
        filebase["gt_human_filepath"] = str(processed_gt_filepath)

        processed_gt_filepath = (
            self.save_dir / "gt_part" / mode / f"{scene_name}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_part.astype(np.int32), fmt="%d")
        filebase["gt_part_filepath"] = str(processed_gt_filepath)

        return filebase

    def joint_database(self, train_modes=("train",)):
        for mode in train_modes:
            joint_db = []
            for let_out in train_modes:
                if mode == let_out:
                    continue
                joint_db.extend(
                    self._load_yaml(
                        self.save_dir / (let_out + "_database.yaml")
                    )
                )
            self._save_yaml(
                self.save_dir / f"train_{mode}_database.yaml", joint_db
            )

    def compute_color_mean_std(
        self,
        train_database_path: str = "./data/processed/scannet/train_database.yaml",
    ):
        pass


if __name__ == "__main__":
    Fire(HumanSegmentationDataset)
