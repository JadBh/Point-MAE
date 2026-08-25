import os

import numpy as np
import torch
import torch.utils.data as data
from point_mae.utils.logger import *

from .build import DATASETS
from .io import IO


@DATASETS.register_module()
class SwissSurface(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset

        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")
        test_data_list_file = os.path.join(self.data_root, "test.txt")

        self.sample_points_num = config.npoints
        self.whole = config.get("whole")

        print_log(
            f"[DATASET] sample out {self.sample_points_num} points",
            logger="SwissSurface",
        )
        print_log(f"[DATASET] Open file {self.data_list_file}", logger="SwissSurface")
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, "r") as f:
                test_lines = f.readlines()
            print_log(
                f"[DATASET] Open file {test_data_list_file}", logger="SwissSurface"
            )
            lines = test_lines + lines
        self.file_list = []
        skipped = 0
        for line in lines:
            line = line.strip()
            if (
                IO.get(os.path.join(self.pc_path, line)).xyz.shape[0]
                < self.sample_points_num
            ):
                skipped += 1
                continue
            taxonomy_id = "no_taxonomy"
            model_id = "no_model_id"
            self.file_list.append(
                {"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": line}
            )
        print(f"skipped {skipped} files")
        print_log(
            f"[DATASET] {len(self.file_list)} instances were loaded",
            logger="SwissSurface",
        )

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        idx = np.random.choice(pc.shape[0], num, replace=False)
        return pc[idx]

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = IO.get(os.path.join(self.pc_path, sample["file_path"])).xyz
        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample["taxonomy_id"], sample["model_id"], data

    def __len__(self):
        return len(self.file_list)
