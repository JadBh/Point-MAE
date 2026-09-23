import os
from pathlib import Path

import cv2
import numpy as np
import torch
from point_mae.tools import builder
from point_mae.utils import misc
from point_mae.utils.logger import *
from rebel_lod import logger
from rebel_lod.datasets.EmbeddingsDiffusionDataset import EmbeddingsDiffusionDataset
from torch.utils.data import DataLoader


def infer_decoder(config):

    embeddings_dataset = EmbeddingsDiffusionDataset(
        files_dir=config.embeddings_loading_path,
        device=config.gpu_device,
    )

    embeddings_dataloader = DataLoader(embeddings_dataset, batch_size=1, shuffle=False)

    base_model = builder.model_builder(config.model)

    builder.load_decoder_weights(base_model, config.chkpt_path, logger=logger)

    if config.use_gpu:
        base_model.to(config.local_rank)

    base_model.eval()  # set model to eval mode

    output_path = Path(
        config.embeddings_loading_path,
        "pointcloud_reconstruction_from_embeddings",
    )
    output_path.mkdir(parents=True, exist_ok=True)
    output_plots_path = Path(output_path, "reconstruction_plots")
    output_pointclouds_path = Path(output_path, "reconstructed_pointclouds")

    with torch.no_grad():
        for data, name in embeddings_dataloader:
            """
            data is concatination of embedding and their centers in 3D.
            Expected shape: B, 64, (384+3)
            """
            name = name[0]
            data = data.cuda()
            B, _, _ = data.shape

            if B != 1:
                raise ValueError(
                    f"Batch size should be 1 for decoder \
                                but recieved: {B}"
                )

            reconstruction, centers = base_model(data)

            if config.vis:
                output_plots_path.mkdir(parents=True, exist_ok=True)

                roll, pitch = 110, -90

                centers_img = centers.squeeze().detach().cpu().numpy()
                centers_img = misc.get_ptcloud_img(
                    centers_img, roll, pitch, draw_axes=True
                )
                centers_img = np.ascontiguousarray(centers_img[150:650, 150:675, :])

                reconstruction_img = reconstruction.squeeze().detach().cpu().numpy()
                reconstruction_img = misc.get_ptcloud_img(
                    reconstruction_img, roll, pitch
                )
                reconstruction_img = np.ascontiguousarray(
                    reconstruction_img[150:650, 150:675, :]
                )

                cv2.putText(
                    centers_img,
                    "Centers",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                )

                cv2.putText(
                    reconstruction_img,
                    "Reconstruction",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                )

                img = np.concatenate([centers_img, reconstruction_img], axis=1)

                # Create a blank header above the image
                line_height = 40
                num_lines = 2
                title_height = line_height * num_lines + 20
                title = np.ones((title_height, img.shape[1], 3), dtype=np.uint8) * 255

                # Parameters to display
                title_lines = [
                    "Reconstruction from embeddings",
                    f"model: {config.chkpt_path.parent.name}",
                ]

                # Draw each line
                for i, text in enumerate(title_lines):
                    y = 35 + i * line_height

                    cv2.putText(
                        title,
                        text,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # font size
                        (0, 0, 0),  # black
                        2,  # thickness
                        cv2.LINE_AA,
                    )

                # Put title above the combined image
                img = np.concatenate([title, img], axis=0)

                img_path = os.path.join(
                    output_plots_path, f"plot_for_sample_{name}.jpg"
                )

                print("image saving path: ", img_path)
                cv2.imwrite(img_path, img)

            if config.save_output_pointclouds:
                output_pointclouds_path.mkdir(parents=True, exist_ok=True)

                pointcloud_path = Path(
                    output_pointclouds_path,
                    f"pointcloud_for_sample_{name}.pt",
                )

                torch.save(
                    reconstruction.squeeze().detach().cpu(),
                    pointcloud_path,
                )
                print(f"saved {pointcloud_path}")
