"""
this copy of 'runner_inference_decoder' has one main difference: it loads two embeddings
A and B, and takes the patch centers from A and replaces it with the patch centers
from B (the last 3 dimensions), then decodes B.

This is to test the importance of patch centers in decoding.
"""

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


def get_index_from_embedding_name(embedding_name: str) -> int:
    """
    Given the name of an embedding file, extract the identifying number from it.

    For example: 'generated_sample_for_centers_86' returns 86. Naming convention is that
    generated embeddings using patch centers from embedding_x carries the same number
    x in its name.
    """
    return int(embedding_name.split("_")[-1])


def infer_decoder_switch(config):

    embedding_A_dataset = EmbeddingsDiffusionDataset(
        files_dir=config.embedding_A_loading_path,
        device=config.gpu_device,
    )

    embedding_B_dataset = EmbeddingsDiffusionDataset(
        files_dir=config.embedding_B_loading_path,
        device=config.gpu_device,
    )

    assert len(embedding_A_dataset) == len(embedding_B_dataset), (
        "number of embeddings to decode does not match number of conditioning centers"
    )

    combined_embeddings = []
    for i in range(len(embedding_B_dataset)):
        embedding_A, embedding_A_name = embedding_A_dataset[i]
        embedding_B, embedding_B_name = embedding_B_dataset[i]

        assert get_index_from_embedding_name(
            embedding_B_name
        ) == get_index_from_embedding_name(embedding_A_name)

        patch_centers = embedding_A[:, -3:]
        # combined_embedding = torch.cat((embedding_B, patch_centers), dim=1)
        combined_embedding = embedding_B

        combined_embeddings.append((combined_embedding, embedding_B_name))

    embeddings_dataloader = DataLoader(combined_embeddings, batch_size=1, shuffle=False)

    base_model = builder.model_builder(config.model)

    builder.load_decoder_weights(base_model, config.chkpt_path, logger=logger)

    if config.use_gpu:
        base_model.to(config.local_rank)

    base_model.eval()  # set model to eval mode

    with torch.no_grad():
        for data, name in embeddings_dataloader:
            """
            data is concatination of embedding and their centers in 3D.
            Expected shape: B, 64, (384+3)
            """
            name = name[0]
            data = data.cuda()
            B, _, _ = data.shape

            if config.vis:
                if B != 1:
                    raise ValueError(
                        f"Batch size should be 1 for visualization \
                                    but recieved: {B}"
                    )

                reconstruction, centers = base_model(data)

                output_plot_path = Path(
                    config.embedding_B_loading_path, "reconstruction_plots"
                )
                output_plot_path.mkdir(parents=True, exist_ok=True)

                centers = centers.squeeze().detach().cpu().numpy()
                centers = misc.get_ptcloud_img(centers, 0, 0, draw_axes=True)
                centers_img = np.ascontiguousarray(centers[150:650, 150:675, :])

                reconstruction = reconstruction.squeeze().detach().cpu().numpy()
                reconstruction = misc.get_ptcloud_img(reconstruction, 0, 0)
                reconstruction_img = np.ascontiguousarray(
                    reconstruction[150:650, 150:675, :]
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
                    "Reconstruction from embeddings, centers switched with different embedding",
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
                    output_plot_path, f"vis_{name}_switch_centers.jpg"
                )
                print("image saving path: ", img_path)
                cv2.imwrite(img_path, img)

            else:
                reconstruction, centers = base_model(data)

        return
