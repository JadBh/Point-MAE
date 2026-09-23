import os
from pathlib import Path

import cv2
import numpy as np
import torch
from point_mae.tools import builder
from point_mae.utils import misc
from point_mae.utils.logger import *


def test_net(config):
    logger = get_logger(config.log_name)
    print_log("Tester start ... ", logger=logger)
    _, test_dataloader = builder.dataset_builder(config, config.dataset.complete)

    base_model = builder.model_builder(config.model)

    builder.load_model(base_model, config.chkpt_path, logger=logger)

    if config.use_gpu:
        base_model.to(config.local_rank)

    #  DDP
    if config.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, config, logger=logger)


# visualization
def test(base_model, test_dataloader, config, logger=None):

    base_model.eval()  # set model to eval mode
    useful_cate = [
        "02691156",  # plane
        "04379243",  # table
        "03790512",  # motorbike
        "03948459",  # pistol
        "03642806",  # laptop
        "03467517",  # guitar
        "03261776",  # earphone
        "03001627",  # chair
        "02958343",  # car
        "04090263",  # rifle
        "03759954",  # microphone
        "no_taxonomy",  # swiss surface
    ]
    with torch.no_grad():
        total_loss = 0.0
        num_evaluated_items = 0

        for idx, (taxonomy_ids, sample_id, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            if taxonomy_ids[0] not in useful_cate:
                continue
            if taxonomy_ids[0] == "02691156":
                a, b = 90, 135
            elif taxonomy_ids[0] == "04379243":
                a, b = 30, 30
            elif taxonomy_ids[0] == "03642806":
                a, b = 30, -45
            elif taxonomy_ids[0] == "03467517":
                a, b = 0, 90
            elif taxonomy_ids[0] == "03261776":
                a, b = 0, 75
            elif taxonomy_ids[0] == "03001627":
                a, b = 30, -45
            elif taxonomy_ids[0] == "no_taxonomy":
                a, b = 110, -90
            else:
                a, b = 0, 0

            dataset_name = config.dataset.complete._base_.NAME
            if dataset_name == "ShapeNet":
                points = data.cuda()
            elif dataset_name == "SwissSurface":
                points = data.cuda()
            else:
                raise NotImplementedError(f"Train phase do not support {dataset_name}")

            if config.vis:
                reconstruction, vis_points_gt, model_input, centers, loss = base_model(
                    points, sample_id, vis=config.vis
                )
                data_path = Path(
                    config.dataset.complete._base_.PC_PATH, "visualization"
                )

                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                points = points.squeeze().detach().cpu().numpy()

                np.savetxt(os.path.join(data_path, "gt.txt"), points, delimiter=";")
                # gt_img = misc.get_ptcloud_img(points, a, b)
                # gt_img = np.ascontiguousarray(gt_img[150:650, 150:675, :])

                model_input = (
                    model_input.squeeze().detach().cpu().numpy().reshape(-1, 3)
                )

                model_input_img = misc.get_ptcloud_img(model_input, a, b)
                model_input_img = np.ascontiguousarray(
                    model_input_img[150:650, 150:675, :]
                )

                # centers = centers.squeeze().detach().cpu().numpy()
                # np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
                # centers = misc.get_ptcloud_img(centers, a, b)
                # final_image.append(centers[150:650,150:675,:])

                vis_points_gt = vis_points_gt.squeeze().detach().cpu().numpy()

                np.savetxt(
                    os.path.join(data_path, "vis.txt"), vis_points_gt, delimiter=";"
                )
                vis_img = misc.get_ptcloud_img(vis_points_gt, a, b)
                vis_img = np.ascontiguousarray(vis_img[150:650, 150:675, :])

                reconstruction = reconstruction.squeeze().detach().cpu().numpy()

                np.savetxt(
                    os.path.join(data_path, "reconstruction.txt"),
                    reconstruction,
                    delimiter=";",
                )
                dense_img = misc.get_ptcloud_img(reconstruction, a, b)
                dense_img = np.ascontiguousarray(dense_img[150:650, 150:675, :])

                # cv2.putText(
                #     gt_img,
                #     f"Ground Truth {len(points)} pts",
                #     (20, 40),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (0, 0, 0),
                # )

                cv2.putText(
                    model_input_img,
                    f"Input {len(model_input)} pts",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                )

                cv2.putText(
                    vis_img,
                    f"Visible Points {len(vis_points_gt)} pts",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                )

                cv2.putText(
                    dense_img,
                    f"Reconstruction {len(reconstruction)} pts",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                )

                # img = np.concatenate(
                #     [gt_img, model_input_img, vis_img, dense_img], axis=1

                if config.model.decode_visible_patches:
                    img = np.concatenate(
                        [model_input_img, dense_img], axis=1
                        )
                else:
                    img = np.concatenate(
                        [model_input_img, vis_img, dense_img], axis=1
                    )


                # Create a blank header above the image
                line_height = 40
                num_lines = 5
                title_height = line_height * num_lines + 20
                title = np.ones((title_height, img.shape[1], 3), dtype=np.uint8) * 255

                # Parameters to display
                title_lines = [
                    f"model: {config.chkpt_path.parent.name}",
                    f"Mask ratio: {config.model.transformer_config.mask_ratio}",
                    f"Decode (only) visible patches: {config.model.decode_visible_patches}",
                ]
                if config.model.decode_visible_patches:
                    title_lines.append("Decode only visible patches")
                    title_lines.append(
                        f"Loss (reconstruction of visible patches): {loss.item():.3g}"
                    )
                else:
                    title_lines.append("Decode only hidden patches")
                    title_lines.append(
                        f"Loss (reconstruction of hidden patches): {loss.item():.3g}"
                    )

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
                    data_path, f"plot_for_sample_{sample_id.item()}.jpg"
                )
                print("image saving path: ", img_path)
                cv2.imwrite(img_path, img)

            else:
                (
                    reconstruction,
                    reconstruction_gt,
                    complete_reconstructed,
                    complete_original,
                    loss,
                ) = base_model(points, sample_id, vis=config.vis)

            if config.save_reconstructed_pointcloud:
                reconstructed_pointcloud_saving_dir = Path(
                    config.experiment_output_path, "reconstructed_pointclouds"
                )
                reconstructed_pointcloud_saving_dir.mkdir(parents=True, exist_ok=True)
                reconstructed_pointcloud_saving_path = os.path.join(
                    reconstructed_pointcloud_saving_dir,
                    f"reconstruction_of_sample_{sample_id.item()}.pt",
                )
                torch.save(
                    complete_reconstructed.squeeze(),
                    reconstructed_pointcloud_saving_path,
                )
                print_log(
                    f"saved reconstructed pointclouds at {reconstructed_pointcloud_saving_path}"
                )

            batch_size = points.shape[0]
            total_loss += loss.item() * batch_size
            num_evaluated_items += batch_size

            if idx % 100 == 0:
                print_log(
                    f"batch {idx}/{len(test_dataloader)} | "
                    f"mean loss: {total_loss / num_evaluated_items:.5g}",
                    logger=logger,
                )

        if num_evaluated_items > 0:
            print_log(
                f"Tester done: {num_evaluated_items} items evaluated | "
                f"mean loss: {total_loss / num_evaluated_items:.4g}",
                logger=logger,
            )

        return
