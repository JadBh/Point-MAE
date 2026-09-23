import time

import mlflow
import torch
from point_mae.datasets import data_transforms
from point_mae.tools import builder
from point_mae.utils.AverageMeter import AverageMeter
from point_mae.utils.logger import *
from rebel_lod.scripts.autoencoder_scripts.autoencoding_utils import (
    log_batched_reconstruction_similarity_metrics,
)
from torch import nn
from torchvision import transforms

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.0):
        if type(acc).__name__ == "dict":
            self.acc = acc["acc"]
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict["acc"] = self.acc
        return _dict


def train_network(config):
    logger = get_logger(config.log_name)
    # build dataset
    (
        (_, train_dataloader),
        (_, test_dataloader),
    ) = (
        builder.dataset_builder(config, config.dataset.train),
        builder.dataset_builder(config, config.dataset.val),
    )
    n_batches = len(train_dataloader)

    base_model = builder.model_builder(config.model)
    if config.train_decoder_only:
        builder.load_model(base_model, config.chkpt_path, logger=logger)
        logger.info(f"loaded weights from {config.chkpt_path}")
    if config.use_gpu:
        base_model.to(config.local_rank)
    base_model = nn.DataParallel(base_model).cuda()
    base_model.zero_grad()

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.0)
    metrics = Acc_Metric(0.0)

    # freeze encoder
    if config.train_decoder_only:
        for name, param in base_model.module.MAE_encoder.named_parameters():
            param.requires_grad = False
        print_log("* Froze all encoder layers *", logger=logger)

    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()
        if config.train_decoder_only:
            base_model.module.MAE_encoder.eval()

        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        num_iter = 0

        cum_train_loss = 0.0
        num_train_samples = 0

        all_reconstruction = []
        all_reconstruction_gt = []
        all_complete_reconstructed = []
        all_complete_original = []

        for idx, (taxonomy_ids, sample_id, data) in enumerate(train_dataloader):
            batch_size = len(data)

            mlflow_step = epoch * n_batches + idx
            mlflow.log_metric("epoch", epoch, step=mlflow_step)

            num_iter += 1
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints

            points = data.cuda()

            assert points.size(1) == npoints
            points = train_transforms(points)
            (
                reconstruction,
                reconstruction_gt,
                complete_reconstructed,
                complete_original,
                loss,
            ) = base_model(points)

            cum_train_loss += loss.item() * batch_size
            num_train_samples += batch_size

            mlflow.log_metric("train/batch/loss", loss.item(), step=mlflow_step)

            # log_batched_reconstruction_similarity_metrics(
            #     reconstruction,
            #     reconstruction_gt,
            #     complete_reconstructed,
            #     complete_original,
            #     mlflow_step,
            #     run_type="train",
            #     scope="batch",
            # )

            all_reconstruction.append(reconstruction.detach().cpu())
            all_reconstruction_gt.append(reconstruction_gt.detach().cpu())
            all_complete_reconstructed.append(complete_reconstructed.detach().cpu())
            all_complete_original.append(complete_original.detach().cpu())

            loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f"
                    % (
                        epoch,
                        config.max_epoch,
                        idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        "%.4f" % loss.item(),
                        optimizer.param_groups[0]["lr"],
                    ),
                    logger=logger,
                )
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        builder.save_checkpoint(
            base_model,
            optimizer,
            epoch,
            metrics,
            best_metrics,
            "ckpt-last",
            config,
            logger=logger,
        )
        avg_train_loss = cum_train_loss / num_train_samples

        all_reconstruction = torch.cat(all_reconstruction, dim=0)
        all_reconstruction_gt = torch.cat(all_reconstruction_gt, dim=0)
        all_complete_reconstructed = torch.cat(all_complete_reconstructed, dim=0)
        all_complete_original = torch.cat(all_complete_original, dim=0)

        mlflow.log_metric(
            "train/epoch/loss",
            avg_train_loss,
            step=epoch,
        )

        log_batched_reconstruction_similarity_metrics(
            all_reconstruction,
            all_reconstruction_gt,
            all_complete_reconstructed,
            all_complete_original,
            mlflow_step=epoch,
            run_type="train",
            scope="epoch",
        )

        avg_val_loss = validate(
            base_model=base_model,
            test_dataloader=test_dataloader,
            epoch=epoch,
            config=config,
            logger=logger,
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            builder.save_checkpoint(
                base_model,
                optimizer,
                epoch,
                metrics,
                best_metrics,
                "ckpt-best",
                config,
                logger=logger,
            )


def validate(
    base_model,
    test_dataloader,
    epoch,
    config,
    logger=None,
):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)

    base_model.eval()

    cum_val_loss = 0.0
    num_val_samples = 0

    all_reconstruction = []
    all_reconstruction_gt = []
    all_complete_reconstructed = []
    all_complete_original = []

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            batch_size = len(data)

            npoints = config.dataset.val.others.npoints
            points = data.cuda()
            assert points.size(1) == npoints

            (
                reconstruction,
                reconstruction_gt,
                complete_reconstructed,
                complete_original,
                loss,
            ) = base_model(points)

            cum_val_loss += loss.item() * batch_size
            num_val_samples += batch_size

            all_reconstruction.append(reconstruction.detach().cpu())
            all_reconstruction_gt.append(reconstruction_gt.detach().cpu())
            all_complete_reconstructed.append(complete_reconstructed.detach().cpu())
            all_complete_original.append(complete_original.detach().cpu())

    all_reconstruction = torch.cat(all_reconstruction, dim=0)
    all_reconstruction_gt = torch.cat(all_reconstruction_gt, dim=0)
    all_complete_reconstructed = torch.cat(all_complete_reconstructed, dim=0)
    all_complete_original = torch.cat(all_complete_original, dim=0)

    log_batched_reconstruction_similarity_metrics(
        all_reconstruction,
        all_reconstruction_gt,
        all_complete_reconstructed,
        all_complete_original,
        mlflow_step=epoch,
        run_type="eval",
        scope="epoch",
    )

    avg_val_loss = cum_val_loss / num_val_samples

    mlflow.log_metric(
        "eval/epoch/loss",
        avg_val_loss,
        step=epoch,
    )

    print_log(
        "[Validation] EPOCH: %d Loss = %.4f" % (epoch, avg_val_loss),
        logger=logger,
    )

    return avg_val_loss
