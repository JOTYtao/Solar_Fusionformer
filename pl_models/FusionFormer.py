import torch
import wandb
from torchmetrics import MeanMetric
import os
from pl_models.utils import ContextMixerModule
import numpy as np

class FusionFormer(ContextMixerModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metrics: dict,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(model, optimizer, scheduler, metrics, criterion, kwargs=kwargs)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True):
        out, bands_weights, feature_weights, attn_output_weights = self.model(
            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask
        )
        return out, bands_weights, feature_weights, attn_output_weights

    def training_step(self, train_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(train_batch)

        y_hat, bands_weights, feature_weights, attn_output_weights = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True)
        # y_hat = y_hat.mean(dim=2)

        loss = self.criterion(y_hat, y_ts)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.train:
            metric = getattr(self, f"train_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"train/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, val_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(val_batch)

        y_hat, bands_weights, feature_weights, attn_output_weights = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=False)
        # y_hat = y_hat.mean(dim=2)


        loss = self.criterion(y_hat, y_ts)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {"predictions": y_hat, "ground_truth": y_ts}


    def test_step(self, test_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(test_batch)
        y_hat, bands_weights, feature_weights, attn_output_weights = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=False)

        # os.makedirs(save_dir, exist_ok=True)
        # np.save(os.path.join(save_dir, f'out_batch_{batch_idx}.npy'), y_hat.cpu().numpy())
        # np.save(os.path.join(save_dir, f'attn_batch_{batch_idx}.npy'), attn.cpu().numpy())
        # np.save(os.path.join(save_dir, f'bands_weights_batch_{batch_idx}.npy'), bands_weights.cpu().numpy())
        # np.save(os.path.join(save_dir, f'satellite_weights_batch_{batch_idx}.npy'), satellite_weights.cpu().numpy())
        # np.save(os.path.join(save_dir, f'feature_weights_batch_{batch_idx}.npy'), feature_weights.cpu().numpy())
        # np.save(os.path.join(save_dir, f'attn_output_weights_batch_{batch_idx}.npy'), attn_output_weights.cpu().numpy())

        loss = self.criterion(y_hat, y_ts)

        self.val_loss(loss)
        self.log("test/loss", self.val_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        # return {"predictions": y_hat, "ground_truth": y_ts}

        return {"predictions": y_hat,
                "ground_truth": y_ts,
                "bands_weights": bands_weights,
                "feature_weights": feature_weights,
                "attn_output_weights": attn_output_weights,
                "time_coords": time_coords
            }

    def test_epoch_end(self, outputs):

        save_dir = 'E:/research/my_code/solar_forecasting/EMD/results/'
        os.makedirs(save_dir, exist_ok=True)
        # all_predictions, all_ground_truths, all_time_coords = [], [], []
        # for batch_outputs in outputs:
        #     # 确保每个批次的预测结果是列向量的形式 (num_samples, 1)
        #     predictions = batch_outputs['predictions'].cpu().numpy().reshape(-1, 1)
        #     ground_truth = batch_outputs['ground_truth'].cpu().numpy().reshape(-1, 1)
        #     time_coords = batch_outputs['time_coords'].cpu().numpy().reshape(-1, 1)
        #
        #     all_predictions.append(predictions)
        #     all_ground_truths.append(ground_truth)
        #     all_time_coords.append(time_coords)
        #
        # # 连接所有批次的预测结果为一个列向量
        # np.save(os.path.join(save_dir, 'all_predictions.npy'), np.vstack(all_predictions))
        # np.save(os.path.join(save_dir, 'all_ground_truths.npy'), np.vstack(all_ground_truths))
        # np.save(os.path.join(save_dir, 'all_time_coords.npy'), np.vstack(all_time_coords))
        all_predictions, all_ground_truths, all_attns, all_bands_weights, all_sat_weights, all_feat_weights, all_attn_out_weights, all_time_coords = [], [], [], [], [], [], [], []
        for batch_outputs in outputs:
            all_predictions.append(batch_outputs['predictions'].cpu().numpy())
            all_ground_truths.append(batch_outputs['ground_truth'].cpu().numpy())
            # all_attns.append(batch_outputs['attn'].cpu().numpy())
            # all_bands_weights.append(batch_outputs['bands_weights'].cpu().numpy())
            # all_sat_weights.append(batch_outputs['satellite_weights'].cpu().numpy())
            # all_feat_weights.append(batch_outputs['feature_weights'].cpu().numpy())
            # all_attn_out_weights.append(batch_outputs['attn_output_weights'].cpu().numpy())
            all_time_coords.append(batch_outputs['time_coords'].cpu().numpy())
        np.save(os.path.join(save_dir, 'all_predictions.npy'), np.concatenate(all_predictions, axis=0))
        np.save(os.path.join(save_dir, 'all_ground_truths.npy'), np.concatenate(all_ground_truths, axis=0))
        # np.save(os.path.join(save_dir, 'all_attns.npy'), np.concatenate(all_attns, axis=0))
        # np.save(os.path.join(save_dir, 'all_bands_weights.npy'), np.concatenate(all_bands_weights, axis=0))
        # np.save(os.path.join(save_dir, 'all_sat_weights.npy'), np.concatenate(all_sat_weights, axis=0))
        # np.save(os.path.join(save_dir, 'all_feat_weights.npy'), np.concatenate(all_feat_weights, axis=0))
        # np.save(os.path.join(save_dir, 'all_attn_out_weights.npy'), np.concatenate(all_attn_out_weights, axis=0))
        np.save(os.path.join(save_dir, 'all_time_coords.npy'), np.concatenate(all_time_coords, axis=0))
    # def test_epoch_end(self, outputs):
    #     save_dir = 'E:/research/my_code/solar_forecasting/EMD/results/'
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     all_predictions, all_ground_truths, all_time_coords = [], [], []
    #
    #     for daily_outputs in outputs:  # 假设 outputs 是按天分组的
    #         daily_predictions = []
    #         daily_ground_truths = []
    #         daily_time_coords = []
    #
    #         for window_output in daily_outputs:
    #
    #             predictions = window_output['predictions'].cpu().numpy()
    #             ground_truth = window_output['ground_truth'].cpu().numpy()
    #             time_coords = window_output['time_coords'].cpu().numpy()
    #
    #             daily_predictions.append(predictions)
    #             daily_ground_truths.append(ground_truth)
    #             daily_time_coords.append(time_coords)
    #
    #         # 每天的预测完成后，只保留非重叠部分
    #         daily_predictions_array = np.concatenate(daily_predictions, axis=0)[24 - 1:]
    #         daily_ground_truths_array = np.concatenate(daily_ground_truths, axis=0)[24 - 1:]
    #         daily_time_coords_array = np.concatenate(daily_time_coords, axis=0)[24 - 1:]
    #
    #         all_predictions.append(daily_predictions_array)
    #         all_ground_truths.append(daily_ground_truths_array)
    #         all_time_coords.append(daily_time_coords_array)
    #
    #     # 最后将所有天的数据合并
    #     all_predictions_array = np.concatenate(all_predictions)
    #     all_ground_truths_array = np.concatenate(all_ground_truths)
    #     all_time_coords_array = np.concatenate(all_time_coords)
    #
    #     # 保存数据
    #     np.save(os.path.join(save_dir, 'all_predictions.npy'), all_predictions_array)
    #     np.save(os.path.join(save_dir, 'all_ground_truths.npy'), all_ground_truths_array)
    #     np.save(os.path.join(save_dir, 'all_time_coords.npy'), all_time_coords_array)