import os
import torch
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import kendalltau

class SocioeconomicRegressionInference(pl.LightningModule):
    def __init__(self,
                 model_output_dir,
                 model,
                 training_objective,
                 loss_fn,
                 batch_size=32,
                 initial_lr=8e-5,
                 weight_decay=1e-4) -> None:
        super().__init__()
        self.model_output_dir = model_output_dir
        self.model = model
        self.training_objective = training_objective
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay

        self.validation_preds = []
        self.validation_labels = []
        self.validation_ids = []

    def predict(self, data):
        images = data[1]

        if self.training_objective == "regression":
            model_output = self.model(images)

        else:
            # contrastive mode, the model is only consisting of the encoder and the input is with example augmentation
            images = torch.cat([images[0], images[1]], dim=0)
            features = self.model(images)
            print("Batch shape after running the encoder {}".format(features.shape))
            f1, f2 = torch.split(features, [self.batch_size, self.batch_size], dim=0)
            model_output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        return model_output

    def training_step(self, data, batch_idx) -> STEP_OUTPUT:
        train_preds = self.predict(data)
        labels = data[2].unsqueeze(1)
        loss = self.loss_fn(train_preds, labels)

        self.log("train_loss", loss, batch_size=self.batch_size, on_epoch=True, on_step=False)

        return loss

    def calculate_regression_metrics(self, split="val"):
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)

        prediction_data = {"ids": self.validation_ids, "labels": self.validation_labels, "preds": self.validation_preds}
        prediction_data = pd.DataFrame(prediction_data)
        prediction_data.to_csv(os.path.join(self.model_output_dir, "predictions_{}_{}.csv".format(split, self.current_epoch)))

        mse_val = mean_squared_error(self.validation_labels, self.validation_preds)
        r2_val = r2_score(self.validation_labels, self.validation_preds)
        tau_val = kendalltau(self.validation_preds, self.validation_labels).statistic
        #pearsonr_val = pearsonr(self.validation_preds, self.validation_labels).statistic
        self.log("{}_R2_entire_set".format(split), r2_val, on_epoch=True, on_step=False)
        self.log("{}_MSE_entire_set".format(split), mse_val, on_epoch=True, on_step=False)
        self.log("{}_tau_entire_set".format(split), tau_val, on_epoch=True, on_step=False)
        #self.log("{}_mae_entire_set".format(split), mae_val, on_epoch=True, on_step=False)
        #self.log("{}_pearsonr_entire_set".format(split), pearsonr_val, on_epoch=True, on_step=False)

        metrics_scores = {
            "mse": [mse_val],
            "r2": [r2_val],
            "tau": [tau_val]}
            #"pearsonr": [pearsonr_val],
            #"mae": [mae_val]

        metrics_scores_df = pd.DataFrame.from_dict(metrics_scores)
        metrics_scores_df.to_csv(os.path.join(self.model_output_dir, "metrics_{}_{}.csv".format(split,
                                                                                                self.current_epoch)))
        self.validation_ids.clear()
        self.validation_preds.clear()
        self.validation_labels.clear()


    def validation_step(self, data, batch_idx) -> Optional[STEP_OUTPUT]:
        val_preds = self.predict(data)
        labels = data[2].unsqueeze(1)
        loss = self.loss_fn(val_preds, labels)

        self.log("val_loss", loss, batch_size=self.batch_size, on_epoch=True, on_step=False)

        if self.training_objective == "regression":
            val_preds_np = val_preds.squeeze().detach().cpu().numpy()
            labels_np = labels.squeeze().detach().cpu().numpy()
            ids = data[0]
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().numpy()
            self.validation_ids.extend(ids)
            self.validation_preds.extend(val_preds_np.tolist())
            self.validation_labels.extend(labels_np.tolist())

        return loss

    def on_validation_epoch_end(self) -> None:
        if self.training_objective == "regression":
            self.calculate_regression_metrics("val")

    def on_test_epoch_end(self) -> None:
        if self.training_objective == "regression":
            self.calculate_regression_metrics("test")

    def test_step(self, data, batch_idx) -> Optional[STEP_OUTPUT]:
        test_preds = self.predict(data)
        labels = data[2].unsqueeze(1)
        loss = self.loss_fn(test_preds, labels)

        self.log("test_loss", loss, batch_size=self.batch_size, on_epoch=True, on_step=False)

        if self.training_objective == "regression":
            test_preds_np = test_preds.squeeze().detach().cpu().numpy()
            labels_np = labels.squeeze().detach().cpu().numpy()
            ids = data[0]
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().numpy()
            self.validation_ids.extend(ids)
            self.validation_preds.extend(test_preds_np.tolist())
            self.validation_labels.extend(labels_np.tolist())

        return loss


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        if self.training_objective == "regression":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        return [optimizer], [scheduler]