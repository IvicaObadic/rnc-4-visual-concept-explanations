import os
import torch
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler, ReduceLROnPlateau
from torchmetrics.classification import Accuracy
from torchmetrics import MeanAbsoluteError

import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import kendalltau, pearsonr

class SocioeconomicClassificationInference(pl.LightningModule):
    def __init__(self, model,  loss_fn=torch.nn.CrossEntropyLoss(), batch_size=32, initial_lr=8e-5) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.mae_metric = MeanAbsoluteError()
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=5)

    def predict(self, data, mode="train"):
        ids = data[0]
        images = data[1]
        labels = data[2]
        if mode != "train":
            labels = None

        predictions = self.model(images, labels)

        return predictions

    def training_step(self, data, batch_idx) -> STEP_OUTPUT:
        train_preds = self.predict(data)
        labels = data[2]
        loss = self.loss_fn(train_preds, labels)

        self.log("train_loss", loss, batch_size=self.batch_size, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, data, batch_idx) -> Optional[STEP_OUTPUT]:
        val_preds = self.predict(data, mode="val")
        scores, val_preds_classes = torch.max(val_preds, dim=1)
        labels = data[2]
        loss = self.loss_fn(val_preds, labels)
        val_accuracy = self.accuracy_metric(val_preds_classes, labels)
        val_mae = self.mae_metric(val_preds_classes, labels)

        self.log("val_loss", loss, batch_size=self.batch_size, on_epoch=True, on_step=False)
        self.log("val_accuracy", val_accuracy, batch_size=self.batch_size, on_epoch=True, on_step=False)
        self.log("val_mae", val_mae, batch_size=self.batch_size, on_epoch=True, on_step=False)

        return loss

    def test_step(self, data, batch_idx) -> Optional[STEP_OUTPUT]:
        test_preds = self.predict(data, mode="test")
        scores, test_preds_classes = torch.max(test_preds, dim=1)
        labels = data[2]
        loss = self.loss_fn(test_preds, labels)
        test_accuracy = self.accuracy_metric(test_preds_classes, labels)
        test_mae = self.mae_metric(test_preds_classes, labels)

        self.log("test_loss", loss, batch_size=self.batch_size, on_epoch=True, on_step=False)
        self.log("test_accuracy", test_accuracy, batch_size=self.batch_size, on_epoch=True, on_step=False)
        self.log("test_mae", test_mae, batch_size=self.batch_size, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=0.0005)
        #scheduler = ReduceLROnPlateau(adam_optimizer, mode="min", factor=0.25, patience=3, verbose=True, min_lr=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, T_max=10, eta_min=1e-9)
        return {
            "optimizer": adam_optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            },
        }


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
        #adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, T_max=self.trainer.max_epochs, eta_min=1e-8)

        #Liveability params
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        if self.training_objective == "regression":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        #scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=3, verbose=True, min_lr=1e-8)

        return [optimizer], [scheduler]