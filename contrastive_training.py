import datetime
import os

from torch.utils.data import DataLoader
import wandb


wandb.login()
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from models.income_prediction_model import *
from models.socioeconomic_inference import *
from models.loss_functions import *
from utils.augmentation import *
from util import *


def socieconomic_model_training(dataset_name,
                                training_objective,
                                lr, num_iterations, weight_decay, batch_size,
                                encoder_name="efficientNet",
                                pretrained_encoder=True,
                                encoder_weights_path=None):

    train_data_loader, val_data_loader, test_data_loader = get_data_loaders(dataset_name, training_objective, batch_size)

    model = init_model(objective=training_objective,
                       encoder_name=encoder_name,
                       pretrained=pretrained_encoder,
                       encoder_weights_path=encoder_weights_path)

    model_output_dir = os.path.join("/home/results/ConceptDiscovery", dataset_name, "models", training_objective, model.model_name())
    if encoder_weights_path is not None:
        model_output_dir = os.path.join(model_output_dir, "fine-tuned")

    run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    model_output_dir = os.path.join(model_output_dir, run_time)

    metric_to_monitor = "val_R2_entire_set"
    checkpoint_filename = "{epoch:02d}-{val_R2_entire_set:.2f}"
    limit_val_batches = 1.0
    mode = "max"
    if training_objective == "contrastive":
        metric_to_monitor = "train_loss"
        checkpoint_filename = "{epoch:02d}-{train_loss:.2f}"
        limit_val_batches = 0.0
        mode = "min"

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=metric_to_monitor,
        mode=mode,
        dirpath=model_output_dir,
        filename=checkpoint_filename)

    socioeconomic_inference = SocioeconomicRegressionInference(model_output_dir,
                                                               model=model,
                                                               training_objective=training_objective,
                                                               loss_fn=get_loss_function(training_objective),
                                                               batch_size=batch_size,
                                                               initial_lr=lr,
                                                               weight_decay=weight_decay)

    wandb_logger = WandbLogger(project='{}_inference_{}_training'.format(dataset_name, training_objective))
    wandb_logger.experiment.config["learning_rate"] = lr
    wandb_logger.experiment.config["pretrained_encoder"] = pretrained_encoder
    wandb_logger.experiment.config["training_objective"] = training_objective

    trainer = pl.Trainer(max_epochs=num_iterations,
                         check_val_every_n_epoch=1,
                         callbacks=[checkpoint_callback],
                         default_root_dir=model_output_dir,
                         log_every_n_steps=10,
                         logger=wandb_logger,
                         limit_val_batches=limit_val_batches)

    trainer.fit(model=socioeconomic_inference, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

    if training_objective == "regression":
        trainer.test(dataloaders=test_data_loader, ckpt_path="best")

if __name__ == '__main__':
    dataset_name = "Liveability"
    #dataset_name = "household_income"
    pretrained_encoder = False
    #setup for contrastive learning
    # socieconomic_model_training(dataset_name,
    #                             "contrastive",
    #                             5e-4,
    #                             num_iterations=400,
    #                             weight_decay=1e-4,
    #                             batch_size=32,
    #                             encoder_name="resnet50",
    #                             pretrained_encoder=pretrained_encoder,
    #                             encoder_weights_path=None)

    #setup for regression training from scratch
    # socieconomic_model_training(dataset_name,
    #                             "contrastive",
    #                             5e-4,
    #                             num_iterations=400,
    #                             weight_decay=1e-4,
    #                             batch_size=32,
    #                             encoder_name="resnet50",
    #                             pretrained_encoder=pretrained_encoder,
    #                             encoder_weights_path=None)

    #setup for regression probing
    #liveability pretrained encoder
    # pretrained_contrastive_encoder_root_dir = "/home/results/ConceptDiscovery/{}/models/contrastive/encoder_resnet50/2024-02-24_18.58.57/".format(dataset_name)
    # pretrained_contrastive_encoder_path = os.path.join(pretrained_contrastive_encoder_root_dir, "epoch=100-train_loss=2.26.ckpt")

    pretrained_contrastive_encoder_root_dir = "/home/results/ConceptDiscovery/{}/models/contrastive/encoder_resnet50/2024-02-24_18.58.57/".format(dataset_name)
    pretrained_contrastive_encoder_path = os.path.join(pretrained_contrastive_encoder_root_dir, "epoch=373-train_loss=1.74.ckpt")
    socieconomic_model_training(dataset_name,
                                "regression",
                                5e-3,
                                num_iterations=100,
                                weight_decay=0,
                                batch_size=32,
                                encoder_name="resnet50",
                                pretrained_encoder=pretrained_encoder,
                                encoder_weights_path=pretrained_contrastive_encoder_path)

