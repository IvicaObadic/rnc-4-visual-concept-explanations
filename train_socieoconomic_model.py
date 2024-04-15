import argparse
import datetime

import wandb
wandb.login()
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import yaml

from models.socieconomic_outcome_model import *
from models.socioeconomic_inference import *
from models.loss_functions import *
from utils.augmentation import *
from util import *


def parse_args():
  parser = argparse.ArgumentParser(description="Parses the parameters for the contrastive pretraining and probing of the deep learning models that predict socieconomic outcomes.")

  parser.add_argument("--dataset_name", type=str, default="household_income", help="Name of the dataset to use for training. Supported names are 'household_income' and 'Liveability'.")
  parser.add_argument("--dataset_root_dir", type=str, default=income_dataset_root_dir, help="Root directory containing the dataset.")
  parser.add_argument("--training_objective", type=str, default="contrastive",help="Training objective. Supported options are 'contrastive' and 'regression'.")
  parser.add_argument("--model_output_root_dir", type=str, default="/home/results/ConceptDiscovery", help="The root dir where the model checkpoints and results are stored.'")
  parser.add_argument("--encoder_weights_path", type=str, default=None, help="Path to pre-trained encoder weights. ")

  return parser.parse_args()


def socieconomic_model_training(dataset_name,
                                dataset_root_dir,
                                training_objective,
                                model_output_root_dir,
                                encoder_weights_path=None):

    setup_file = "experiments_setup/training_params.yaml"
    with open(setup_file) as file:
        training_params = yaml.full_load(file)
    batch_size = training_params[training_objective]["batch_size"]

    train_data_loader, val_data_loader, test_data_loader = get_data_loaders(dataset_name, dataset_root_dir, training_objective, batch_size)

    model = init_model(objective=training_objective,
                       encoder_name=training_params[training_objective]["encoder_name"],
                       pretrained=training_params[training_objective]["pretrained_encoder"],
                       encoder_weights_path=encoder_weights_path)
    model_output_dir = os.path.join(model_output_root_dir, dataset_name, "models", training_objective, model.model_name())
    if encoder_weights_path is not None:
        model_output_dir = os.path.join(model_output_dir, "probing")

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

    lr = training_params[training_objective]["lr"]
    print(lr)
    weight_decay = training_params[training_objective]["weight_decay"]
    socioeconomic_inference = SocioeconomicRegressionInference(model_output_dir,
                                                               model=model,
                                                               training_objective=training_objective,
                                                               loss_fn=get_loss_function(training_objective),
                                                               batch_size=batch_size,
                                                               initial_lr=lr,
                                                               weight_decay=weight_decay)

    wandb_logger = WandbLogger(project='{}_inference_{}_training'.format(dataset_name, training_objective))
    wandb_logger.experiment.config["learning_rate"] = lr
    wandb_logger.experiment.config["pretrained_encoder"] = training_params[training_objective]["pretrained_encoder"]
    wandb_logger.experiment.config["training_objective"] = training_objective

    trainer = pl.Trainer(max_epochs=training_params[training_objective]["num_iterations"],
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
    args = parse_args()
    socieconomic_model_training(args.dataset_name,
                                args.dataset_root_dir,
                                args.training_objective,
                                args.model_output_root_dir,
                                args.encoder_weights_path)
