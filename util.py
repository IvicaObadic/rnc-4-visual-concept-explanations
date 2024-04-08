import os

import torch
from torch.utils.data import DataLoader
from liveability.dataloader import TCAV_dataset
from liveability.pt_funcs import models as liveability_models
from liveability.pt_funcs.dataloaders import LBMLoader
from models.socioeconomic_inference import *
from models.income_prediction_model import *
from models.loss_functions import *
from household_income.householdincomedataset import HouseholdIncomeDataset

from utils.augmentation import *

# dataset_name_root_dir = {
#     'income': "/home/ConceptDiscovery/SESEfficientCAM-master/",
#     "liveability": "/home/datasets/Liveability/"
# }

def create_data_loader(dataset, batch_size, split):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=split != "val", num_workers=8,
                      drop_last=True, pin_memory=True)

def get_data_loaders(dataset_name, training_objective, batch_size):
    if dataset_name == "household_income":
        root_dir = "/home/ConceptDiscovery/SESEfficientCAM-master/"
        root_data_dir = os.path.join(root_dir, "data")
        #labels_file = os.path.join(root_data_dir, "census_data", "squares_to_ses_2019.csv")
        images_root_dir = os.path.join(root_data_dir, "imagery_out")
        train_file = os.path.join(images_root_dir, "train.csv")
        val_file = os.path.join(images_root_dir, "val.csv")
        test_file = os.path.join(images_root_dir, "test.csv")

        dataset_train = HouseholdIncomeDataset(train_file,
                                               transform=get_transforms(training_objective, train_transforms))
        dataset_val = HouseholdIncomeDataset(val_file, transform=get_transforms(training_objective, test_transforms))
        dataset_test = HouseholdIncomeDataset(test_file, transform=get_transforms(training_objective, test_transforms))

        return (create_data_loader(dataset_train, batch_size, "train"),
                create_data_loader(dataset_val, batch_size, "val"),
                create_data_loader(dataset_test, batch_size, "val"))
    else:
        dataset_root_dir = "/home/datasets/{}/".format(dataset_name)
        print(dataset_root_dir)
        dataset_info_file = os.path.join(dataset_root_dir, "grid_geosplit_not_rescaled.geojson")
        lbm_data_module = LBMLoader(n_workers=8, batch_size=batch_size)
        lbm_data_module.setup_data_classes(dataset_info_file,
                                           dataset_root_dir,
                                           splits=['train', 'val', "test"],
                                           train_transforms=get_transforms(training_objective, train_transforms),
                                           val_transforms=get_transforms(training_objective, test_transforms),
                                           test_transforms=get_transforms(training_objective, test_transforms))

        return lbm_data_module.train_dataloader(), lbm_data_module.val_dataloader(), lbm_data_module.test_dataloader()

def get_liveability_model():
    # Set up model
    label_info = {}
    label_info['dim_scores'] = {}
    label_info['lbm_score'] = {}
    label_info['lbm_score']['ylims'] = [-1.5, 1.5]
    checkpoint_path = 'codebase_TCAV/epoch=14-val_lbm_mse=0.03.ckpt'
    model = liveability_models.LBMBaselineModel('outputs_tcav/',
                             'baseline', label_info, splits=['val', 'test', 'train'])
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))[
        'state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    if torch.cuda.is_available():
        model.model.model.cuda()

    return model.model.model

def get_trained_model(model_root_dir, model_checkpoint_path):
    objective = "regression"
    loss_fn = get_loss_function(objective)
    socioeconomicinference_model = init_model(objective, "resnet50", False, None)
    model = SocioeconomicRegressionInference(model_root_dir, socioeconomicinference_model, objective, loss_fn)
    model_state_dict = torch.load(os.path.join(model_root_dir, model_checkpoint_path), map_location=torch.device('cpu'))["state_dict"]
    model.load_state_dict(model_state_dict)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    return model.model


def get_liveability_dataloader():
    data_path = 'data/source/amsterdam_data_dict.pkl'
    test_dataset = TCAV_dataset(data_path, transform=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4)
    return test_dataloader


def get_household_income_data_loader():
    data_path = "/home/ConceptDiscovery/SESEfficientCAM-master/data/imagery_out/test.csv"
    test_dataset = HouseholdIncomeDataset(data_path, transform=test_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4)
    return test_dataloader




