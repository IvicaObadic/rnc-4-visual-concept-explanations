import os.path

from torch.utils.data import DataLoader
from datasets.liveability.pt_funcs.dataloaders import LBMLoader
from models.socioeconomic_inference import *
from models.socieconomic_outcome_model import *
from models.loss_functions import *
from datasets.household_income.householdincomedataset import HouseholdIncomeDataset

from utils.augmentation import *

income_dataset_root_dir = "/home/ConceptDiscovery/SESEfficientCAM-master/"
liveability_dataset_root_dir = "/home/datasets/Liveability/"

def create_data_loader(dataset, batch_size, split):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=split != "val", num_workers=8,
                      drop_last=True, pin_memory=True)

def get_data_loaders(dataset_name, dataset_root_dir, training_objective, batch_size):
    if dataset_name == "household_income":
        root_data_dir = os.path.join(dataset_root_dir, "data")
        images_root_dir = os.path.join(root_data_dir, "imagery_out")
        train_file = os.path.join(images_root_dir, "train.csv")
        val_file = os.path.join(images_root_dir, "val.csv")
        test_file = os.path.join(images_root_dir, "test.csv")

        dataset_train = HouseholdIncomeDataset(train_file, transform=get_transforms(training_objective, train_transforms))
        dataset_val = HouseholdIncomeDataset(val_file, transform=get_transforms(training_objective, test_transforms))
        dataset_test = HouseholdIncomeDataset(test_file, transform=get_transforms(training_objective, test_transforms))

        return (create_data_loader(dataset_train, batch_size, "train"),
                create_data_loader(dataset_val, batch_size, "val"),
                create_data_loader(dataset_test, batch_size, "val"))
    else:
        dataset_info_file = os.path.join(dataset_root_dir, "grid_geosplit_not_rescaled.geojson")
        lbm_data_module = LBMLoader(n_workers=8, batch_size=batch_size)
        lbm_data_module.setup_data_classes(dataset_info_file,
                                           dataset_root_dir,
                                           splits=['train', 'val', "test"],
                                           train_transforms=get_transforms(training_objective, train_transforms),
                                           val_transforms=get_transforms(training_objective, test_transforms),
                                           test_transforms=get_transforms(training_objective, test_transforms))

        return lbm_data_module.train_dataloader(), lbm_data_module.val_dataloader(), lbm_data_module.test_dataloader()

def get_trained_model(model_root_dir, model_checkpoint_path):
    objective = "regression"
    loss_fn = get_loss_function(objective)
    socioeconomicinference_model = init_model(objective, "resnet50", False, None)
    print("Loading model checkpoint from {}".format(os.path.join(model_root_dir, model_checkpoint_path)), flush=True)
    model = SocioeconomicRegressionInference(model_root_dir, socioeconomicinference_model, objective, loss_fn)
    model_state_dict = torch.load(os.path.join(model_root_dir, model_checkpoint_path), map_location=torch.device('cpu'))["state_dict"]
    model.load_state_dict(model_state_dict)
    model = model.model
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    return model




