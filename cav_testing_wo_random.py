import os
import pickle
import shutil

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity

from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV

from util import *
from codebase_TCAV.utils_CAVs import assemble_concept, assemble_random_concept
from codebase_TCAV.Custom_Classifier import EfficientClassifier, StratifiedClassifier
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist


import datetime
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# determine device
if torch.cuda.is_available():
    device = 'cuda'
    print('Device: cuda available')
else:
    device = 'cpu'
    print('Device: cuda not available, using cpu')

# Define the hidden layers on which the TCAV algorithm is run
layers = ["encoder.encoder.avgpool"]

def compute_cavs_and_activations(dataset_name, probing, timestamp, model_checkpointpath):
    def save_instance_activations(data_loader, target_layer, layer_cavs, store_tsne=True):
        activations = []
        cosine_sim_cavs_instances = []
        layer_cavs_tensor = torch.from_numpy(layer_cavs.values).float().to(device)

        def getActivation(layer_name):
            # the hook signature
            def hook(model, input, output):
                activations_flattened = torch.flatten(output)
                if store_tsne:
                    activations.append(activations_flattened.cpu().detach().numpy())
                activations_flattened_matr = activations_flattened.unsqueeze(dim=0)
                cosine_siimlarity_fn = CosineSimilarity()
                distance_activations_cavs = cosine_siimlarity_fn(activations_flattened_matr, layer_cavs_tensor).squeeze()
                cosine_sim_cavs_instances.append(distance_activations_cavs.cpu().detach().numpy().flatten())

            return hook

        #define the hook on the target layers
        hook_handles = []
        for layer_name, layer in model.named_modules():
            if layer_name == target_layer:
                print("Defining a hook for layer {}".format(layer_name))
                hook_handle = layer.register_forward_hook(getActivation(layer_name))
                hook_handles.append(hook_handle)

        test_ids = []
        all_preds = []
        # run the forward pass to store the activations
        print("Running the forward pass to store the activations")
        for idx, batch in enumerate(data_loader):
            image_id = batch[0][0]
            image = batch[1].to(device)
            label = batch[2]

            output = model(image, label)
            test_ids.append(image_id)
            label = label.squeeze().cpu().detach().numpy()
            prediction = output.squeeze().cpu().detach().numpy()
            all_preds.append((image_id, label, prediction))

        all_preds_df = pd.DataFrame(all_preds, columns=["image_id", "label", "prediction"])
        all_preds_df.to_csv(os.path.join(out_folder, "predictions.csv"))

        cosine_sim_cavs_instances_df = pd.DataFrame(cosine_sim_cavs_instances, index=test_ids, columns=layer_cavs.index)
        cosine_sim_cavs_instances_df.to_csv(os.path.join(out_folder, "{}_cosine_sim_instances_cavs.csv".format(target_layer)))

        # remove hook handles
        for hook_handle in hook_handles:
            hook_handle.remove()

        #compute tsne of the activations and the cavs and save only them in memory
        if store_tsne:
            layer_instance_activations_df = pd.DataFrame(np.array(activations), index=test_ids)

            print("Computing tSNE for the activations and CAVs of layer {}".format(layer))
            print(layer_instance_activations_df.shape)
            tsne = TSNE(n_components=2)
            tsne_instance_activations = tsne.fit_transform(layer_instance_activations_df)
            tsne_instance_activations = pd.DataFrame(tsne_instance_activations,columns=["tsne_dim_1", "tsne_dim_2"], index=test_ids)

            print("Saving instance activations for layer {}".format(layer))
            tsne_instance_activations.to_csv(os.path.join(out_folder, "{}_tsne_instances_cavs.csv".format(target_layer)))


    objective = "regression"
    model_dir = os.path.join('/home/results/ConceptDiscovery/{}'.format(dataset_name), "models", objective,
                             "encoder_resnet50")
    if probing:
        model_dir = os.path.join(model_dir, "fine-tuned")
    model_dir = os.path.join(model_dir, timestamp)

    model = get_trained_model(model_dir, model_checkpointpath)

    #setup the TCAV output directory
    out_folder = os.path.join(model_dir, "TCAV_output_wo_random/")
    os.makedirs(out_folder, exist_ok=True)

    # Setting up Concepts
    concept_path = "/home/ConceptDiscovery/concepts/TCAV_data/"
    concepts_bank = ['impervious_surface', 'vegetation_wo10', 'city_dense', 'city_medium', 'city_sparse', 'agriculture', 'original_water']

    experiment_setup = []
    for concept_idx, concept in enumerate(concepts_bank):
        all_other_concepts = [other_concept for other_concept in concepts_bank if other_concept != concept]
        concept_dataset = assemble_concept(concept, concept_idx, concept_path, device)
        random_dataset = assemble_random_concept(all_other_concepts, concept_idx + len(concepts_bank), concept_path, device)
        experiment_setup.append([concept_dataset, random_dataset])

    # Define Setup
    setup_list = []
    concept_names_list = []
    for concept_pair in experiment_setup:
        setup_list.append(f'{concept_pair[0].id}-{concept_pair[1].id}')
        concept_names_list.append(f'{concept_pair[0].name}-{concept_pair[1].name}')

    # define TCAV Vectors for concepts
    TCAV_0 = TCAV(model=model,
                  layers=layers,
                  layer_attr_method=LayerIntegratedGradients(
                      model, None,
                      multiply_by_inputs=True),
                  classifier=StratifiedClassifier(epochs=1, device=device, batch_size=400))

    # TCAV_0 = TCAV(model=model.model.model,layers=layers,classifier=EfficientClassifier(epochs=1, device=device))
    print("Saving CAVs")
    # compute Classifier and save for accuracies and CAV weights
    cavs_computed = TCAV_0.compute_cavs(experiment_setup, force_train=True)
    cav_weights_per_layer = {}
    cav_accuracy_per_layer = []
    for idx, layer in enumerate(layers):
        cav_weights = {}
        for concept_pair_idx, concept_pair in enumerate(setup_list):
            accs_score = cavs_computed[concept_pair][f"{layer}"].stats['accs']
            f1_score = cavs_computed[concept_pair][f"{layer}"].stats['f1_score']
            concept_pair_name = concept_names_list[concept_pair_idx]
            print(f'Accs score for concept pair {concept_pair_name} in {layer} is {accs_score}')
            cav_accuracy_per_layer.append((layer, concept_pair_name, accs_score, f1_score))
            cav_weights[concept_pair_name] = cavs_computed[concept_pair][f"{layer}"].stats["weights"][1].cpu().detach().numpy()

        cavs_per_layer_df = pd.DataFrame.from_dict(cav_weights, orient="index")
        cavs_per_layer_df.index.name = "concept_name"
        cav_weights_per_layer[layer] = cavs_per_layer_df
        #create a deep copy to store the accuracy of the activations
        cavs_per_layer_df.to_csv(os.path.join(out_folder, "{}_cavs_activations.csv".format(layer)))

    cav_accuracy_per_layer = pd.DataFrame(cav_accuracy_per_layer, columns=["layer", "concept", "accuracy", "f1_score"])
    cav_accuracy_per_layer.to_csv(os.path.join(out_folder, "cav_accuracy.csv"))

    train_data_loader, val_data_loader, test_dataloader = get_data_loaders(dataset_name, objective, batch_size=1)
    for target_layer in layers:
        save_instance_activations(test_dataloader, target_layer, cav_weights_per_layer[target_layer])

    tcav_results = []
    # iterate over dataset and compute TCAV scores
    num_batches = len(test_dataloader)
    for idx, batch in enumerate(test_dataloader):
        print(f'Compute TCAV Scores for batch {idx} out of {num_batches}')
        image_id = batch[0][0]
        image = batch[1].to(device)
        label = batch[2].item()
        tcav_scores = TCAV_0.interpret(
            inputs=image,
            experimental_sets=experiment_setup,
            n_steps=5)
        for i, concept_pair in enumerate(setup_list):
            for layer in layers:
                concept_pair_name = concept_names_list[i]
                tcav_sign_result = tcav_scores[concept_pair][layer]['sign_count'].cpu().detach().numpy()[0]
                tcav_magnitude_result = tcav_scores[concept_pair][layer]['magnitude'].cpu().detach().numpy()[0]
                tcav_results.append((image_id, concept_pair_name, layer, label, tcav_sign_result, tcav_magnitude_result))

    tcav_results_df = pd.DataFrame(tcav_results,
                                   columns=["image_id", "concept_pair", "layer", "label", "concept_sign", "TCAV_value"])
    tcav_results_df.set_index("image_id", inplace=True)

    results_file = os.path.join(out_folder, "tcav_results_{}.csv".format(",".join(layers)))
    print("Saving the results to the file {}".format(results_file))
    tcav_results_df.to_csv(results_file)


if __name__ == '__main__':
    dataset_name = "household_income"
    probing = False
    #Household income checkpoint finetuned
    # timestamp = "2024-02-28_08.55.11"
    # model_checkpointpath = "epoch=97-val_R2_entire_set=0.62.ckpt"

    #Household income checkpoint baseline
    timestamp = "2024-02-26_17.51.13"
    model_checkpointpath = "epoch=93-val_R2_entire_set=0.53.ckpt"

    #Liveability checkpoint finetuned
    # timestamp = "2024-02-29_09.08.17"
    # model_checkpointpath = "epoch=45-val_R2_entire_set=0.68.ckpt"

    # Liveability checkpoint baseline
    # timestamp = "2024-02-24_19.13.35"
    # model_checkpointpath = "epoch=29-val_R2_entire_set=0.70.ckpt"
    compute_cavs_and_activations(dataset_name, probing, timestamp, model_checkpointpath)
