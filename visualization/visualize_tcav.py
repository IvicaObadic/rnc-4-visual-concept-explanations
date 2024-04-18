import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.preprocessing import normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

#layers = ["encoder.encoder.layer1.2", "encoder.encoder.layer2.3", "encoder.encoder.layer4.2", "encoder.encoder.avgpool"]
layers = ["encoder.encoder.avgpool"]

concept_name_mapping = {"vegetation_wo10": "vegetation", "impervious_surface": "impervious",
                   "city_dense": "dense res.", "city_medium": "medium res.",
                   "city_sparse": "sparse res.", "agriculture": "agriculture",
                   "original_water": "water"}


layer_name_mapping = {"encoder.encoder.layer1.2": "layer 1", "encoder.encoder.layer2.3": "layer 2","encoder.encoder.layer4.2":
                      "layer 4", "encoder.encoder.avgpool": "Average Pooling layer"}

CONCEPT_COLOR_MAPPING = {"vegetation":"g",
                         "agriculture": "khaki",
                         "water": "b",
                         "sparse res.": "violet",
                         "medium res.": "mediumvioletred",
                         "dense res.":"coral",
                         'impervious': "gray"}


def get_out_folder(dataset_name, probing):
    objective = "regression"
    if dataset_name == "household_income":
        if probing:
            model_type = "contrastive pretrained model"
            timestamp = "2024-04-18_08.33.23"
        else:
            model_type = "regression model"
            timestamp = "2024-04-16_14.27.25"
    else:
        if probing:
            model_type = "contrastive pretrained model"
            timestamp = "2024-02-29_09.08.17"
        else:
            model_type = "regression model"
            timestamp = "2024-02-24_19.13.35"

    tcav_results_folder = os.path.join('/home/results/ConceptDiscovery/{}'.format(dataset_name), "models", objective,
                                       "encoder_resnet50")
    if probing:
        tcav_results_folder = os.path.join(tcav_results_folder, "probed")

    tcav_results_folder = os.path.join(tcav_results_folder, timestamp, "TCAV_output")
    return tcav_results_folder, model_type

def get_tcav_scores_with_tsne_activations(results_folder):
    tsne_activations = pd.read_csv(os.path.join(results_folder, "{}_tsne_instances_cavs.csv".format(layer)), index_col=0)
    tcav_results = pd.read_csv(os.path.join(results_folder, "tcav_results_{}.csv".format(layer)), index_col=0)

    activations_tcav_scores = pd.merge(tsne_activations, tcav_results, left_index=True, right_index=True)
    activations_tcav_scores.rename(columns={"concept_pair": "concept"}, inplace=True)
    activations_tcav_scores["concept"] = activations_tcav_scores["concept"].apply(lambda x: x.split("-")[0])
    activations_tcav_scores["concept"].replace(concept_name_mapping, inplace=True)

    return activations_tcav_scores

def normalize_TCAV_value(concept_data):
    max_concept_value = concept_data["TCAV_value"].max()
    min_concept_value = concept_data["TCAV_value"].min()
    concept_data["TCAV_value"] = 2 * ((concept_data.TCAV_value - min_concept_value) / (max_concept_value - min_concept_value)) - 1
    return concept_data

def visualize_tcav_scores_with_tsne_activations():

    tcav_scores_income_probed = get_tcav_scores_with_tsne_activations(out_folder_income_probed)
    tcav_scores_liveability_probed = get_tcav_scores_with_tsne_activations(out_folder_liveability_probed)

    visualization_out_dir = os.path.join(out_folder_income_probed, "visualization")
    os.makedirs(visualization_out_dir, exist_ok=True)

    for concept in tcav_scores_liveability_probed["concept"].unique():

        liveability_concept_data = tcav_scores_liveability_probed.loc[tcav_scores_liveability_probed["concept"] == concept]
        liveability_concept_data = normalize_TCAV_value(liveability_concept_data)

        income_concept_data = tcav_scores_income_probed.loc[tcav_scores_income_probed["concept"] == concept]
        income_concept_data = normalize_TCAV_value(income_concept_data)

        fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(6, 2.0))
        divider = make_axes_locatable(axs[1])  # Create a divider for colorbar placement
        cax = divider.append_axes('right', size='3%', pad=0.05)  # Add axes for colorbar

        plt.subplots_adjust(right=0.8)  # Adjust margins for spacing

        ax_income = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='TCAV_value', palette="plasma",
                                    data=income_concept_data, s=5, ax=axs[0])
        norm = plt.Normalize(income_concept_data['TCAV_value'].min(), income_concept_data['TCAV_value'].max())
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        ax_income.xaxis.set_major_locator(ticker.NullLocator())
        ax_income.yaxis.set_major_locator(ticker.NullLocator())
        ax_income.set(xlabel=None, ylabel=None)
        ax_income.get_legend().remove()
        ax_income.set_title("Income")

        ax_liveability = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='TCAV_value', palette="plasma",
                                         data=liveability_concept_data, s=5, ax=axs[1])
        norm = plt.Normalize(liveability_concept_data['TCAV_value'].min(), liveability_concept_data['TCAV_value'].max())
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        ax_liveability.xaxis.set_major_locator(ticker.NullLocator())
        ax_liveability.yaxis.set_major_locator(ticker.NullLocator())
        ax_liveability.set(xlabel=None, ylabel=None)
        ax_liveability.get_legend().remove()
        ax_liveability.set_title("Liveability")
        label = "Conceptual sensitivity"
        colorbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        colorbar.set_label(label)
        colorbar.ax.set_ylabel(label, rotation=270, labelpad=15, fontsize=10)

        fig_title = fig.text(0.45, 1.0, '{}'.format(concept.capitalize()), ha='center', va='top', fontsize=12)
        fig.tight_layout()
        # plt.tight_layout(rect=(0, 0.02, 1, 0.97))
        plt.savefig(os.path.join(visualization_out_dir, "tcav_{}.png".format(concept)), dpi=300, bbox_inches='tight')



def get_cavs_activations(out_folder, layer, encoder="L1 loss"):
    cav_activations_file_layer = os.path.join(out_folder, "{}_cavs_activations.csv".format(layer))
    cav_layer_accuracy = pd.read_csv(cav_activations_file_layer, usecols=["concept_name", "accuracy"])
    cav_layer_accuracy["concept"] = cav_layer_accuracy["concept_name"].apply(lambda x: x.split("-")[0])
    cav_layer_accuracy["layer"] = layer
    cav_layer_accuracy["encoder"] = encoder

    cav_layer_accuracy["concept"].replace(concept_name_mapping, inplace=True)
    cav_layer_accuracy["layer"].replace(layer_name_mapping, inplace=True)

    return cav_layer_accuracy

def get_concept_layer_accuracy(baseline_model_folder, finetuned_model_folder):
    visualization_out_dir_probed = os.path.join(finetuned_model_folder, "")
    os.makedirs(visualization_out_dir_probed, exist_ok=True)

    concept_layer_accuracies_baseline = pd.read_csv(os.path.join(baseline_model_folder, "cav_accuracy.csv"))
    concept_layer_accuracies_baseline["objective"] = "L1 loss"

    concept_layer_accuracies_probed = pd.read_csv(os.path.join(finetuned_model_folder, "cav_accuracy.csv"))
    concept_layer_accuracies_probed["objective"] = "Rank-N-Contrast"

    concept_accuracy = pd.concat([concept_layer_accuracies_baseline, concept_layer_accuracies_probed], axis=0)
    concept_accuracy["concept"] = concept_accuracy["concept"].apply(lambda x: x.split("-")[0])
    concept_accuracy["concept"].replace(concept_name_mapping, inplace=True)

    fig, axs = plt.subplots(figsize=(4.5, 4.5))
    axs.set_title("{}".format(dataset_title))
    axs.set_ylim(bottom=0, top=1.01)
    sns.barplot(data=concept_accuracy, x="concept", y="accuracy", hue="objective", ax=axs)
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right')
    axs.set_ylabel("SVM concept accuracy")
    fig.tight_layout()
    plt.savefig(os.path.join(visualization_out_dir_probed, "{}_concept_layer_accuracy.pdf".format(dataset_name)),
                dpi=300)
    plt.close()


if __name__ == '__main__':

    dataset_name = "household_income"
    dataset_title = "Liveability" if dataset_name == "Liveability" else "Income"
    out_folder_income_probed, _ = get_out_folder("household_income", True)

    out_folder_liveability_probed, _ = get_out_folder("Liveability", True)

    for layer in layers:
        visualize_tcav_scores_with_tsne_activations()










