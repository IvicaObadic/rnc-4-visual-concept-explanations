import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.preprocessing import normalize
from visualize_tcav import get_out_folder, concept_name_mapping, CONCEPT_COLOR_MAPPING, sorted_concepts

layer = "encoder.encoder.avgpool"

def get_normalized_cosine_sim_per_concept(outfolder, target_layer):

    cosine_sim_to_cav = pd.read_csv(os.path.join(outfolder, "{}_cosine_sim_instances_cavs.csv".format(target_layer)), index_col=0)
    cosine_sim_to_cav_np = cosine_sim_to_cav.to_numpy()
    cosine_sim_to_cav_np = normalize(cosine_sim_to_cav_np, axis=0)
    cosine_sim_to_cav = pd.DataFrame(cosine_sim_to_cav_np, columns=cosine_sim_to_cav.columns, index=cosine_sim_to_cav.index).reset_index()

    predictions = pd.read_csv(os.path.join(outfolder, "predictions.csv"), index_col=0)
    cosine_sim_to_cav_with_preds = pd.merge(cosine_sim_to_cav, predictions, left_on="index", right_on="image_id").drop(columns=["image_id", "index"])
    corr_cav_label = cosine_sim_to_cav_with_preds.corr()["label"]
    corr_cav_label.to_csv(os.path.join(outfolder, "", "label_cav_corr_{}.csv").format(target_layer))

    cosine_sim_to_cav = pd.melt(cosine_sim_to_cav, id_vars=["index"], var_name="concept", value_name="cosine similarity to CAV")
    predictions_cosine_sim = pd.merge(cosine_sim_to_cav, predictions, left_on="index", right_on="image_id")
    predictions_cosine_sim["concept"] = predictions_cosine_sim["concept"].apply(lambda x: x.split("-")[0])
    predictions_cosine_sim["concept"].replace(concept_name_mapping, inplace=True)

    predictions_cosine_sim = predictions_cosine_sim.sort_values("cosine similarity to CAV", ascending=False).drop_duplicates(['index'])
    tsne_activations = pd.read_csv(os.path.join(outfolder, "{}_tsne_instances_cavs.csv".format(layer)), index_col=0)
    predictions_cosine_sim = pd.merge(predictions_cosine_sim, tsne_activations, left_on="index", right_index=True)

    return predictions_cosine_sim

def cluster_concepts_for_single_model(concept_similarities):

    figsize = (1.5, 1.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='concept', palette=CONCEPT_COLOR_MAPPING,
                         data=concept_similarities, ax=ax, s=5, edgecolor=None)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.get_legend().remove()
    ax.set(xlabel=None, ylabel=None)
    ax.set_title("Concepts")
    fig.tight_layout()
    plt.savefig(os.path.join(out_folder_income_contrastive, "small_income_clusters.png"), dpi=300)
    plt.close()

def cluster_concepts_for_contrastive_encoder(income_conpcet_similarities, liveability_concept_similarities):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 2.5))
    fig.subplots_adjust(right=0.8)  # Adjust right side for legend space

    ax_hi_contrastive = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='concept', palette=CONCEPT_COLOR_MAPPING,
                                        data=income_conpcet_similarities, ax=axs[0], s=5, edgecolor=None)
    ax_hi_contrastive.xaxis.set_major_locator(ticker.NullLocator())
    ax_hi_contrastive.yaxis.set_major_locator(ticker.NullLocator())
    ax_hi_contrastive.set(xlabel=None, ylabel=None)
    ax_hi_contrastive.set_title("Income")
    ax_hi_contrastive.set_ylabel("Concept clusters")

    ax_liveability_contrastive = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='concept',
                                                 palette=CONCEPT_COLOR_MAPPING,
                                                 data=liveability_concept_similarities, ax=axs[1], s=5,
                                                 edgecolor=None)
    ax_liveability_contrastive.xaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_contrastive.yaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_contrastive.set(xlabel=None, ylabel=None)
    ax_liveability_contrastive.set_title("Liveability")

    lines, labels = [], []
    for ax in fig.axes:
        legend_lines, legend_labels = ax.get_legend_handles_labels()
        lines.extend(legend_lines)
        labels.extend(legend_labels)
        ax.get_legend().remove()

    # Combine lines and labels from all subplots, removing duplicates
    unique_labels, lines_for_plot = [], []
    for line, label in zip(lines, labels):
        if label not in unique_labels:
            print(label, line.get_color())
            unique_labels.append(label)
            lines_for_plot.append(
                Rectangle((0, 0), 0.5, 0.5, color=line.get_color(), label=label))  # Create square legend symbols


    sorted_legend_items = [label for label in sorted_concepts if label in unique_labels]
    sorted_lines_for_plot = [lines_for_plot[unique_labels.index(label)] for label in sorted_legend_items]

    # Create the legend entirely outside the plot with one column
    legend = fig.legend(sorted_lines_for_plot, sorted_legend_items, loc='upper left', title="Concept",
                        bbox_to_anchor=(0.98, 0.88))  # Adjust location as needed

    # Adjust layout (optional)
    fig.tight_layout()
    plt.savefig(os.path.join(out_folder_income_contrastive, "", "contrastive_instance_cluster_allignment.png"), bbox_inches='tight',
                dpi=300)
    plt.close()

if __name__ == '__main__':

    out_folder_income_baseline, model_type = get_out_folder("household_income", False)
    out_folder_income_contrastive, model_type = get_out_folder("household_income", True)

    out_folder_liveability_baseline, model_type = get_out_folder("Liveability", False)
    out_folder_liveability_contrastive, model_type = get_out_folder("Liveability", True)

    hi_contrastive_results = get_normalized_cosine_sim_per_concept(out_folder_income_contrastive, layer)
    hi_baseline_results = get_normalized_cosine_sim_per_concept(out_folder_income_baseline, layer)

    liveability_contrastive_results = get_normalized_cosine_sim_per_concept(out_folder_liveability_contrastive, layer)
    liveability_baseline_results = get_normalized_cosine_sim_per_concept(out_folder_liveability_baseline, layer)

    cluster_concepts_for_single_model(hi_contrastive_results)
    cluster_concepts_for_contrastive_encoder(hi_contrastive_results, liveability_contrastive_results)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 3.5))
    fig.subplots_adjust(right=0.8)  # Adjust right side for legend space

    ax_hi_contrastive = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='concept', palette=CONCEPT_COLOR_MAPPING,
                                        data=hi_contrastive_results, ax=axs[0][0], s=5, edgecolor=None)
    ax_hi_contrastive.xaxis.set_major_locator(ticker.NullLocator())
    ax_hi_contrastive.yaxis.set_major_locator(ticker.NullLocator())
    ax_hi_contrastive.set(xlabel=None, ylabel=None)
    ax_hi_contrastive.set_title("Rank-N-Contrast loss")
    ax_hi_contrastive.set_ylabel("Income")

    ax_hi_baseline = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='concept', palette=CONCEPT_COLOR_MAPPING,
                                     data=hi_baseline_results, ax=axs[0][1], s=5, edgecolor=None)
    ax_hi_baseline.xaxis.set_major_locator(ticker.NullLocator())
    ax_hi_baseline.yaxis.set_major_locator(ticker.NullLocator())
    ax_hi_baseline.set(xlabel=None, ylabel=None)
    ax_hi_baseline.set_title("$L_1$ loss")

    ax_liveability_contrastive = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='concept',
                                                 palette=CONCEPT_COLOR_MAPPING,
                                                 data=liveability_contrastive_results, ax=axs[1][0], s=5, edgecolor=None)
    ax_liveability_contrastive.xaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_contrastive.yaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_contrastive.set(xlabel=None, ylabel=None)
    ax_liveability_contrastive.set_ylabel("Liveability")

    ax_liveability_baseline = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='concept',
                                              palette=CONCEPT_COLOR_MAPPING,
                                              data=liveability_baseline_results, ax=axs[1][1], s=5, edgecolor=None)
    ax_liveability_baseline.xaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_baseline.yaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_baseline.set(xlabel=None, ylabel=None)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # remove the legends from the individual axis
    for ax in fig.axes:
        ax.get_legend().remove()

    # Combine lines and labels from all subplots, removing duplicates
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    unique_labels, lines_for_plot = [], []
    for line, label in zip(lines, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            lines_for_plot.append(
                Rectangle((0, 0), 0.5, 0.5, color=line.get_color(), label=label))  # Create square legend symbols

    sorted_legend_items = [label for label in sorted_concepts if label in unique_labels]
    sorted_lines_for_plot = [lines_for_plot[unique_labels.index(label)] for label in sorted_legend_items]

    # Create the legend entirely outside the plot with one column
    legend = fig.legend(sorted_lines_for_plot, sorted_legend_items, loc='upper left', title="Concept",
                        bbox_to_anchor=(0.98, 0.91))  # Adjust location as needed

    # Adjust layout (optional)
    fig.tight_layout()
    plt.savefig(os.path.join(out_folder_income_contrastive, "", "instance_cluster_allignment.png"), bbox_inches='tight',
                dpi=300)

