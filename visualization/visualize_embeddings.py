import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from visualize_tcav import get_out_folder


layer = "encoder.encoder.avgpool"

def get_tsne_embeddings_with_prediction(outfolder, target_layer):

    tsne_activations = pd.read_csv(os.path.join(outfolder, "{}_tsne_instances_cavs.csv".format(target_layer)), index_col=0)
    predictions = pd.read_csv(os.path.join(outfolder, "predictions.csv"), index_col=0)

    activations_with_labels = pd.merge(tsne_activations, predictions, left_index=True, right_on="image_id")

    return activations_with_labels

if __name__ == '__main__':


    out_folder_income_baseline, model_type = get_out_folder("household_income", False)
    out_folder_income_finetuned, model_type = get_out_folder("household_income", True)

    out_folder_liveability_baseline, model_type = get_out_folder("Liveability", False)
    out_folder_liveability_finetuned, model_type = get_out_folder("Liveability", True)

    hi_finetuned_results = get_tsne_embeddings_with_prediction(out_folder_income_finetuned, layer)
    hi_baseline_results = get_tsne_embeddings_with_prediction(out_folder_income_baseline, layer)

    liveability_finetuned_results = get_tsne_embeddings_with_prediction(out_folder_liveability_finetuned, layer)
    liveability_baseline_results = get_tsne_embeddings_with_prediction(out_folder_liveability_baseline, layer)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 3.5))
    plt.subplots_adjust(right=0.8)  # Adjust margins for spacing

    ax_hi_finetuned = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='label', palette="plasma", data=hi_finetuned_results, ax=axs[0][0], s=5, edgecolor=None)
    ax_hi_finetuned.xaxis.set_major_locator(ticker.NullLocator())
    ax_hi_finetuned.yaxis.set_major_locator(ticker.NullLocator())
    ax_hi_finetuned.set(xlabel=None, ylabel=None)
    ax_hi_finetuned.set_title("Rank-N-Contrast loss")
    ax_hi_finetuned.set_ylabel("Income")
    ax_hi_finetuned.get_legend().remove()
    norm_income = plt.Normalize(hi_finetuned_results['label'].min(), hi_finetuned_results['label'].max())
    income_cm = plt.cm.ScalarMappable(cmap="plasma", norm=norm_income)

    ax_hi_baseline = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='label', palette="plasma",
                                  data=hi_baseline_results, ax=axs[0][1], s=5, edgecolor=None)
    ax_hi_baseline.xaxis.set_major_locator(ticker.NullLocator())
    ax_hi_baseline.yaxis.set_major_locator(ticker.NullLocator())
    ax_hi_baseline.set(xlabel=None, ylabel=None)
    ax_hi_baseline.set_title("$L_1$ loss")
    ax_hi_baseline.get_legend().remove()
    ax_liveability_finetuned = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='label', palette="plasma",
                                  data=liveability_finetuned_results, ax=axs[1][0], s=5, edgecolor=None)
    ax_liveability_finetuned.xaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_finetuned.yaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_finetuned.set(xlabel=None, ylabel=None)
    ax_liveability_finetuned.set_ylabel("Liveability")
    ax_liveability_finetuned.get_legend().remove()

    norm_liveability = plt.Normalize(liveability_finetuned_results['label'].min(), liveability_finetuned_results['label'].max())
    liveability_cm = plt.cm.ScalarMappable(cmap="plasma", norm=norm_liveability)

    ax_liveability_baseline = sns.scatterplot(x='tsne_dim_1', y='tsne_dim_2', hue='label', palette="plasma",
                                 data=liveability_baseline_results, ax=axs[1][1], s=5, edgecolor=None)
    ax_liveability_baseline.xaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_baseline.yaxis.set_major_locator(ticker.NullLocator())
    ax_liveability_baseline.set(xlabel=None, ylabel=None)
    ax_liveability_baseline.get_legend().remove()

    income_colorbar = fig.colorbar(income_cm, ax=axs[0][1], fraction=0.05, pad=0.02)
    income_colorbar.ax.set_title("Income", fontsize=10)

    liveability_colorbar = fig.colorbar(liveability_cm, ax=axs[1][1], fraction=0.05, pad=0.02)
    liveability_colorbar.ax.set_title("Liveability", fontsize=10)


    fig.tight_layout()
    plt.savefig(os.path.join(out_folder_income_finetuned, "", "tsne_all_models.png"),
                #bbox_extra_artists=[legend], bbox_inches='tight',
                dpi=300)

