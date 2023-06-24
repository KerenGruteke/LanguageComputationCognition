import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimension_and_plot(
    dimension_redcution_method,
    vectors_matrix,
    names,
    labels,
    embedding_name,
    K,
    plot_names=True,
):
    """
    :param vectors_matrix: vectors data
    :param names: list that contains for each vector its category name
    :param labels: list that contains for each vector its cluster number
    :param embedding_name: the embedding name
    :param K: k clusters
    :param plot_names: flag to indicate whether to plot names or not
    :return:
    """

    if dimension_redcution_method == "TSNE":
        # Create a t-SNE instance
        reduce_model = TSNE(n_components=K, random_state=42, perplexity=15)
    elif dimension_redcution_method == "UMAP":
        reduce_model = umap.UMAP(
            n_components=K, random_state=42
        )  # Reduce to 2 dimensions
    elif dimension_redcution_method == "PCA":
        # Create a PCA instance
        reduce_model = PCA(n_components=2)  # Reduce to 2 principal components

    # Fit the data and perform dimensionality reduction
    X_reduced = reduce_model.fit_transform(vectors_matrix)

    if dimension_redcution_method == "PCA":
        # print explained_variance_ratio_
        print("explained variance ratio:\n")
        print(reduce_model.explained_variance_ratio_)

    # Plot the t-SNE results with sample names and color-coded class labels
    fig, ax = plt.subplots()
    colors_list = sns.color_palette("husl", len(set(labels)))
    color_dict = {label: colors_list[i] for i, label in enumerate(set(labels))}

    # colors = [color_dict[label] for label in labels]
    # scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colors, label=labels)

    if plot_names:
        for i in range(X_reduced.shape[0]):
            plt.text(
                X_reduced[i, 0],
                X_reduced[i, 1],
                names[i],
                fontsize=8,
                ha="center",
                va="center",
            )

    # Create legend handles and labels based on unique labels and colors
    unique_labels = list(set(labels))
    legend_handles = [
        mpatches.Patch(color=color_dict[label], label=label) for label in unique_labels
    ]

    # Add legend to the plot
    ax.legend(handles=legend_handles, loc="lower right", title="Classes")

    plt.xlabel(f"{dimension_redcution_method} Component 1")
    plt.ylabel(f"{dimension_redcution_method} Component 2")
    plt.title(f"{dimension_redcution_method} {embedding_name} k={K}")
    plt.show()

    df = pd.DataFrame(data={"names": names, "labels": labels})
    df_sorted = df.sort_values(by=["labels", "names"])
    df_sorted.to_csv(f"{dimension_redcution_method} {embedding_name} k={K}.csv")
