import numpy as np
import pandas as pd


def get_clusters_similarity(
    vec_type_1: str,
    vec_type_2: str,
    k: int,
    method: str = "TSNE",
    before_after: str = "before",
):
    clusters_1_df = pd.read_csv(
        f"results/clustering {before_after} {method} {vec_type_1} k={k}.csv"
    )
    clusters_2_df = pd.read_csv(
        f"results/clustering {before_after} {method} {vec_type_2} {k=}.csv"
    )
    dfs = [clusters_1_df, clusters_2_df]
    names = clusters_1_df["names"].unique()
    n = len(names)

    # for each pair of categories names -> check whether the 2 clustering agree:
    agree_results = []
    score_results = []
    for i in range(n):
        for j in range(i + 1, n):
            name_i = names[i]
            name_j = names[j]
            # check if they are in the same cluster in each df
            decision = []
            for df in dfs:
                cluster_i = df[df["names"] == name_i]["labels"].item()
                cluster_j = df[df["names"] == name_j]["labels"].item()
                in_same_cluster = cluster_i == cluster_j
                decision.append(in_same_cluster)
            # if dfs agree  [True, False]
            if decision[0] == decision[1]:
                agree_results.append(1)  # agree
                if decision[0]:  # True
                    score_results.append(1)
            else:
                agree_results.append(0)  # disagree
                score_results.append(0)

    decisions = np.round(np.mean(agree_results), 2)
    score = np.round(np.mean(score_results), 2)

    print(
        f"{k=}: {vec_type_1} and {vec_type_2} clustering results agree on {decisions*100}% of their decisions"
    )
    print(f"{k=}: {vec_type_1} and {vec_type_2} clustering similarity score={score*100}")


if __name__ == "__main__":
    get_clusters_similarity(vec_type_1="fMRI", vec_type_2="Glove", k=5)
    get_clusters_similarity(vec_type_1="fMRI", vec_type_2="BERT", k=5)
    get_clusters_similarity(vec_type_1="BERT", vec_type_2="Glove", k=5)
    get_clusters_similarity(vec_type_1="fMRI", vec_type_2="Glove", k=10)
    get_clusters_similarity(vec_type_1="fMRI", vec_type_2="BERT", k=10)
    get_clusters_similarity(vec_type_1="BERT", vec_type_2="Glove", k=10)
