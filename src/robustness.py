import knn
import pandas as pd
import sys
from pathlib import Path


def robustness_analysis(k, output, family, input, remove):
    """
    Robustness analysis for KNN taxonomic classification
    A partial removal with bootstrapping.
    :param k: hyperparameter k used for KNN
    :param type: int
    :param output: the parent dir for the robustness results
    :param type: str
    :param family: path of tsv of species to taxon mapping
    :param type: str
    :param input: path to input df for KNN classification,
    count df of clusters in each species
    :param type: str
    :param remove: percentage of feature to remove in each
    iteration
    :param type: int
    """
    # Building subdir
    # dir of all features removed in each iteration
    iterations_out = f"{output}/iterations/"
    # dir of the results of LOO classification
    results_out = f"{output}/results/"
    # building the path dirs
    Path(iterations_out).mkdir(parents=True, exist_ok=True)
    Path(results_out).mkdir(parents=True, exist_ok=True)
    # preprocess the dataframe
    processed = knn.pre_process(input, family, k - 1)
    # get a list of unique cluster ID for even sampling
    unique_values = processed["cluster_ID"].unique()
    # 100 iterations
    for i in range(0, 100):
        # make a list of removed items to write to a file
        # iteration and then the removed items in a df
        sampled_values = pd.Series(unique_values).sample(frac=remove, random_state=i)
        df_filtered = processed[~processed["cluster_ID"].isin(sampled_values)]
        # prep the args for the file
        out_file = f"{results_out}iteration_{i}.csv"
        # pass to the leave one out experiment with a return
        leave_one_species_out_robust(df_filtered, out_file, k, i)
        # make a dataframe with what was left out
        iteration = pd.DataFrame(sampled_values, columns=["removed"])
        iteration["iteration"] = i
        # write the features left out in each iteration
        iteration.to_csv(f"{iterations_out}results_{i}.csv", index=False, header=False)


def leave_one_species_out_robust(melted_df, output, k, iteration):
    """
    Leave one out experiment specific to the robustness analysis
    Iteration tracking is included
    :param melted_df: dataframe of the counts in each species
    :param type: pandas df
    :param output: path to write each iteration of KNN results
    :param type: str
    :param k: hyperparameter k
    :param type: int
    :param iteration: the iteration of robustness trial
    :param type: int
    """
    # pivot to mimic pre-processing from original KNN function
    melted_df = melted_df.pivot_table(
        index=["family", "species"], columns="cluster_ID", values="present"
    )
    knn_results = []
    knn_labels = ["species", "known_family", "predicted_family", "score", "iteration"]
    for species in melted_df.index.get_level_values("species").unique():
        # get the training set
        filtered_df = melted_df.loc[
            melted_df.index.get_level_values("species") != species
        ]
        # get the test set
        test_set = melted_df.loc[melted_df.index.get_level_values("species") == species]
        # family label for test value
        real_label = test_set.index.get_level_values("family")[0]
        # make a numpy array of family labels
        family_label = filtered_df.index.get_level_values(level="family").to_numpy()
        # remove family index from df
        filtered_df = filtered_df.reset_index(level="family", drop=True)
        # remove family label from the test point
        test_set = test_set.reset_index(level="family", drop=True)
        # doing the KNN prediction
        pred_family, prob = knn.KNN(filtered_df, family_label, k, test_set)
        # adding the KNN predictions to the results list
        knn_results.append((species, real_label, pred_family, prob, iteration))
    # appending to final results
    k_results = pd.DataFrame(knn_results, columns=knn_labels)
    k_results.to_csv(output, index=False, header=False)


def main():
    # we need to define
    k = int(sys.argv[1])
    remove = float(sys.argv[2])
    # outfile path
    out = sys.argv[3]
    # input file
    input_df = sys.argv[4]
    # family annotations
    family_anno = sys.argv[5]
    # run the robustness analysis
    robustness_analysis(k, out, family_anno, input_df, remove)


if __name__ == "__main__":
    main()
