# importing libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import sys
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def pre_process(df, family_anno, min_family_size):
    """
    Preprocessing the dataframe before KNN
    :param df: filepath to count df taxon- col, cluster -row
    :param type: str
    :param family anno: str of file mapping the taxon in the df
    to taxonomic level of interest, does not need to be family
    :param min_family_size: the minimum size of the taxon level
    in order for the species to be included in the final df
    :param type: int
    :return:
    :return type: pandas Dataframe
    """
    # reading in the counts df
    df=pd.read_csv(df, index_col="cluster_ID")
    # reading in the family df, skip the first for compatibility
    family=pd.read_csv(family_anno, sep="\t", header=None, skiprows=1,
                       names=["family", "species"])
    # remove items without taxon label, NA/unknown is not a class
    family.dropna(inplace=True)
    # pivot the column names into a column
    melted_df=df.melt(var_name="species", value_name="present",
                      ignore_index=False)
    melted_df=melted_df.reset_index()
    # join on species, adding the family labels
    melted_df=pd.merge(family,melted_df, how="inner", on="species")
    # count the number of unique species in a family
    melted_df["counts"]=melted_df.groupby("family")["species"].transform("nunique")
    #removes species from families with less than the min taxon size
    melted_df=melted_df.loc[melted_df["counts"]>min_family_size]
    # remove the counts column
    melted_df.drop("counts", axis=1, inplace=True)
    return melted_df

def leave_one_species_out(melted_df, output,k_size):
    """
    Leave-one out experiment to evaluate performance of the KNN
    :param melted_df: preprocessed count df
    :param type: Pandas Dataframe
    :param output: filepath to write results
    :param type: str
    :param k_size: hyperparameter k used for classification
    :param type: int
    """
    # turned into rows- species (observations)
    # cols- clusters (features)
    melted_df=melted_df.pivot_table(index=["family","species"],
                                    columns="cluster_ID", values="present")
    # adding the results into a list
    knn_results=[]
    # results labels and order
    knn_labels=["species", "known_taxa", "predicted_taxa", "score"]
    # iterate over all unique species in the index
    for species in melted_df.index.get_level_values("species").unique():
        # remove species from df
        filtered_df=melted_df.loc[melted_df.index.get_level_values("species")!=species]
        # get the observation from the dataframe
        test_set=melted_df.loc[melted_df.index.get_level_values("species")==species]
        # extract family classification
        real_label=test_set.index.get_level_values("family")[0]
        # get the family class
        family_label=filtered_df.index.get_level_values(level="family").to_numpy()
        # remove the family label from the input
        filtered_df=filtered_df.reset_index(level="family", drop=True)
        # remove the family index before classification
        test_set=test_set.reset_index(level="family", drop=True)
        # results from the KNN prediction
        pred_family, prob=KNN(filtered_df,family_label,k_size, test_set)
        # adding the KNN predictions to the results list
        knn_results.append((species, real_label, pred_family, prob))
    # make the nested list of results into a df
    k_results=pd.DataFrame(knn_results, columns=knn_labels)
    # write to a csv of results
    k_results.to_csv(output, index=False)

def KNN(X,Y,k,test_point):
    """
    KNN classification
    :param X: the input for the classifier
    :param type: pandas df
    :param Y: the labels for each observation
    :param type: numpy array
    :param k: hyperparameter of number of nearest neighbors
    :param type: int
    :param test_point: the observation vector
    :param type: pandas df
    :return: predicted class, probability of the class
    :return type: str, int
    """
    # intialization
    knn=KNeighborsClassifier(n_neighbors=k, metric="jaccard")
    # fitting to the previous observations
    knn.fit(X,Y)
    # get the classification of test point
    classification=knn.predict(test_point)
    # retrieve the classification index
    col=np.where(knn.classes_==classification[0])[0]
    # get a number for how many voted for this class
    prob=knn.predict_proba(test_point)
    return str(classification[0]), prob[0,col][0]

######MAIN########
def main():
    df=sys.argv[1]
    family_species=sys.argv[2]
    output=sys.argv[3]
    k=int(sys.argv[4])
    # the minimum family size is k-1 for basic classification
    processed_df=pre_process(df,family_species, k-1)
    leave_one_species_out(processed_df, output, k)
if __name__=="__main__":
    main()
