import knn
import pandas as pd
import sys
import numpy as np


def human_analysis(input, family, k, outpath):
    """
    KNN classification of human-specific viruses
    :param input: path to the count df, row clusters, col species
    :param type: str
    :param family: path to the tsv mapping of species to family
    :param type: str
    :param k: hyperparameter k for classification
    :param type: int
    :param outpath: path of classification results
    :param type: str
    """
    # don't run if k+1 in a family because it's going to be family only classification
    preprocessed_df = knn.pre_process(input, family, k + 1)
    # merge the taxon id with the species
    species_taxid = pd.read_csv(
        "./intermediate_files/species_taxid.tsv", sep="\t", skiprows=[1]
    )
    # filter species with no id to prevent overflow.
    # (should also be prevented by left join)
    taxid_map = (
        species_taxid.dropna()
        .drop_duplicates(subset="species")
        .set_index("species")["taxonID"]
    )
    # create a new taxonID column
    preprocessed_df["taxonID"] = preprocessed_df["species"].map(taxid_map)
    # if this errors its because the annotation files can't be found
    human_inf = pd.read_csv("./intermediate_files/Uniprot_human_taxon_id.csv")
    human_inf_2 = pd.read_csv(
        "./intermediate_files/human_infecting_viruses_ncbi.csv",
        header=None,
        names=["species"],
    )
    # make into a species list
    hu = [str(i) for i in human_inf["taxonID"]]
    hu2 = [str(i) for i in human_inf_2["species"]]
    # this is from the dataframe
    hu3 = [
        val
        for val in preprocessed_df["species"]
        if "human" in str(val).lower() and "associated" not in str(val).lower()
    ]
    human = hu2 + hu3
    # now add the human column to the df if the species is in the list
    preprocessed_df["human"] = np.where(
        (preprocessed_df["species"].str.lower().isin([s.lower() for s in human]))
        | (preprocessed_df["taxonID"].isin(hu)),
        "Human",
        "Non-human",
    )
    # now drop the taxid column
    preprocessed_df.drop("taxonID", axis=1, inplace=True)
    # now drop duplicates
    preprocessed_df.drop_duplicates(subset=["species"])

    # now do the leave one out experiment within the family
    for family in preprocessed_df["family"].unique():
        # filter the family of interest
        sub_df = preprocessed_df.loc[preprocessed_df["family"] == family]
        # if there are not enough human infecting viruses, don't classify this family
        print(sub_df.loc[sub_df["human"] == "Human"].shape)
        if sub_df.loc[sub_df["human"] == "Human"].shape[0] < k:
            continue
        else:
            print(sub_df.loc[sub_df["human"] == "Human"].shape)
        # drop the family column
        sub_df.drop("family", axis=1, inplace=True)
        # rename the column to family for compatibility
        sub_df.rename(columns={"human": "family"}, inplace=True)
        filename = f"{outpath}/{family}.csv"
        print(filename)
        knn.leave_one_species_out(sub_df, filename, k)


def main():
    # hyperparameter
    k = int(sys.argv[1])
    # outfile path
    out = sys.argv[2]
    # input file
    input_df = sys.argv[3]
    # family annotations
    family_anno = sys.argv[4]
    # running the human_knn classification
    human_analysis(input_df, family_anno, k, out)


if __name__ == "__main__":
    main()
