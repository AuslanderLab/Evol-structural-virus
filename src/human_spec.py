# Imports --------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# Universal plot parameters
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.facecolor"] = "none" 

def prevalence_rules(rt1):

    """
    This chooses prevalent clusters with consistent trends (up or down)
    :param rt1: it is the difference dataframe in the
    :return type: list
    :return: clusters to be used in downstream analysis
    """
    c1 = rt1.max() > 0
    c2 = rt1.min() < 0
    c11 = c1[(c1 > 0) | (c2 > 0)].index

    tbb = rt1[c11]  # keep any cluster where there is a difference in prevalence

    xh = tbb > 0
    xl = tbb < 0
    xh.sum()[
        (xh.sum() > 0) & (xl.sum() < 2)
    ]  ##(1) clusters that are higher in human viruses in at least 2 families and never higher in non-human viruses
    xh.sum()[
        (xl.sum() > 0) & (xh.sum() < 2)
    ]  ##(2) clusters that are higher in non-human viruses in at least 2 families and never higher in human viruses

    ## get those that are consistently higher or lower in human  infecting across families (1) + (2) above
    include_clst = list(xh.sum()[(xh.sum() > 1) & (xl.sum() == 0)].keys()) + list(
        xh.sum()[(xl.sum() > 1) & (xh.sum() == 0)].keys()
    )
    return include_clst


def prevalence_analysis(file):
    """
    This computes the prevalence of human-infective clusters and does visualization
    :param file: the merged.tax.tsv file from the Nomburg dataset
    :type file: str
    """

    ##Get a table of species X cluster
    
    dat = pd.read_csv(file, sep="\t")
    dat["v"] = [1 for i in range(len(dat))]

    # pivot table so row is the species and columns are clusters
    tab = dat.pivot_table(
        columns="cluster_ID",
        index="species",
        values="v",
        aggfunc="max",
        fill_value=0,
    )

    # Dictionaries to Map species to higher taxa levels
    sp2gen = {dat["species"][i]: dat["genus"][i] for i in range(len(dat))}
    sp2fam = {dat["species"][i]: dat["family"][i] for i in range(len(dat))}
    sp2ord = {dat["species"][i]: dat["order"][i] for i in range(len(dat))}
    sp2clas = {dat["species"][i]: dat["class"][i] for i in range(len(dat))}
    sp2phyl = {dat["species"][i]: dat["phylum"][i] for i in range(len(dat))}


    ### A way to order the viruses by phyla (used internally and could be useful)
    tab2 = tab.copy()
    phylastr = [
        str(sp2gen[tab2.index[i]])
        + "_"
        + str(sp2fam[tab2.index[i]])
        + "_"
        + str(sp2ord[tab2.index[i]])
        + "_"
        + str(sp2clas[tab2.index[i]])
        + "_"
        + str(sp2phyl[tab2.index[i]])
        for i in range(len(tab2.index))
    ]

    srtar = np.argsort(phylastr)
    srttab = tab2.iloc[srtar].copy()

    ##Get the % presence of each cluster in every family
    taba = tab.copy()  # copy so the original stays the same
    taba.index = [
        sp2fam[taba.index[i]] for i in range(len(taba.index))
    ]  # extract the family
    # ta - the % presence of each cluster in every family, human and non
    ta = taba.groupby(
        taba.index
    ).mean()  
    ########################## Defining human-specific viruses ####################################
    # defining the human specific viruses
    hu = [
        dat["species"][i]
        for i in range(len(dat["species"]))
        if "Human" in str(dat["species"][i]) # contains Human
    ]

    # add in the uniprot human information
    human_inf = pd.read_csv(
        "./intermediate_files/Uniprot_taxonomy_host_9606_2025_07_14.csv"
    )
    taxidhu = [str(i) for i in human_inf["Taxon Id"]]

    # add in the ncbi host information
    human_inf_2 = pd.read_csv(
        "./intermediate_files/human_infecting_viruses_ncbi.csv",
        header=None,
        names=["species"],
    )
    taxidhu2 = [str(i) for i in human_inf_2["species"]]

    hu = [
        dat["species"][i]  # keep i if
        for i in range(len(dat["species"]))  # loop over all the species
        if (
            (
                str(dat["taxonID"][i]) in taxidhu or str(dat["species"][i]) in taxidhu2
            )  # keep if in Human db
            or (
                "Human" in str(dat["species"][i])
                and "Human_associated" not in str(dat["species"][i])
            )  # include if human in name, but not associated
        )
    ]


    ##Get the % presence of each cluster in every family - only human infecting ones
    tabh = tab.loc[hu].copy()
    tabh.index = [sp2fam[tabh.index[i]] for i in range(len(tabh.index))]
    th = (
        tabh.groupby(tabh.index).mean()
    )  # th - the % presence of each cluster in every family of human infecting virses only

    ##tabke the families that have any human infecting viruses for the figure
    lss = list(set(ta.index) & set(th.index))
    taa = ta.loc[lss]
    thh = th.loc[lss]

    ## diff of (the % presence of cluster in family for human infecting viruses) -(the % presence of cluster in family for all viruses)
    df = thh - taa


    # Plot
    included_clust = prevalence_rules(df)
    g = sns.clustermap(df[included_clust], cmap="bwr", vmin=-0.25, vmax=0.25)
    g.savefig("./results/human_specific_prots.png")

    ########################## Add in alternative analysis with negative set ####################################
    # this ended up being the analysis used in the paper
    neg_h = tab.loc[~tab.index.isin(hu)].copy()
    neg_h.index = [
        sp2fam[neg_h.index[i]] for i in range(len(neg_h.index))
    ]  # replacing with family labels
    nh = neg_h.groupby(neg_h.index).mean()  # computing the mean of negative values

    # make sure you are comparing the same families
    ss = list(set(nh.index) & set(th.index))  # extract shared index
    th_pos = th.loc[ss]
    nh_neg = nh.loc[ss]

    diff = th_pos - nh_neg  # compute the pairwise positive- negative matrices

    sig_clust = prevalence_rules(diff) 
    df_save = diff[sig_clust]
    df_save.to_csv("./results/prevalence_change_analysis.csv")
    p = sns.clustermap(diff[sig_clust], cmap="bwr", vmin=-0.25, vmax=0.25)
    p.savefig("./results/human_specific_prots_neg_set.png")

    ## apply the same rules for selection as above

    ########################## Add in alternative analysis with a foldchange analysis ####################################
    crtl = tab.loc[~tab.index.isin(hu)].copy()
    case = tab.loc[tab.index.isin(hu)].copy()

    # normalize
    crtl = crtl.sum() / len(crtl)
    case = case.sum() / len(case)

    # calculate the log fc
    fc = np.log2((case + 1) / (crtl + 1))
    # filtering??? very arbitrary
    fc = (fc[(fc < -0.15) | (fc > 0.15)]).index

    # select the columns
    crtl_count = tab.loc[~tab.index.isin(hu)].copy()  # process the negative set
    crtl_count = crtl_count[fc]
    crtl_count.index = [
        sp2fam[crtl_count.index[i]] for i in range(len(crtl_count.index))
    ]  # replacing with family labels
    ctl = crtl_count.groupby(crtl_count.index).mean()

    case_count = tab.loc[tab.index.isin(hu)].copy()  # process the positive set
    case_count = case_count[fc]
    case_count.index = [
        sp2fam[case_count.index[i]] for i in range(len(case_count.index))
    ]  # replacing with family labels
    cs = case_count.groupby(case_count.index).mean()

    ss = list(set(cs.index) & set(ctl.index))  # extract shared index
    ctl = ctl.loc[ss]
    cs = cs.loc[ss]

    diff = cs - ctl
    p = sns.clustermap(diff, cmap="bwr", vmin=-0.25, vmax=0.25)
    p.savefig("./results/human_specific_prots_lfc.png")

def main():
    input=sys.argv[1]
    prevalence_analysis(input)


if __name__=="__main__":
    main()