import numpy as np
from scipy import stats
import pandas as pd
from joblib import Parallel, delayed
import functools
from tqdm import tqdm
from copy import copy


def bootstrap(pairs, data: pd.DataFrame, x, y, hue, hue_order, n_boot=10000, **kwargs):
    rng = np.random.default_rng()
    subsample_mean = lambda arr: np.mean(rng.choice(arr, size=len(arr), replace=True))

    n_pairs = len(pairs)
    for i, p in enumerate(pairs):
        print(p)
        g1 = data[(data[x] == p[0][0]) & (data[hue] == p[0][1])][y].values
        g2 = data[(data[x] == p[1][0]) & (data[hue] == p[1][1])][y].values
        g1_means = np.array([subsample_mean(g1) for _ in range(n_boot)])
        g2_means = np.array([subsample_mean(g2) for _ in range(n_boot)])

        p_val = np.count_nonzero(g1_means >= g2_means) / n_boot
        print(f"{p[0][0]}_{p[0][1]} vs {p[1][0]}_{p[1][1]}: {p_val}")


def resample_sd(df: pd.DataFrame, level="both", apply=None):
    """Resamples variables by sampling with replacement at hierarichy levels: 'session' and 'associated variables'. The associated variables are columns other than the 'session' columns. They could be pairwise correlations, ripple frequencies etc.

    session grp var1 var2 var3 ..........
    0       SD  0.2   0.3  0.4
    0       SD  0.5   0.3  0.4
    1       SD  0.2   0.73  0.2
    1       SD  0.2   0.73  0.2
    2       SD  0.1   0.2  0.23
    2       SD  0.1   0.2  0.23
    5       NSD  0.2   0.3  0.4
    5       NSD  0.2   0.3  0.4
    6       NSD  0.5   0.3  0.4
    6       NSD  0.2   0.73  0.2
    6       NSD  0.1   0.2  0.23

    Parameters
    ----------
    df : pd.DataFrame
        columns should specify session and variables
    level : str, optional
        _description_, by default "both"
    apply : callable/function, optional
        a function to operate on resampled dataframe, by default None

    Returns
    -------
    _type_
        _description_
    """
    animal_ids = df["mouse"].unique()
    n_mice = len(animal_ids)
    # print(sess_ids)
    if level in {"animal", "all"}:
        # bootstrap session_ids
        rng = np.random.default_rng()
        animal_ids = rng.choice(animal_ids, size=n_mice, replace=True)

    new_df = []
    for i, idx in enumerate(animal_ids):
        idx_df = df[df.animal == idx].copy()  # df of variables for that session
        idx_df.loc[:, "animal"] = i  # make selected session independent

        if level in {"all", "session_id"}:
            # bootstrap second level
            session_names = df["session_id"]

            idx_df = idx_df.sample(frac=1, replace=True, ignore_index=True)

        new_df.append(idx_df)
    new_df = pd.concat(new_df, ignore_index=True)

    if apply is not None:
        assert callable(apply), "apply can only be a function"
        new_df = apply(new_df)

    return new_df


def resample_corrs_paired(df_list: pd.DataFrame, apply=None):
    """Resamples variables by sampling with replacement at 'mouse', 'day_pair', and 'corrs_sm' level.
    df_list is a list of pandas dataframes: One in the Shock and the other in the Open (Neutral) arena.

    session grp var1 var2 var3 ..........
    0       SD  0.2   0.3  0.4
    0       SD  0.5   0.3  0.4
    1       SD  0.2   0.73  0.2
    1       SD  0.2   0.73  0.2
    2       SD  0.1   0.2  0.23
    2       SD  0.1   0.2  0.23
    5       NSD  0.2   0.3  0.4
    5       NSD  0.2   0.3  0.4
    6       NSD  0.5   0.3  0.4
    6       NSD  0.2   0.73  0.2
    6       NSD  0.1   0.2  0.23

    Parameters
    ----------
    df_list : [pd.DataFrame, pd.DataFrame]. First dataframe is Shock only, second is Neutral/Open only.
        columns should specify session and variables
    apply : callable/function, optional
        a function to operate on resampled dataframe, by default None

    Returns
    -------
    _type_
        _description_
    """
    mouse_ids = df_list[0]["mouse"].unique()
    n_mice = len(mouse_ids)
    df, df2 = df_list

    rng = np.random.default_rng()
    mouse_ids = rng.choice(mouse_ids, size=n_mice, replace=True)

    new_df = []
    for i, idx in enumerate(mouse_ids):
        idx_df = df[df.mouse == idx].copy()  # df of variables for that session
        idx_df.loc[:, "mouse_rs"] = i  # make selected session independent

        idx_df2 = df2[df2.mouse == idx].copy()  # df of variables for that session
        idx_df2.loc[:, "mouse_rs"] = i  # make selected session independent

        day_pairs = df["day_pair"].unique()
        day_pair = rng.choice(day_pairs, size=1, replace=True)[0]

        idx_df = idx_df[idx_df["day_pair"] == day_pair].sample(frac=1, replace=True, ignore_index=True)
        idx_df2 = idx_df2[idx_df2["day_pair"] == day_pair].sample(frac=1, replace=True, ignore_index=True)

        new_df.append(idx_df)
        new_df.append(idx_df2)
    new_df = pd.concat(new_df, ignore_index=True)

    if apply is not None:
        assert callable(apply), "apply can only be a function"
        new_df = apply(new_df)

    return new_df

def resample(df, level=['mouse', 'session', 'corrs_sm'], n_level=None, apply=None):
    """Resample data with replacement at each level indicated.
    n_level = number of samples to grab at each level. If None, it will resample with replacement from each level n times,
    where n is the number of unique values in that level. If not None, it must be a list the same length as `level`
    with either None or an int for the number of samples to grab from the corresponding level.
    e.g., if you wanted to resample smoothed correlation values from all mice but only grab ONE session from each mouse
    for each bootstrap, you would enter:
    >>> resample(df, level=['mouse', 'session', 'corrs_sm'], n_level=[None, 1, None])"""

    if apply is not None:
        assert callable(apply), "apply can only be a function"
        df = resample(df, level=level, n_level=n_level, apply=None)
        new_df = apply(df)

    else:
        n_level = [None]*len(level) if n_level == None else n_level
        assert (len(n_level) == len(level)) & np.all([isinstance(_, int) or _ is None for _ in n_level]), "`n_level` must be a list of ints and None the same length as `level`"
        if len(level) > 1:
            param = level[0]  # Get name of level to resample at
            next_levels = copy(level[1:])  # Get next levels to resample
            next_n_levels = copy(n_level[1:])
            ids = df[param].unique()  # Grab unique values to resample from, e.g. animal names or session ids
            n_samples = len(ids) if n_level[0] is None else n_level[0]

            # Now resample
            rng = np.random.default_rng()
            # resample_ids = rng.choice(ids, size=len(ids), replace=True)
            resample_ids = rng.choice(ids, size=n_samples, replace=True)

            new_df = []
            # Loop through and generate a new dataframe for each id in the resample ids
            for i, idx in enumerate(resample_ids):
                idx_df = df[df[param] == idx].copy()
                idx_df.loc[:, param] = i  # Make each sample "independent"

                # Recursively call resample to resample at the next level(s)
                # idx_df = resample(idx_df, level=next_levels, apply=apply)
                idx_df = resample(idx_df, level=next_levels, n_level=next_n_levels, apply=apply)
                new_df.append(idx_df)

            new_df = pd.concat(new_df, ignore_index=True)

            # if apply is not None:
            #     assert callable(apply), "apply can only be a function"
            #     new_df = apply(new_df)

        elif len(level) == 1:  # If at the bottom level, actually resample!
            assert level[0] in df.keys(), 'Last parameter in "level" not in keys of df'
            # new_df = [df.sample(frac=1, replace=True, ignore_index=True)]
            new_df = df.sample(frac=1, replace=True, ignore_index=True)

    return new_df


def bootstrap_resample(df: pd.DataFrame, n_iter, n_jobs=1, apply=None,
                       level=['mouse', 'session', 'corrs_sm'], n_level=None):
    # groups = df["grp"].unique()

    partial_resample = functools.partial(resample, level=level, n_level=n_level, apply=apply)
    out_df = []
    # for grp in groups:
    #     print(f"Running bootstraps for {grp} group")
    #     df_grp = df[df["grp"] == grp]
    data = [
        r
        for r in tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(partial_resample)(df) for _ in range(n_iter)
            ),
            total=n_iter,
            # position=1,
            # leave=False,
        )
    ]
    data = pd.concat(data, ignore_index=True)
        # data["grp"] = grp
        # out_df.append(data)

    # return pd.concat(out_df, ignore_index=True)
    return data

def get_bootstrap_prob(sample1, sample2):
    """
    get_direct_prob Returns the direct probability of items from sample2 being
    greater than or equal to those from sample1.
       Sample1 and Sample2 are two bootstrapped samples and this function
       directly computes the probability of items from sample 2 being greater
       than or equal to those from sample1. Since the bootstrapped samples are
       themselves posterior distributions, this is a way of computing a
       Bayesian probability. The joint matrix can also be returned to compute
       directly upon.
    obtained from: https://github.com/soberlab/Hierarchical-Bootstrap-Paper/blob/master/Bootstrap%20Paper%20Simulation%20Figure%20Codes.ipynb

    References
    ----------
    Saravanan, Varun, Gordon J Berman, and Samuel J Sober. “Application of the Hierarchical Bootstrap to Multi-Level Data in Neuroscience.” Neurons, Behavior, Data Analysis and Theory 3, no. 5 (2020): https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience.

    Parameters
    ----------
    sample1: array
        numpy array of values
    sample2: array
        numpy array of values

    Returns
    ---------
    pvalue1
        using joint probability distribution
    pvalue2
        using the number of sample2 being greater than sample1
    2d array
        joint probability matrix
    """

    # assert len(sample1) == len(sample2), "both inputs lengths should match"
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    joint_low_val = min([min(sample1), min(sample2)])
    joint_high_val = max([max(sample1), max(sample2)])

    nbins = 100
    p_axis = np.linspace(joint_low_val, joint_high_val, num=nbins)
    edge_shift = (p_axis[2] - p_axis[1]) / 2
    p_axis_edges = p_axis - edge_shift
    p_axis_edges = np.append(p_axis_edges, (joint_high_val + edge_shift))

    # Calculate probabilities using histcounts for edges.

    p_sample1 = np.histogram(sample1, bins=p_axis_edges)[0] / np.size(sample1)
    p_sample2 = np.histogram(sample2, bins=p_axis_edges)[0] / np.size(sample2)

    # Now, calculate the joint probability matrix:
    # p_joint_matrix = np.zeros((nbins, nbins))
    # for i in np.arange(np.shape(p_joint_matrix)[0]):
    #     for j in np.arange(np.shape(p_joint_matrix)[1]):
    #         p_joint_matrix[i, j] = p_sample1[i] * p_sample2[j]

    p_joint_matrix = p_sample1[:, np.newaxis] * p_sample2[np.newaxis, :]

    # Normalize the joint probability matrix:
    p_joint_matrix = p_joint_matrix / p_joint_matrix.sum()

    # Get the volume of the joint probability matrix in the upper triangle:
    p_test = np.sum(np.triu(p_joint_matrix))

    p_test = 1 - p_test if p_test >= 0.5 else p_test

    statistic = np.abs(sample1.mean() - sample2.mean()) / np.sqrt(
        (sample1.std() ** 2 + sample2.std() ** 2) / 2
    )
    return statistic, p_test


def get_bootstrap_prob_paired(arr1, arr2):
    l1, l2 = len(arr1), len(arr2)
    assert l1 == l2, f"len(arr1)={l1},len(arr2)={l2}: both inputs lengths should match"
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    delta = arr1 - arr2
    p_test = np.sum(delta >= 0) / len(delta)
    p_test = 1 - p_test if p_test >= 0.5 else p_test

    return np.nanmean(delta), p_test
