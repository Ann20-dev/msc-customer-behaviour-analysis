"""Missingness analysis and KDE imputation for the streaming customer dataset.

`missing_pattern` runs a two-sample t-test to check whether a feature's missingness is related
to the target (a missing-at-random / MCAR check). `kde_imputation` fills numeric gaps by
sampling from a Kernel Density Estimate of the observed values, which preserves each feature's
shape and variance.
"""

import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity


def missing_pattern(df, feature_col, target_col):
    """Test whether missingness in feature_col is related to target_col via a two-sample t-test."""
    if feature_col not in df.columns or target_col not in df.columns:
        print("Error: one or both columns not found in the DataFrame.")
        return None

    analysis_df = df.copy()
    analysis_df["is_missing"] = analysis_df[feature_col].isna().astype(int)   # 1 where missing

    non_missing_group = analysis_df.loc[analysis_df["is_missing"] == 0, target_col]
    missing_group = analysis_df.loc[analysis_df["is_missing"] == 1, target_col]
    if missing_group.empty:
        print(f"No missing values in '{feature_col}'.")
        return None

    # group statistics, then Welch's t-test (equal_var=False, safer when variances differ)
    statistics = (analysis_df.groupby("is_missing")[target_col]
                  .agg(["count", "mean", "std", "min", "max"])
                  .rename(index={0: "non_missing", 1: "missing"}))
    t_stat, p_value = stats.ttest_ind(non_missing_group, missing_group, equal_var=False)

    # p >= 0.05 means the target mean does not differ with missingness, i.e. missingness is not
    # predictive of the target (consistent with MCAR)
    return {"Feature_Checked": feature_col, "Target_Variable": target_col,
            "Statistics": statistics,
            "T-Test_Results": {"T_Statistic": t_stat, "P_Value": p_value}}


def kde_imputation(df, columns_to_impute):
    """Impute numeric columns by sampling from a KDE of their observed values."""
    df_imputed = df.copy()
    for col in columns_to_impute:
        if df_imputed[col].dtype in ["int64", "float64"] and df_imputed[col].isna().any():
            observed = df_imputed[col].dropna().values.reshape(-1, 1)
            missing_idx = df_imputed[df_imputed[col].isna()].index
            # 'scott' picks the bandwidth automatically to balance bias and variance
            kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(observed)
            samples = kde.sample(n_samples=len(missing_idx), random_state=42).flatten()
            df_imputed.loc[missing_idx, col] = samples
            print(f"Imputed {len(missing_idx)} values in '{col}' using KDE.")

    # Age is a whole number, and negative Monthly_Spend rows are invalid so they are dropped
    df_imputed["Age"] = df_imputed["Age"].round().astype(int)
    df_imputed = df_imputed[df_imputed["Monthly_Spend"] >= 0].copy()
    return df_imputed
