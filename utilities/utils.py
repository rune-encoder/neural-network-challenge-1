import pandas as pd

# Feature selection
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Model evaluation metrics
import statsmodels.api as sm
import numpy as np


def highlight_vif(row: pd.Series, threshold: float) -> list:
    """
    Highlight VIF values below a given threshold.

    Parameters:
    row (pd.Series): A row of VIF values.
    threshold (float): The threshold for highlighting.

    Returns:
    list: A list of styles for each cell in the row.
    """
    return ["background-color: black" if value < threshold else "" for value in row]


def calc_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with features.

    Returns:
    pd.DataFrame: A DataFrame containing VIF values for each feature.
    """
    vif_values = [variance_inflation_factor(
        df.values, i) for i in range(df.shape[1])]
    vif = pd.DataFrame(data={"VIF": vif_values}, index=df.columns).sort_values(
        by="VIF", ascending=True)
    return vif


def highlight_p_values(row: pd.Series) -> list:
    """
    Highlight p-values below a significance level of 0.05.

    Parameters:
    row (pd.Series): A row of p-values.

    Returns:
    list: A list of styles for each cell in the row.
    """
    return ["background-color: black" if value <= 0.05 else "" for value in row]


def calc_p_values(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Calculate p-values for each feature using OLS regression.

    Parameters:
    X (pd.DataFrame): The input DataFrame with features.
    y (pd.Series): The target variable.

    Returns:
    tuple: A DataFrame containing p-values for each feature and the OLS model.
    """
    ols_model = sm.OLS(y, X).fit()
    p_values_df = ols_model.pvalues.sort_values().to_frame(name="p_value")
    return p_values_df, ols_model


def calc_correlation(df):
    """
    Calculate the correlation matrix and apply a color gradient for visualization.

    Parameters:
    df (pd.DataFrame): The input DataFrame with features.

    Returns:
    pd.io.formats.style.Styler: A styled DataFrame with the correlation matrix.
    """
    corr_matrix = df.corr()
    styled_corr_matrix = corr_matrix.style.background_gradient(cmap="coolwarm")
    return styled_corr_matrix
