import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor
from src.utils import plot_binary_outliers


df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
outlier_columns = list(df.columns[:6])

plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

## Testing Boxplot as an outlier detector (distribution based)
df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 3))


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1, Q3 = dataset[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


## Testing Chauvent's Criteron as an outlier detector (distribution based)

df[outlier_columns[:3] + ["label"]].plot.hist(
    by="label", figsize=(20, 10), layout=(3, 3)
)
df[outlier_columns[3:] + ["label"]].plot.hist(
    by="label", figsize=(20, 10), layout=(3, 3)
)
# Checking for normal distribution


def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )


## Testing LOF as an outlier detector (distance based)
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    # X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return (dataset, outliers)


dataset, outliers = mark_outliers_lof(df, outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True
    )


## Testing IsolationForest as an outlier detector
from sklearn.ensemble import IsolationForest


def mark_outliers_isolation_forest(dataset, columns, n=0.1):

    dataset = dataset.copy()

    iso_forest = IsolationForest(contamination=n)
    data = dataset[columns]
    outliers = iso_forest.fit_predict(data)

    dataset["outlier_isf"] = outliers == -1
    return (dataset, outliers)


dataset, outliers = mark_outliers_isolation_forest(df, outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col="outlier_isf", reset_index=True
    )


## Testing Gaussian (Normal) Distribution method as an outlier detector (distribution based)


def mark_outliers_gaussian(dataset, col, threshold=3):
    """
    Function to mark values as outliers using the Gaussian distribution method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column to apply outlier detection to
        threshold (float): The Z-score threshold to use for identifying outliers

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    mean = dataset[col].mean()
    std = dataset[col].std()
    z_scores = (dataset[col] - mean) / std
    dataset[col + "_outlier"] = np.abs(z_scores) > threshold
    return dataset


for col in outlier_columns:
    dataset = mark_outliers_gaussian(df, col)
    plot_binary_outliers(
        dataset=dataset, col=col, outlier_col=col + "_outlier", reset_index=True
    )
