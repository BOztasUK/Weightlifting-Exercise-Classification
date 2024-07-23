import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction

df = pd.read_pickle("../../data/interim/02_removed_outliers_dataset.pkl")

predictor_columns = list(df.columns[:6])

for col in predictor_columns:
    df[col] = df[col].interpolate()


for s in df["set"].unique():

    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

avg_rep_duration_heavy = duration_df.iloc[0] / 5
avg_rep_duration_medium = duration_df.iloc[1] / 10


# Butterwork Lowpass filter
df_lowpass = df.copy()
lowPass = LowPassFilter()

fs = 1000 / 200
cutoff = 1.2

for col in predictor_columns:
    df_lowpass = lowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]


# Apply PCA
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained varaince ")
plt.show()

# Chosen 3 as principal component number using the elbow technique
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)


# creating sum of squares feature
df_squared = df_pca.copy()
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)
