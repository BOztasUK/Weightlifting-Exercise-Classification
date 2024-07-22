import pandas as pd
from glob import glob

# single_file_acc = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
# )

# single_file_gyr = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
# )

# files = glob("../../data/raw/MetaMotion/*.csv")
# len(files)

# data_path = "../../data/raw/MetaMotion/"

# # f = files[1]

# # participant = f.split("-")[0].replace(data_path, "")
# # label = f.split("-")[1]
# # category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

# # df = pd.read_csv(f)

# # df["participant"] = participant
# # df["label"] = label
# # df["category"] = category

# acc_df = pd.DataFrame()
# gyr_df = pd.DataFrame()

# acc_set = 1
# gyr_set = 1

# for f in files:
#     participant = f.split("-")[0].replace(data_path, "")
#     label = f.split("-")[1]
#     category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

#     df = pd.read_csv(f)

#     df["participant"] = participant
#     df["label"] = label
#     df["category"] = category

#     if "Acceleromete" in f:
#         df["set"] = acc_set
#         acc_set += 1
#         acc_df = pd.concat([acc_df, df])
#     else:
#         df["set"] = gyr_set
#         gyr_set += 1
#         gyr_df = pd.concat([gyr_df, df])


# acc_df.info()

# pd.to_datetime(df['epoch (ms)'], unit='ms')
# pd.to_datetime(df['time (01:00)']).dt.month


# acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
# gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

# del acc_df["epoch (ms)"]
# del acc_df["time (01:00)"]
# del acc_df['elapsed (s)']

# del gyr_df["epoch (ms)"]
# del gyr_df["time (01:00)"]
# del gyr_df['elapsed (s)']

data_path = "../../data/raw/MetaMotion/"
files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Acceleromete" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        else:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    acc_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)
    gyr_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)

data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "label",
    "category",
    "participant",
    "set",
]


sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

data_reampled = data_merged.resample(rule="200ms").apply(sampling).dropna()

data_reampled.info()
