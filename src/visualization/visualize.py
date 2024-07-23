import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


# mlp.rcParams["figure.figsize"] = (20, 5)
mlp.rcParams["figure.dpi"] = 100

# labels = df["label"].unique()
# participants = df["participant"].unique()

# for label in labels:
#     for participant in participants:
#         all_axis_df = (
#             df.query(f"label == '{label}'")
#             .query(f"participant == '{participant}'")
#             .reset_index()
#         )

#         fig, ax = plt.subplots()
#         all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
#         ax.set_ylabel("acc_y")
#         ax.set_xlabel("samples")
#         plt.title(f"{label} ({participant})".title())
#         plt.legend()


# def plot_data(df, x_col, y_col, z_col):
#     labels = df["label"].unique()
#     participants = df["participant"].unique()

#     for label in labels:
#         for participant in participants:
#             all_axis_df = df.query(
#                 f"label == '{label}' and participant == '{participant}'"
#             ).reset_index()

#             if not all_axis_df.empty:
#                 fig, ax = plt.subplots()
#                 all_axis_df[[x_col, y_col, z_col]].plot(ax=ax)
#                 ax.set_ylabel(y_col)
#                 ax.set_xlabel("samples")
#                 plt.title(f"{label} ({participant})".title())
#                 plt.legend()
#                 plt.show()


# plot_data(df, "acc_x", "acc_y", "acc_z")
# plot_data(df, "gyr_x", "gyr_y", "gyr_z")


# label = "row"
# participant = "A"
# combine_plot_df = df.query(
#     f"label == '{label}' and participant == '{participant}'"
# ).reset_index(drop=True)


# fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
# combine_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
# combine_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

# ax[0].legend(
#     loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
# )
# ax[1].legend(
#     loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
# )
# ax[1].set_xlabel("samples")


# labels = df["label"].unique()
# participants = df["participant"].unique()

# for label in labels:
#     for participant in participants:
#         combine_plot_df = df.query(
#             f"label == '{label}' and participant == '{participant}'"
#         ).reset_index()

#         if not combine_plot_df.empty:
#             fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
#             combine_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
#             combine_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

#             ax[0].legend(
#                 loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
#             )
#             ax[1].legend(
#                 loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
#             )
#             ax[1].set_xlabel("samples")

#             # plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
#             plt.show()


def plot_combined_data(df, acc_cols, gyr_cols, figsize=(20, 10)):
    """
    Plots combined accelerometer and gyroscope data for each label and participant.

    Parameters:
    df: The input DataFrame containing the data.
    acc_cols (list of str): List of column names for accelerometer data.
    gyr_cols (list of str): List of column names for gyroscope data.
    figsize: Figure size for the plots.
    """

    labels = df["label"].unique()
    participants = df["participant"].unique()

    for label in labels:
        for participant in participants:
            combine_plot_df = df.query(
                f"label == '{label}' and participant == '{participant}'"
            ).reset_index(drop=True)

            if not combine_plot_df.empty:
                fig, ax = plt.subplots(nrows=2, figsize=figsize)
                combine_plot_df[acc_cols].plot(ax=ax[0])
                combine_plot_df[gyr_cols].plot(ax=ax[1])

                ax[0].legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=3,
                    fancybox=True,
                    shadow=True,
                )
                ax[1].legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.15),
                    ncol=3,
                    fancybox=True,
                    shadow=True,
                )
                ax[1].set_xlabel("samples")

                plt.savefig(
                    f"../../reports/figures/{label.title()} ({participant}).png"
                )
                plt.show()


# Example call to the function
plot_combined_data(df, ["acc_x", "acc_y", "acc_z"], ["gyr_x", "gyr_y", "gyr_z"])
