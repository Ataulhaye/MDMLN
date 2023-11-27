import numpy as np
import pandas as pd
import scipy
from sklearn import datasets
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from BrainDataConfig import BrainDataConfig

from BrainDataLabel import BrainDataLabel


class BrainData:
    def __init__(self, load_data=False):
        if load_data is True:
            STG_data = scipy.io.loadmat("./left_STG_MTG_AALlable_ROI.rex.mat")
            # self.STG_raw = STG["R"]
            self.STG = tuple(("STG", STG_data["R"]))

            IFG_data = scipy.io.loadmat("./ROI_aal_wfupick_left44_45.rex.mat")

            # self.IFG_raw = IFG["R"]
            self.IFG = tuple(("IFG", IFG_data["R"]))

            # Labels
            # self.subject_labels = np.array(["N"] * 4 * 43 + ["D"] * 4 * 33 + ["S"] * 4 * 46)
            # self.subject_labels_o = tuple(("subject_labels",np.array(["N"] * 4 * 43 + ["D"] * 4 * 33 + ["S"] * 4 * 46),))

            config = BrainDataConfig()

            self.subject_labels = BrainDataLabel(
                name="subject_labels",
                popmean=config.subject_label_popmean,
                labels=np.array(
                    [config.neurotypical]
                    * config.conditions
                    * config.neurotypical_patients
                    + [config.depressive_disorder]
                    * config.conditions
                    * config.depressive_disorder_patients
                    + [config.schizophrenia_spectrum]
                    * config.conditions
                    * config.schizophrenia_spectrum_patients
                ),
            )
            # subject_labels 172=N, 132=D, 184=S sumup to 488

            # self.image_labels = np.array(["AR", "AU", "CR", "CU"] * (43 + 33 + 46))

            # self.image_labels_o = tuple(("image_labels", np.array(["AR", "AU", "CR", "CU"] * (43 + 33 + 46))))
            self.image_labels = BrainDataLabel(
                name="image_labels",
                popmean=config.image_label_popmean,
                labels=np.array(
                    [
                        config.abstract_related,
                        config.abstract_unrelated,
                        config.concrete_related,
                        config.concrete_unrelated,
                    ]
                    * (
                        config.neurotypical_patients
                        + config.depressive_disorder_patients
                        + config.schizophrenia_spectrum_patients
                    )
                ),
            )
            # self.all_labels = [
            # sb + im for sb, im in zip(self.subject_labels, self.image_labels)
            # ]
            self.all_labels = tuple(
                (
                    "all_labels",
                    np.array(
                        [
                            sb + im
                            for sb, im in zip(
                                # self.subject_labels[1],
                                # self.image_labels[1],
                                self.subject_labels.labels,
                                self.image_labels.labels,
                            )
                        ]
                    ),
                )
            )

    def normalize_data(self, data, strategy="mean"):
        """_summary_

        Args:
        data (_type_): numpy.ndarray
        strategy (str, optional):The imputation strategy. Defaults to "mean".
        If "mean", then replace missing values using the mean along each column. Can only be used with numeric data.
        If "median", then replace missing values using the median along each column. Can only be used with numeric data.
        If "most_frequent", then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.
        If "constant", then replace missing values with fill_value. Can be used with strings or numeric data.
        if "remove-trails", then all trails contains nans will be removed
        if "remove-voxels", will remove all (Columns) in which nans are present
        if "n_neighbors", nearest neighbour approach, default is 2 neighbours

        Returns:
            numpy.ndarray
        """
        if strategy == "n_neighbors":
            imputer = KNNImputer(n_neighbors=2)
            return imputer.fit_transform(data)
        elif strategy == "remove-voxels":
            return data[:, ~np.isnan(data).any(axis=0)]
        elif strategy == "remove-trails":
            return NotImplementedError
        elif strategy == None:
            return data
        else:
            imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
            imputer.fit(data)
            return imputer.transform(data)

    def calculate_nans_trail_wise(self, data):
        # lis = [sum(np.isnan(x)) for x in zip(*data)]
        nans_len_list = []
        for row in data:
            nans_length = 0
            for i in row:
                if np.isnan(i):
                    nans_length += 1
            nans_len_list.append(nans_length)
        return nans_len_list

    def calculate_nan_positions(self, data):
        rows = []
        for row in data:
            nan_row = []
            for i in row:
                nan = 0
                if np.isnan(i):
                    nan = 1
                nan_row.append(nan)
            rows.append(nan_row)
        return np.array(rows)

    def calculate_nans_voxel_wise(self, data):
        """It is column wise if we consider as a data matrix

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        nans_len_list = []
        r, c = data.shape
        for column_index in range(c):
            nans_length = 0
            for row_index in range(r):
                if np.isnan(data[row_index][column_index]):
                    nans_length += 1
            nans_len_list.append(nans_length)

        return nans_len_list


# Ata:
# Test different classifiers with the MTG and IFG
# Try to classify the subjects labes and image labels
# We predict IFG can classify both, MTG only IMG label
# Tuthera:
# Download and familiarize yourself with Net2Brain and RSA
# claculate RDMs for each Subject + image condition pair and each ROI
# Calculate correlation (do RSA) using the RDMs

# Tuhera: Unraveling The Computational Basis Of Mental Disorders Using Task Specific Artificial Nueral NEtworks
# Ata: Using Machine Learning to Diagnose Mental Disorders from Neuroimaging Data

# df = pd.DataFrame()

# left IFG (BA44/45; 523 voxels) and the left STG/MTG (7238 voxels) as both are relevant
# for the task but the left IFG should differentiate between groups (N,D,S)
# 43 N 33 D 46 S x 4 images [AR, AU, CR, CU]
# print((43 + 33 + 46) * 4)


# what are exactly AR, AU, CR and CU
# 172, 132 and 184
# nans are errors or?
# hardcoded means it known for that data

# STG = np.array(
#    rows,
# )

"""
# Labels
subject_labels = ["N"] * 4 * 43 + ["D"] * 4 * 33 + ["S"] * 4 * 46
# subject_labels 172=N, 132=D, 184=S sumup to 488
image_labels = ["AR", "AU", "CR", "CU"] * (43 + 33 + 46)


allL = [sb + im for sb, im in zip(subject_labels, image_labels)]

# print(set(allL))

STG = scipy.io.loadmat("./left_STG_MTG_AALlable_ROI.rex.mat")

STG["R"].shape  # (488, 7238)

STG_raw = STG["R"]

IFG = scipy.io.loadmat("./ROI_aal_wfupick_left44_45.rex.mat")

IFG["R"].shape

IFG_raw = IFG["R"]

"""
"""
rows = []
for row in STG_raw:
print(len(row))
row = [x for x in row if ~np.isnan(x)]
rows.append(row)
print(len(row))
print("----------------")

def plot_with_0():
    rows = []
    for row in STG_raw:
        # print(len(row))
        # row = [x for x in row if ~np.isnan(x)]
        row = [0 if np.isnan(x) else x for x in row]
        rows.append(row)
        # print(len(row))
        # print("----------------")

    fig = go.Figure()
    i = 0
    for row in rows:
        fig.add_trace(
            go.Scatter(x=x, y=row, name=f"{subject_labels[i]}{image_labels[i]}")
        )
        i += 1
    fig.update_traces(marker=dict(color="red"))
    fig.show()


def plot_with_nan():
    fig = go.Figure()
    i = 0
    for row in STG_raw:
        py_list = list(row)
        fig.add_trace(
            go.Scatter(x=x, y=py_list, name=f"{subject_labels[i]}{image_labels[i]}")
        )
        i += 1
    fig.show()

"""
