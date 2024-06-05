import copy
import itertools

import numpy as np
import pandas as pd
import scipy

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa

# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from BrainDataConfig import BrainDataConfig
from BrainDataLabel import BrainDataLabel
from Enums import Lobe
from TestTrainingSet import TestTrainingSet


class Brain:

    def __init__(
        self,
        lobe: Lobe = None,
        data_path: str = None,
        current_labels: BrainDataLabel = None,
        mask: str = None,
        NIfTI: str = None,
        voxel_label=None,
        Talairach_MNI_space=None,
    ):
        if lobe is not None:
            self.lobe = lobe
        self.current_labels = current_labels
        if voxel_label is not None:
            self.voxel_label = voxel_label
        if NIfTI is not None:
            self.NIfTI = NIfTI
        if mask is not None:
            self.mask = mask
        if Talairach_MNI_space is not None:
            self.Talairach_MNI_space = Talairach_MNI_space
        if data_path is not None:
            data = scipy.io.loadmat(data_path)
            self.voxels: np.ndarray = data["R"]

            # STG_data = scipy.io.loadmat("./left_STG_MTG_AALlable_ROI.rex.mat")
            # self.STG = tuple(("STG", STG_data["R"]))

            # IFG_data = scipy.io.loadmat("./ROI_aal_wfupick_left44_45.rex.mat")

            # self.IFG_raw = IFG["R"]
            # self.IFG = tuple(("IFG", IFG_data["R"]))

            # Labels
            # self.subject_labels = np.array(["N"] * 4 * 43 + ["D"] * 4 * 33 + ["S"] * 4 * 46)
            # self.subject_labels_o = tuple(("subject_labels",np.array(["N"] * 4 * 43 + ["D"] * 4 * 33 + ["S"] * 4 * 46),))

        bdc = BrainDataConfig()

        self.subject_labels = BrainDataLabel(
            name="subject_labels",
            popmean=bdc.subject_label_popmean,
            labels=np.array(
                [bdc.neurotypical] * bdc.conditions * bdc.neurotypical_patients
                + [bdc.depressive_disorder]
                * bdc.conditions
                * bdc.depressive_disorder_patients
                + [bdc.schizophrenia_spectrum]
                * bdc.conditions
                * bdc.schizophrenia_spectrum_patients
            ),
        )
        # subject_labels 172=N, 132=D, 184=S sumup to 488

        # self.image_labels = np.array(["AR", "AU", "CR", "CU"] * (43 + 33 + 46))

        # self.image_labels_o = tuple(("image_labels", np.array(["AR", "AU", "CR", "CU"] * (43 + 33 + 46))))
        self.image_labels = BrainDataLabel(
            name="image_labels",
            popmean=bdc.image_label_popmean,
            labels=np.array(
                [
                    bdc.abstract_related,
                    bdc.abstract_unrelated,
                    bdc.concrete_related,
                    bdc.concrete_unrelated,
                ]
                * (
                    bdc.neurotypical_patients
                    + bdc.depressive_disorder_patients
                    + bdc.schizophrenia_spectrum_patients
                )
            ),
        )

        bdc = BrainDataConfig()
        self.subject_labels_int = BrainDataLabel(
            name="subject_labels_int",
            popmean=bdc.subject_label_popmean,
            labels=np.array(
                [bdc.neurotypical_int] * bdc.conditions * bdc.neurotypical_patients
                + [bdc.depressive_disorder_int]
                * bdc.conditions
                * bdc.depressive_disorder_patients
                + [bdc.schizophrenia_spectrum_int]
                * bdc.conditions
                * bdc.schizophrenia_spectrum_patients
            ),
        )

        self.image_labels_int = BrainDataLabel(
            name="image_labels_int",
            popmean=bdc.image_label_popmean,
            labels=np.array(
                [
                    bdc.abstract_related_int,
                    bdc.abstract_unrelated_int,
                    bdc.concrete_related_int,
                    bdc.concrete_unrelated_int,
                ]
                * (
                    bdc.neurotypical_patients
                    + bdc.depressive_disorder_patients
                    + bdc.schizophrenia_spectrum_patients
                )
            ),
        )

        match lobe:
            case Lobe.STG:
                self.NIfTI = bdc.nii_STG
                self.Talairach_MNI_space = pd.read_csv(bdc.STG_Talairach_MNI_space)
            case Lobe.IFG:
                self.NIfTI = bdc.nii_IFG
                self.Talairach_MNI_space = pd.read_csv(bdc.IFG_Talairach_MNI_space)
            case Lobe.ALL:
                self.NIfTI = bdc.nii_ALL
                self.Talairach_MNI_space = pd.read_csv(
                    bdc.All_lobes_Talairach_MNI_space
                )
            case _:
                if NIfTI is not None:
                    self.NIfTI = NIfTI
                if Talairach_MNI_space is not None:
                    self.Talairach_MNI_space = Talairach_MNI_space

        """Talairach coordinates, also known as Talairach space, is a 3-dimensional coordinate system (known as an 'atlas') of the human brain"""

        # self.all_labels = [
        # sb + im for sb, im in zip(self.subject_labels, self.image_labels)
        # ]
        # self.all_labels = tuple(("all_labels",np.array([sb + im for sb, im in zip( self.subject_labels.labels,self.image_labels.labels,)]),))

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

    def normalize_data_safely(self, data_set: TestTrainingSet, strategy="mean"):
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
            TestTrainingSet
        """
        if strategy == "n_neighbors":
            imputer = KNNImputer(n_neighbors=2, keep_empty_features=True)
            imputer.fit(data_set.X_train)
            x_train = imputer.transform(data_set.X_train)
            data_set.X_train = x_train
            x_test = imputer.transform(data_set.X_test)
            data_set.X_test = x_test
            return data_set
        elif strategy == "remove-voxels":
            X = np.concatenate((data_set.X_train, data_set.X_test))
            train_len = data_set.X_train.shape[0]
            X_r = X[:, ~np.isnan(X).any(axis=0)]
            data_set.X_train = X_r[0:train_len]
            data_set.X_test = X_r[train_len:]
            return data_set
        elif strategy == "remove-trails":
            return NotImplementedError
        elif strategy == None:
            return data_set
        elif strategy == "mice":
            mice_imput = IterativeImputer(keep_empty_features=True)
            mice_imput.fit(data_set.X_train)
            x_train = mice_imput.transform(data_set.X_train)
            x_test = mice_imput.transform(data_set.X_test)
            data_set.X_train = x_train
            data_set.X_test = x_test
            return data_set
        else:
            imputer = SimpleImputer(
                missing_values=np.nan, strategy=strategy, keep_empty_features=True
            )
            # , keep_empty_features=True
            imputer.fit(data_set.X_train)
            x_train = imputer.transform(data_set.X_train)
            data_set.X_train = x_train
            x_test = imputer.transform(data_set.X_test)
            data_set.X_test = x_test
            return data_set

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

    def unary_fmri_subject_or_image(self, config: BrainDataConfig):
        """
        Binarize the fmri data based on the Image or subject labels
        In the brain object current labels are considered
        """
        brain_data: list[Brain] = []

        comb_src, subject, int_label = self.__get_combination_source(config)

        for label in comb_src:
            voxels = None
            labels = None
            if subject:
                voxels = self.subject_binary_or_unary_data(self.voxels, config, label)
                labels = self.subject_binary_or_unary_data(
                    self.current_labels.labels, config, label
                )
            else:
                trail_pos = self.__get_subject_or_image_label_position(config, label)
                voxels = self.image_based_unary_selection(self.voxels, trail_pos)
                labels = self.image_based_unary_selection(
                    self.current_labels.labels, trail_pos
                )
                # voxels = self.image_unary_data(self.voxels, config, label)
                # labels = self.image_unary_data(self.current_labels.labels, config, label)
            brain = copy.deepcopy(self)
            brain.voxels = voxels
            brain.voxel_label = label

            unary_label = BrainDataLabel(
                name=f"{self.current_labels.name}_unary_{label}",
                popmean=config.unary_popmean,
                labels=labels,
            )
            brain.current_labels = unary_label

            brain_data.append(brain)

        return brain_data

    def validate_data_dim(self, label):
        if self.current_labels.labels.shape != label.shape:
            raise Exception("Data dimension mismatch")

    def binarize_fmri_image_or_subject(self, config: BrainDataConfig):
        """
        Binarize the fmri data based on the Image or subject labels
        In the brain object current labels are considered
        """
        brain_data: list[Brain] = []

        comb_src, subject, int_label = self.__get_combination_source(config)

        combinations = list(itertools.combinations(comb_src, 2))

        for combination in combinations:
            if subject:
                # self.modify_data_config(config, combination)
                voxels = self.subject_binary_or_unary_data(
                    self.voxels, config, combination
                )
                labels = self.subject_binary_or_unary_data(
                    self.current_labels.labels, config, combination
                )
            else:
                trail_pos1 = self.__get_subject_or_image_label_position(
                    config, combination[0]
                )
                trail_pos2 = self.__get_subject_or_image_label_position(
                    config, combination[1]
                )

                voxels = self.image_based_binary_selection(
                    self.voxels, trail_pos1, trail_pos2
                )
                labels = self.image_based_binary_selection(
                    self.current_labels.labels, trail_pos1, trail_pos2
                )

            brain = Brain(lobe=self.lobe)
            brain.voxel_label = combination
            brain.voxels = voxels
            binary_label = BrainDataLabel(
                name=f"{self.current_labels.name}_binary_{combination[0]}-{combination[1]}",
                popmean=config.binary_popmean,
                labels=labels,
            )
            brain.current_labels = binary_label
            brain_data.append(brain)

        return brain_data

    def __get_combination_source(self, config: BrainDataConfig):
        int_label = False
        subject = False
        if "subject" in self.current_labels.type and "int" in self.current_labels.type:
            comb_src = config.subject_labels_int
            subject = True
            int_label = True
        elif (
            "subject" in self.current_labels.type
            and not "int" in self.current_labels.type
        ):
            comb_src = config.subject_labels
            subject = True
        elif "image" in self.current_labels.type and "int" in self.current_labels.type:
            comb_src = config.image_labels_int
            int_label = True
        elif (
            "image" in self.current_labels.type
            and not "int" in self.current_labels.type
        ):
            comb_src = config.image_labels
        else:
            raise Exception("Label matching went wrong")
        return comb_src, subject, int_label

    def binary_subject_image_based_data(self):
        """
        Not use it, use instead: binarize_fmri_image_or_subject
        """
        config = BrainDataConfig()
        brain_data: list[Brain] = []

        combinations = list(itertools.combinations(config.subject_labels, 2))

        for combination in combinations:
            voxels = self.subject_binary_or_unary_data(self.voxels, config, combination)
            labels = self.subject_binary_or_unary_data(
                self.current_labels.labels, config, combination
            )

            brain = Brain(lobe=self.lobe)
            brain.voxels = voxels
            name = (
                f"{self.current_labels.name}-Binary_{combination[0]}-{combination[1]}"
            )

            brain.current_labels = BrainDataLabel(name, config.binary_popmean, labels)

            brain_data.append(brain)

        return brain_data

    def binary_fmri_image_trails(self):
        """
        deprecated use instead: binarize_fmri_image_or_subject
        """
        config = BrainDataConfig()
        brain_data: list[Brain] = []

        combinations = list(itertools.combinations(config.image_labels, 2))

        for combination in combinations:
            trail_pos1 = self.__get_subject_or_image_label_position(
                config, combination[0]
            )
            trail_pos2 = self.__get_subject_or_image_label_position(
                config, combination[1]
            )

            voxels = self.image_based_binary_selection(
                self.voxels, trail_pos1, trail_pos2
            )
            labels = self.image_based_binary_selection(
                self.current_labels.labels, trail_pos1, trail_pos2
            )

            brain = Brain(lobe=self.lobe)
            brain.voxels = voxels
            name = f"{self.current_labels.name}-{combination[0]}_{combination[1]}"

            if combination[1] == "ARCU":
                name = f"{self.current_labels.name}-{combination[0]}"

            popmean = self.current_labels.popmean
            brain.current_labels = BrainDataLabel(name, popmean, labels)

            brain_data.append(brain)
        # this should return the listof tuples of brain and the config
        return brain_data

    def concatenate_fmri_image_trails(self):
        config = BrainDataConfig()
        brain_data: list[Brain] = []

        combinations = list(itertools.combinations(config.image_labels, 2))
        combinations.append(("ARCU", "ARCU"))
        # combinations.insert(0, ("ARCU", "ARCU"))

        for combination in combinations:
            trail_pos1 = self.__get_subject_or_image_label_position(
                config, combination[0]
            )
            trail_pos2 = self.__get_subject_or_image_label_position(
                config, combination[1]
            )

            voxels = self.__image_based_concatenation(
                self.voxels, trail_pos1, trail_pos2
            )
            labels = self.__image_based_concatenation(
                self.current_labels.labels, trail_pos1, trail_pos2
            )

            brain = Brain(lobe=self.lobe)
            brain.voxels = voxels
            brain.voxel_label = combination
            name = f"{self.current_labels.name}-{combination[0]}_{combination[1]}"

            if combination[1] == "ARCU":
                name = f"{self.current_labels.name}-{combination[0]}"

            popmean = self.current_labels.popmean
            brain.current_labels = BrainDataLabel(name, popmean, labels)

            brain_data.append(brain)
        # this should return the listof tuples of brain and the config
        return brain_data

    def image_based_unary_selection(self, data: np.ndarray, trail_pos: int):
        selection = []
        i = 0
        while i < data.shape[0]:
            index = i + trail_pos
            print("index", index)
            selection.append(data[index])
            i = i + 4
        return np.array(selection)

    def image_based_binary_selection(
        self, data: np.ndarray, trail_pos1: int, trail_pos2: int
    ):
        selection = []
        i = 0
        while i < data.shape[0]:
            index1 = i + trail_pos1
            print("index1", index1)
            index2 = i + trail_pos2
            print("index2", index2)
            if data.ndim == 1:
                if len(data[index1]) == 1 and data[index1] != data[index2]:
                    raise Exception(
                        "These label must be same, Data label calculation is wrong"
                    )
            selection.append(data[index1])
            selection.append(data[index2])
            i = i + 4
        return np.array(selection)

    def __image_based_concatenation(
        self, data: np.ndarray, trail_pos1: int, trail_pos2: int
    ):
        concatenated = []
        if trail_pos1 is None and trail_pos2 is None:
            trail_pos1 = 0
            trail_pos2 = 1
            trail_pos3 = 2
            trail_pos4 = 3
            i = 0
            while i < data.shape[0]:
                index1 = i + trail_pos1
                print("index1", index1)
                index2 = i + trail_pos2
                print("index2", index2)
                index3 = i + trail_pos3
                print("index3", index3)
                index4 = i + trail_pos4
                print("index4", index4)
                if data.ndim == 1:
                    if data[index1] != data[index2] != data[index3] != data[index4]:
                        concatenated.append(
                            f"{data[index1]}_{data[index2]}_{data[index3]}_{data[index4]}_"
                        )
                    else:
                        concatenated.append(data[index1])
                else:
                    chunk = np.concatenate(
                        (data[index1], data[index2], data[index3], data[index4])
                    )
                    concatenated.append(chunk)
                i = i + 4
        else:
            i = 0
            while i < data.shape[0]:
                index1 = i + trail_pos1
                print("index1", index1)
                index2 = i + trail_pos2
                print("index2", index2)
                if data.ndim == 1:
                    if data[index1] != data[index2]:
                        concatenated.append(f"{data[index1]}_{data[index2]}")
                    else:
                        concatenated.append(data[index1])
                else:
                    chunk = np.concatenate((data[index1], data[index2]))
                    concatenated.append(chunk)
                i = i + 4

        return np.array(concatenated)

    def __get_subject_or_image_label_position(self, config: BrainDataConfig, label):
        position = np.NAN
        match label:
            case config.abstract_related:
                position = config.AR_position
            case config.abstract_unrelated:
                position = config.AU_position
            case config.concrete_related:
                position = config.CR_position
            case config.concrete_unrelated:
                position = config.CU_position
            case "ARCU":
                position = None
            case config.neurotypical:
                position = config.neurotypical_int
            case config.depressive_disorder:
                position = config.depressive_disorder_int
            case config.schizophrenia_spectrum:
                position = config.schizophrenia_spectrum_int
        self.__validate_voxel_trail_position(config, label, position)
        return position

    def __validate_voxel_trail_position(
        self, config: BrainDataConfig, combination, position
    ):
        match combination:
            case config.abstract_related | config.neurotypical:
                if position != 0:
                    raise Exception(
                        "Voxel trail positions are not calculated coorectly"
                    )
            case config.abstract_unrelated | config.depressive_disorder:
                if position != 1:
                    raise Exception(
                        "Voxel trail positions are not calculated coorectly"
                    )
            case config.concrete_related | config.schizophrenia_spectrum:
                if position != 2:
                    raise Exception(
                        "Voxel trail positions are not calculated coorectly"
                    )
            case config.concrete_unrelated:
                if position != 3:
                    raise Exception(
                        "Voxel trail positions are not calculated coorectly"
                    )
            case "ARCU":
                if position != None:
                    raise Exception(
                        "Voxel trail positions are not calculated coorectly"
                    )

    def subject_binary_or_unary_data(self, data, config: BrainDataConfig, combination):
        """
        Binary or unary data will be returned depends upon the combination provided
        """
        chunks = []
        patients = config.patients
        trails = config.conditions
        for label in combination:
            match label:
                case config.neurotypical_int | config.neurotypical:
                    end = patients[config.neurotypical_int] * trails
                    v = data[0:end]
                    chunks.append(v)
                case config.depressive_disorder_int | config.depressive_disorder:
                    start = patients[config.neurotypical_int] * trails
                    end = patients[config.depressive_disorder_int] * trails
                    v = data[start : (start + end)]
                    chunks.append(v)
                case config.schizophrenia_spectrum_int | config.schizophrenia_spectrum:
                    start = patients[config.neurotypical_int] * trails
                    end = patients[config.depressive_disorder_int] * trails
                    v = data[(start + end) :]
                    chunks.append(v)
                case _:
                    raise Exception("Unsupported label")

        return np.concatenate(chunks)

    def brain_subset(
        self,
        size: int,
        config: BrainDataConfig,
    ):
        if size > self.voxels.shape[0]:
            raise IndexError(
                "Cardinality of the subset cannot be greater then the set itself"
            )

        subset_X = self.extract_subset(self.voxels, size, config)
        self.voxels = subset_X
        subset_labels = self.extract_subset(self.current_labels.labels, size, config)
        self.current_labels.labels = subset_labels
        return self

    def extract_subset(self, data: np.ndarray, size: int, config: BrainDataConfig):
        subset = None
        chunks = []
        start = 0
        for patient in config.patients:
            subset_per_patient = patient * config.conditions
            v = data[start : (start + size)]
            chunks.append(v)
            start = start + subset_per_patient
        subset = np.concatenate(chunks)
        return subset

    def voxels_labels_subset(
        self,
        voxels: np.ndarray,
        size: int,
        config: BrainDataConfig,
        label: BrainDataLabel,
    ):
        if size > voxels.shape[0]:
            raise IndexError(
                "Cardinality of the subset cannot be greater then the set itself"
            )
        subset_X = self.extract_subset(voxels, size, config)
        subset_labels = self.extract_subset(label.labels, size, config)
        label.labels = subset_labels
        return subset_X, label

    def extract_subset(self, data: np.ndarray, size: int, config: BrainDataConfig):
        subset = None
        chunks = []
        start = 0
        for patient in config.patients:
            subset_per_patient = patient * config.conditions
            v = data[start : (start + size)]
            chunks.append(v)
            start = start + subset_per_patient
        subset = np.concatenate(chunks)
        return subset

    def __repr__(self) -> str:
        return (
            f"Lobe:{self.lobe.name}, Voxels:{self.voxels.shape}, {self.current_labels}"
        )


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
