import copy
import pickle
import statistics

import matplotlib.pyplot as plt
import nilearn as ni
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import torch
from nilearn import image, plotting
from scipy.spatial import KDTree
from scipy.stats import spearmanr

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from DataTraining import DataTraining
from Enums import Lobe
from ExportData import ExportData
from RepresentationalSimilarityAnalysis import RepresentationalSimilarityAnalysis
from RSAConfig import RSAConfig
from TrainingConfig import TrainingConfig


class FMRIAnalyser:

    def __init__(
        self,
        lobe: Lobe,
        brain: Brain = None,
        classifiers=None,
        strategies=None,
        training_config: TrainingConfig = None,
        data_config: BrainDataConfig = None,
        Talairach_MNI_space: pd.DataFrame = None,
    ):
        if lobe is None:
            raise Exception("Lobe of the brain must be defined.")
        else:
            self.lobe = lobe

        if classifiers is None:
            self.classifiers = ["SVM", "MLP", "LinearDiscriminant", "LGBM"]
        else:
            self.classifiers = classifiers

        if strategies is None:
            self.strategies = ["mean", "remove-voxels", "median"]
        else:
            self.strategies = strategies

        if training_config is None:
            self.training_config = TrainingConfig()
        else:
            self.training_config = training_config

        if data_config is None:
            self.data_config = BrainDataConfig()
        else:
            self.data_config = data_config

        if lobe is Lobe.STG:
            self.brain = Brain(
                area=self.data_config.STG,
                data_path=self.data_config.STG_path,
            )
            self.Talairach_MNI_space = self.brain.STG_Talairach_MNI_space
        elif lobe is Lobe.IFG:
            self.brain = Brain(
                area=self.data_config.IFG, data_path=self.data_config.IFG_path
            )
            self.Talairach_MNI_space = self.brain.IFG_Talairach_MNI_space
        elif lobe is Lobe.ALL:
            self.brain = Brain(
                area=self.data_config.ALL,
                data_path=self.data_config.all_lobes_path,
            )
            self.Talairach_MNI_space = self.brain.All_lobes_Talairach_MNI_space
        else:
            self.brain = brain
            self.Talairach_MNI_space = Talairach_MNI_space

    def binary_subject_classification(self):
        """
        Binarize the fMRI data based on subjects, image labels remains same.
        """

        self.brain.current_labels = self.brain.subject_labels
        stg_subject_binary_data = self.brain.binarize_fmri_image_or_subject(
            self.data_config
        )

        all_export_data = []

        for bd in stg_subject_binary_data:
            training = DataTraining()
            self.training_config.analyze_binary_trails = True
            self.training_config.analyze_concatenated_trails = False
            print("Patients", self.data_config.patients)
            self.__modify_patients(self.data_config, bd.voxel_label)
            print("Patients changed", self.data_config.patients)
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
                self.data_config,
            )
            all_export_data.extend(export_data)
        self.training_config.brain_area = self.brain.area
        export = ExportData()

        print_config = self.__get_note(
            stg_subject_binary_data[0].current_labels.popmean,
            self.binary_subject_classification,
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def binary_image_classification(self):
        """
        Binarize the fMRI data based on images. i.e AR, AU
        """

        self.brain.current_labels = self.brain.image_labels
        stg_image_binary_data = self.brain.binarize_fmri_image_or_subject(
            self.data_config
        )

        all_export_data = []

        for bd in stg_image_binary_data:
            training = DataTraining()
            self.training_config.analyze_binary_trails = True
            self.training_config.analyze_concatenated_trails = False
            self.data_config.conditions = 2
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
                self.data_config,
            )
            all_export_data.extend(export_data)
        self.training_config.brain_area = self.brain.area
        export = ExportData()

        print_config = self.__get_note(
            stg_image_binary_data[0].current_labels.popmean,
            self.binary_image_classification,
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def unary_subject_binary_image_classification(self):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance image wise binarization takes place
        i.e unary_subject_labels_N_binary_AR-AU
        """
        all_export_data = []

        self.brain.current_labels = self.brain.subject_labels
        stg_subject_unary_data = self.brain.unary_fmri_subject_or_image(
            self.data_config
        )
        mean = None
        for un_brain in stg_subject_unary_data:
            slice_to = un_brain.current_labels.labels.shape[0]
            un_brain.current_labels.labels = un_brain.image_labels.labels[0:slice_to]
            un_brain.current_labels.type = "image"
            bin_brain = un_brain.binarize_fmri_image_or_subject(self.data_config)
            for bd in bin_brain:
                training = DataTraining()
                self.data_config.conditions = 2
                self.training_config.analyze_binary_trails = True
                self.training_config.analyze_concatenated_trails = False
                print("Patients", self.data_config.patients)
                self.__modify_patients(self.data_config, un_brain.voxel_label)
                print("Patients changed", self.data_config.patients)
                mean = bd.current_labels.popmean
                export_data = training.brain_data_classification(
                    bd,
                    self.training_config,
                    self.strategies,
                    self.classifiers,
                    self.data_config,
                )
                all_export_data.extend(export_data)
        self.training_config.brain_area = self.brain.area
        export = ExportData()
        print_config = self.__get_note(
            mean, self.unary_subject_binary_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
            single_label=True,
        )
        # self.plot_images(all_export_data)
        self.plot_detailed_bars(all_export_data)
        # this will groupby the mean, median,...

        print("End")

    def binary_subject_concatenated_image_classification(self):
        """
        Binarize the fMRI data based on subjects, then for every binarized instance image wise binarization with concatenation takes place
        """
        self.brain.current_labels = self.brain.subject_labels

        all_data = []

        stg_subject_binary_data = self.brain.binarize_fmri_image_or_subject(
            self.data_config
        )

        self.training_config.analyze_concatenated_trails = True
        self.training_config.analyze_binary_trails = False
        mean = None
        for mod_brain in stg_subject_binary_data:
            binary_data = mod_brain.concatenate_fmri_image_trails()
            for bd in binary_data:
                mean = bd.current_labels.popmean
                training = DataTraining()
                self.data_config.conditions = 1
                print("Patients", self.data_config.patients)
                self.__modify_patients(self.data_config, mod_brain.voxel_label)
                print("Patients changed", self.data_config.patients)
                export_data = training.brain_data_classification(
                    bd,
                    self.training_config,
                    self.strategies,
                    self.classifiers,
                    self.data_config,
                )
                all_data.extend(export_data)

        self.training_config.brain_area = self.brain.area
        export = ExportData()
        print_config = self.__get_note(
            mean, self.binary_subject_concatenated_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def binary_subject_binary_image_classification(self):
        """
        Binarize the fMRI data based on subjects, then for every binarized instance image wise binary selection takes place
        """
        self.brain.current_labels = self.brain.subject_labels

        all_data = []

        stg_subject_binary_data = self.brain.binarize_fmri_image_or_subject(
            self.data_config
        )

        self.training_config.analyze_concatenated_trails = False
        self.training_config.analyze_binary_trails = True
        mean = None
        for mod_brain in stg_subject_binary_data:
            mod_brain.current_labels.type = "image"
            binary_data = mod_brain.binarize_fmri_image_or_subject(self.data_config)
            for bd in binary_data:
                mean = bd.current_labels.popmean
                training = DataTraining()
                self.data_config.conditions = 2
                print("Patients", self.data_config.patients)
                self.__modify_patients(self.data_config, mod_brain.voxel_label)
                print("Patients changed", self.data_config.patients)
                export_data = training.brain_data_classification(
                    bd,
                    self.training_config,
                    self.strategies,
                    self.classifiers,
                    self.data_config,
                )
                all_data.extend(export_data)

        self.training_config.brain_area = self.brain.area
        export = ExportData()
        print_config = self.__get_note(
            mean, self.binary_subject_binary_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def binary_subject_unary_image_classification(self):
        """
        Binarize the fMRI data based on subjects, then for every binarized instance image unary selection takes place
        """
        self.brain.current_labels = self.brain.subject_labels

        all_data = []

        stg_subject_binary_data = self.brain.binarize_fmri_image_or_subject(
            self.data_config
        )

        self.training_config.analyze_binary_trails = True
        self.training_config.analyze_concatenated_trails = False

        mean = None

        for mod_brain in stg_subject_binary_data:
            slice_to = mod_brain.current_labels.labels.shape[0]
            predecessor = copy.deepcopy(mod_brain.current_labels)
            mod_brain.current_labels.labels = self.brain.image_labels.labels[0:slice_to]
            mod_brain.current_labels.type = "image"
            binary_data = mod_brain.unary_fmri_subject_or_image(self.data_config)
            for bd in binary_data:
                bd.current_labels.popmean = predecessor.popmean
                bd.current_labels.labels = bd.image_based_unary_selection(
                    predecessor.labels, 0
                )
                mean = bd.current_labels.popmean
                training = DataTraining()
                self.data_config.conditions = 1
                print("Patients", self.data_config.patients)
                self.__modify_patients(self.data_config, mod_brain.voxel_label)
                print("Patients changed", self.data_config.patients)
                export_data = training.brain_data_classification(
                    bd,
                    self.training_config,
                    self.strategies,
                    self.classifiers,
                    self.data_config,
                )
                all_data.extend(export_data)

        self.training_config.brain_area = self.brain.area
        export = ExportData()
        print_config = self.__get_note(
            mean, self.binary_subject_unary_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def subject_concatenated_image_classification(self):
        """Concate the (two out of four) image labels. i.e AR and AC from four images. AR and AC are concatenated.
        The subject labels remain same. Popmean would be 0.3
        """

        self.brain.current_labels = self.brain.image_labels

        stg_image_binary_data = self.brain.concatenate_fmri_image_trails()

        self.training_config.analyze_concatenated_trails = True
        self.training_config.analyze_binary_trails = False

        all_data = []
        mean = None
        for bd in stg_image_binary_data:
            training = DataTraining()
            self.data_config.conditions = 1
            bd.current_labels.popmean = self.data_config.subject_label_popmean
            # a = bd.image_based_unary_selection(self.brain.subject_labels.labels, 0)
            bd.current_labels.labels = bd.image_based_unary_selection(
                self.brain.subject_labels.labels, 0
            )
            mean = bd.current_labels.popmean
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
                self.data_config,
            )
            all_data.extend(export_data)
        self.training_config.brain_area = self.brain.area
        export = ExportData()
        print_config = self.__get_note(
            mean, self.subject_concatenated_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def subject_binary_image_classification(self):
        """
        Select two (two out of four) image labels. i.e AR and AC from four images. AR and AC are for one subject i.e Neurotypical.
        The subject labels remain same. Popmean would be 0.3
        """
        self.brain.current_labels = self.brain.image_labels

        stg_image_binary_data = self.brain.binarize_fmri_image_or_subject(
            self.data_config
        )

        self.training_config.analyze_binary_trails = True
        self.training_config.analyze_concatenated_trails = False

        all_data = []
        mean = None

        for bd in stg_image_binary_data:
            training = DataTraining()
            self.data_config.conditions = 2
            bd.current_labels.popmean = self.data_config.subject_label_popmean
            bd.current_labels.labels = bd.image_based_binary_selection(
                self.brain.subject_labels.labels, 0, 1
            )
            mean = bd.current_labels.popmean
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
                self.data_config,
            )
            all_data.extend(export_data)

        self.training_config.brain_area = self.brain.area
        export = ExportData()
        print_config = self.__get_note(mean, self.subject_binary_image_classification)
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def binary_image_and_binary_subject_classification_with_shaply(self):

        split = "r_split"
        if self.training_config.predefined_split:
            split = "cr_split"

        self.brain.current_labels = self.brain.subject_labels_int
        stg_subject_binary_data = self.brain.binarize_fmri_image_or_subject(
            self.data_config
        )

        self.training_config.dimension_reduction = True
        self.training_config.explain = True
        self.training_config.folds = 1
        self.training_config.predefined_split = False

        for bd in stg_subject_binary_data:
            training = DataTraining()
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
                self.data_config,
            )
            self.training_config.brain_area = self.brain.area
            export = ExportData()
            note = export.create_note([self.training_config])
            export.create_and_write_datasheet(
                data=export_data,
                sheet_name=f"{self.brain.area}-Results",
                title=f"{self.brain.area}-{self.training_config.folds}-Folds-{split}-Clf",
                notes=note,
                transpose=True,
            )
        self.brain.current_labels = self.brain.image_labels_int
        stg_image_binary_data = self.brain.binarize_fmri_image_or_subject(
            self.data_config
        )

        for bd in stg_image_binary_data:
            training = DataTraining()
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
                self.data_config,
            )
            self.training_config.brain_area = self.brain.area
            export = ExportData()
            note = export.create_note([self.training_config])
            export.create_and_write_datasheet(
                data=export_data,
                sheet_name=f"{self.brain.area}-Results",
                title=f"{self.brain.area}-{self.training_config.folds}-Folds-{split}-Clf",
                notes=note,
                transpose=True,
            )

    def subject_and_image_classification(self):
        """
        Basic classification with image and subject labels
        """
        self.brain.current_labels = self.brain.subject_labels

        training = DataTraining()

        export_data = training.brain_data_classification(
            self.brain,
            self.training_config,
            self.strategies,
            self.classifiers,
            self.data_config,
        )

        self.brain.current_labels = self.brain.image_labels
        e_data = training.brain_data_classification(
            self.brain,
            self.training_config,
            self.strategies,
            self.classifiers,
            self.data_config,
        )
        export_data.extend(e_data)

        self.training_config.brain_area = self.brain.area
        export = ExportData()
        note = export.create_note([self.training_config, ""])
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
        )

    def RSA_Audio_RDM(self):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """
        self.brain.current_labels = self.brain.subject_labels
        subject_unary_data = self.brain.unary_fmri_subject_or_image(self.data_config)

        rsa_config = RSAConfig()
        rsa = RepresentationalSimilarityAnalysis()

        r_means = rsa.RSA(
            subject_unary_data,
            rsa_config.audio_RDM,
            rsa_config.radius,
            self.Talairach_MNI_space,
        )

        k_max, max_from_pairs = rsa.generate_RSA_results(r_means)

        file_name = f"{self.brain.area}_Radius-{rsa_config.radius}_AudioRDM_RSA.pickle"
        with open(file_name, "wb") as output:
            pickle.dump((k_max, max_from_pairs), output)

        return k_max, max_from_pairs

    def RSA_related_unrelated_RDM(self):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """
        self.brain.current_labels = self.brain.subject_labels
        subject_unary_data = self.brain.unary_fmri_subject_or_image(self.data_config)

        rsa_config = RSAConfig()
        rsa = RepresentationalSimilarityAnalysis()

        r_means = rsa.RSA(
            subject_unary_data,
            rsa_config.related_unrelated_RDM,
            rsa_config.radius,
            self.Talairach_MNI_space,
        )

        k_max, max_from_pairs = rsa.generate_RSA_results(r_means)

        file_name = f"{self.brain.area}_Radius-{rsa_config.radius}_related_unrelated_RDM_RSA.pickle"
        with open(file_name, "wb") as output:
            pickle.dump((k_max, max_from_pairs), output)

        return k_max, max_from_pairs

    def plot_detailed_bars(self, all_export_data):
        nested_dict = self.groupby_strategy(all_export_data)

        N = self.data_config.neurotypical
        D = self.data_config.depressive_disorder
        S = self.data_config.schizophrenia_spectrum

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_patients(N, D, S, clasiifiers)

            models = []
            for label, y in bar_dict.items():
                for j, v in y.items():
                    if j not in models:
                        models.append(j)
                    for i in v:
                        i.column_name = i.column_name.split("_")[-1]

            for mod in models:
                if "Linear" in mod:
                    ind = models.index("LinearDiscriminantAnalysis")
                    models[ind] = "LDA"

            penguin_means = {
                f"{N}_AR-AU": {"data": [], "std": []},
                f"{N}_AR-CR": {"data": [], "std": []},
                f"{N}_AR-CU": {"data": [], "std": []},
                f"{N}_AU-CR": {"data": [], "std": []},
                f"{N}_AU-CU": {"data": [], "std": []},
                f"{N}_CR-CU": {"data": [], "std": []},
                f"{D}_AR-AU": {"data": [], "std": []},
                f"{D}_AR-CR": {"data": [], "std": []},
                f"{D}_AR-CU": {"data": [], "std": []},
                f"{D}_AU-CR": {"data": [], "std": []},
                f"{D}_AU-CU": {"data": [], "std": []},
                f"{D}_CR-CU": {"data": [], "std": []},
                f"{S}_AR-AU": {"data": [], "std": []},
                f"{S}_AR-CR": {"data": [], "std": []},
                f"{S}_AR-CU": {"data": [], "std": []},
                f"{S}_AU-CR": {"data": [], "std": []},
                f"{S}_AU-CU": {"data": [], "std": []},
                f"{S}_CR-CU": {"data": [], "std": []},
            }

            for patient, resu in bar_dict.items():
                for classi, res in resu.items():
                    for it in res:
                        k = f"{patient}_{it.column_name}"
                        penguin_means[k]["std"].append(it.standard_deviation)
                        penguin_means[k]["data"].append(it.mean)

            self.plot_diagram_per_strategy(strategy, models, penguin_means)

    def plot_images(self, all_export_data):
        nested_dict = self.groupby_strategy(all_export_data)

        N = self.data_config.neurotypical
        D = self.data_config.depressive_disorder
        S = self.data_config.schizophrenia_spectrum

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_patients(N, D, S, clasiifiers)

            models, bar_dictc = self.merge_results(N, D, S, bar_dict)

            for mod in models:
                if "Linear" in mod:
                    ind = models.index("LinearDiscriminantAnalysis")
                    models[ind] = "LDA"

            self.plot_diagram(strategy, models, bar_dictc)

    def plot_diagram_per_strategy(self, strategy, models, bar_dictc):
        barWidth = 0.5
        i = 0
        br_pre_pos = None
        all_br_positions = []
        colors = [
            "r",
            "r",
            "r",
            "r",
            "r",
            "r",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
        ]
        br_position = None
        legend_bars = []
        plt.subplots(figsize=(25, 10))
        for key, br_data in bar_dictc.items():
            if i > 0:
                br_position = [x + barWidth for x in br_pre_pos]
                br_pre_pos = br_position
            else:
                br_pre_pos = [0, int(len(bar_dictc) * barWidth) + 1]
                br_position = [0, int(len(bar_dictc) * barWidth) + 1]
            all_br_positions.extend(br_position)
            a = plt.bar(
                br_position,
                br_data["data"],
                color=colors[i],
                width=barWidth,
                edgecolor="grey",
                label=key,
            )

            plt.errorbar(
                br_position, br_data["data"], yerr=br_data["std"], fmt="o", color="k"
            )
            if i % (int(len(bar_dictc) / 3)) == 0:
                legend_bars.append(a)
            i = i + 1
            # Adding Xticks
        name = f"{self.brain.area} Results, {strategy} as Norm."

        plt.xlabel(name, fontweight="bold", fontsize=15)
        plt.ylabel("Accuracy", fontweight="bold", fontsize=15)

        bar_labels = [
            j.split("_")[-1] for i in range(len(models)) for j in list(bar_dictc.keys())
        ]
        all_br_positions.sort()
        plt.xticks(all_br_positions, bar_labels)
        plt.legend(legend_bars, ["N", "D", "S"], fontsize=12, title="Test")
        gname = f"{self.brain.area}_{strategy}_{self.unary_subject_binary_image_classification.__name__}"
        graph_name = ExportData.get_file_name(".png", gname)
        plt.savefig(graph_name, dpi=1200)
        plt.show()
        plt.close()

    def plot_diagram(self, strategy, models, bar_dictc):
        barWidth = 0.25
        i = 0
        br_pre = None
        colors = ["r", "g", "b"]
        br_p = None
        for key, br in bar_dictc.items():
            if i > 0:
                br_p = [x + barWidth for x in br_pre]
                br_pre = br_p
            else:
                br_pre = np.arange(len(br))
                br_p = np.arange(len(br))

            plt.bar(
                br_p,
                br,
                color=colors[i],
                width=barWidth,
                edgecolor="grey",
                label=key,
            )
            i = i + 1
            # Adding Xticks
        name = f"{self.brain.area} Results, {strategy} as Norm."

        plt.xlabel(name, fontweight="bold", fontsize=15)
        plt.ylabel("Accuracy", fontweight="bold", fontsize=15)
        plt.xticks(
            [r + barWidth for r in range(len(br_p))],
            models,
        )
        plt.legend()
        gname = f"{self.brain.area}_{strategy}_{self.unary_subject_binary_image_classification.__name__}"
        graph_name = ExportData.get_file_name(".png", gname)
        plt.savefig(graph_name, dpi=1200)
        # plt.show()
        plt.close()

    def merge_results(self, N, D, S, bar_dict):
        models = []
        bar_dictc = {
            N: [],
            D: [],
            S: [],
        }
        for label, y in bar_dict.items():
            for j, v in y.items():
                if j not in models:
                    models.append(j)
                means_per_classi = [x.mean for x in v]
                bar_dictc[label].append(statistics.mean(means_per_classi))
        return models, bar_dictc

    def separate_results_by_patients(self, N, D, S, clasiifiers):
        bar_dict = {
            N: {},
            D: {},
            S: {},
        }
        for classifier, list_per_classifier in clasiifiers.items():
            for p in list_per_classifier:
                if N in p.column_name:
                    if bar_dict[N].get(classifier) is None:
                        bar_dict[N][classifier] = [p]
                    else:
                        bar_dict[N][classifier].append(p)
                if D in p.column_name:
                    if bar_dict[D].get(classifier) is None:
                        bar_dict[D][classifier] = [p]
                    else:
                        bar_dict[D][classifier].append(p)
                if S in p.column_name:
                    if bar_dict[S].get(classifier) is None:
                        bar_dict[S][classifier] = [p]
                    else:
                        bar_dict[S][classifier].append(p)
        return bar_dict

    def groupby_strategy(self, all_export_data):
        nested_dict = {}
        for data in all_export_data:
            if nested_dict.get(data.sub_column_name) is None:
                nested_dict[data.sub_column_name] = [data]
            else:
                nested_dict[data.sub_column_name].append(data)

        for strategy in nested_dict:
            value = nested_dict.get(strategy)
            nested_dict[strategy] = {}
            for data in value:
                if nested_dict[strategy].get(data.row_name) is None:
                    nested_dict[strategy][data.row_name] = [data]
                else:
                    nested_dict[strategy][data.row_name].append(data)
        return nested_dict

    def __modify_patients(self, config: BrainDataConfig, combination):
        config.patients = []
        for comb in combination:
            match comb:
                case config.neurotypical | config.neurotypical_int:
                    config.patients.append(config.neurotypical_patients)
                case config.depressive_disorder | config.depressive_disorder_int:
                    config.patients.append(config.depressive_disorder_patients)
                case config.schizophrenia_spectrum | config.schizophrenia_spectrum_int:
                    config.patients.append(config.schizophrenia_spectrum_patients)

    def __get_note(self, mean, method):
        return f"Popmean:{mean}, Conditions:{self.data_config.conditions}, PCA fix components:{self.training_config.has_fix_components}, Executed By:{method.__name__}(...)"
