import copy
import itertools
import os
import pickle
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from nilearn import image, plotting
from sklearn.decomposition import PCA

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from DataTraining import DataTraining
from Enums import Lobe
from ExportData import ExportData
from Helper import Helper
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
        rsa_config: RSAConfig = None,
        Talairach_MNI_space: pd.DataFrame = None,
    ):
        if lobe is None:
            raise Exception("Lobe of the brain must be defined.")
        else:
            self.lobe = lobe

        if classifiers is None:
            self.classifiers = ["SVM", "MLP", "LinearDiscriminant"]
            # self.classifiers = ["SVM", "MLP", "LinearDiscriminant", "LGBM"]
        else:
            self.classifiers = classifiers

        if strategies is None:
            self.strategies = ["mice", "mean", "remove_voxels"]
            # self.strategies = ["mean", "remove_voxels", "median"]
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

        if rsa_config is None:
            self.rsa_config = RSAConfig()
        else:
            self.rsa_config = rsa_config

        match lobe:
            case Lobe.STG:
                self.brain = Brain(
                    lobe=lobe,
                    data_path=self.data_config.STG_path,
                )
                self.Talairach_MNI_space = self.brain.Talairach_MNI_space
            case Lobe.IFG:
                self.brain = Brain(
                    lobe=lobe,
                    data_path=self.data_config.IFG_path,
                )
                self.Talairach_MNI_space = self.brain.Talairach_MNI_space
            case Lobe.ALL:
                self.brain = Brain(
                    lobe=lobe,
                    data_path=self.data_config.all_lobes_path,
                )
                self.Talairach_MNI_space = self.brain.Talairach_MNI_space
            case _:
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
        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()

        print_config = self.__get_note(
            stg_subject_binary_data[0].current_labels.popmean,
            self.binary_subject_classification,
        )
        note = export.create_note([self.training_config, print_config])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files", self.binary_subject_classification.__name__
        )
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
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
        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()

        print_config = self.__get_note(
            stg_image_binary_data[0].current_labels.popmean,
            self.binary_image_classification,
        )
        note = export.create_note([self.training_config, print_config])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files", self.binary_image_classification.__name__
        )
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
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
        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.unary_subject_binary_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files",
            self.unary_subject_binary_image_classification.__name__,
        )
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
            transpose=True,
            single_label=True,
        )
        self.plot_images(
            self.unary_subject_binary_image_classification.__name__, all_export_data
        )
        self.plot_detailed_bars(
            self.unary_subject_binary_image_classification.__name__, all_export_data
        )
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.binary_subject_concatenated_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files",
            self.binary_subject_concatenated_image_classification.__name__,
        )
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.binary_subject_binary_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files",
            self.binary_subject_binary_image_classification.__name__,
        )
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.binary_subject_unary_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files",
            self.binary_subject_unary_image_classification.__name__,
        )
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
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
        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.subject_concatenated_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files",
            self.subject_concatenated_image_classification.__name__,
        )
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(mean, self.subject_binary_image_classification)
        note = export.create_note([self.training_config, print_config])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files", self.subject_binary_image_classification.__name__
        )
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
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
            self.training_config.lobe = self.brain.lobe.name
            export = ExportData()
            note = export.create_note([self.training_config])
            directory_path = Helper.ensure_dir(
                "Automated_Excel_files",
                self.binary_image_and_binary_subject_classification_with_shaply.__name__,
            )
            export.create_and_write_datasheet(
                data=export_data,
                sheet_name=f"{self.brain.lobe.name}-Results",
                title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds-{split}-Clf",
                notes=note,
                directory=directory_path,
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
            self.training_config.lobe = self.brain.lobe.name
            export = ExportData()
            note = export.create_note([self.training_config])
            directory_path = Helper.ensure_dir(
                "Automated_Excel_files",
                self.binary_image_and_binary_subject_classification_with_shaply.__name__,
            )
            export.create_and_write_datasheet(
                data=export_data,
                sheet_name=f"{self.brain.lobe.name}-Results",
                title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds-{split}-Clf",
                notes=note,
                directory=directory_path,
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        note = export.create_note([self.training_config, ""])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files", self.subject_and_image_classification.__name__
        )
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
            transpose=True,
        )

        # results = pickle.load(open("subject_and_image_classification.pickle", "rb"))

        self.plot_detailed_bars_data_labels(
            self.subject_and_image_classification.__name__, export_data
        )

    def Searchlight_Text_Embeddings(self, set_name, query, plotting=True):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """
        self.brain.current_labels = self.brain.subject_labels

        file_name = f"{self.Searchlight_Text_Embeddings.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}_{set_name}_{query}.pickle"
        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()

        results = None

        if file_path.is_file():
            results = pickle.load(open(file_path, "rb"))
        else:
            print(f"There are no saved RSA results {file_path}. Executing results.")

            if self.rsa_config.normalize:
                x = self.brain.normalize_data(
                    self.brain.voxels, self.rsa_config.strategy
                )
                self.brain.voxels = x

            subject_unary_data = self.brain.unary_fmri_subject_or_image(
                self.data_config
            )

            embeddings = pickle.load(
                open(
                    Path(f"PickleFiles/MCQ_Embeddings_{set_name}_{query}.pickle")
                    .absolute()
                    .resolve(),
                    "rb",
                )
            )

            txt_embeddings = self.average_embeddings(embeddings, 1)
            del embeddings
            txt_rdm = RepresentationalSimilarityAnalysis().create_embedding_RDM(
                txt_embeddings
            )
            results = RepresentationalSimilarityAnalysis().run_RSA(
                subject_unary_data,
                txt_rdm,
                self.rsa_config.radius,
                self.rsa_config.radius_adjustment,
                self.Talairach_MNI_space,
                self.brain.NIfTI,
            )
            with open(file_path, "wb") as output:
                pickle.dump(results, output)

        if plotting:
            for brain, smoothed_img, rsa_result in results:
                title = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} on top of Text Embeddings RDM, {set_name} {query}".replace(
                    "  ", " "
                )
                self.plot_brain_image(
                    smoothed_img, title, self.Searchlight_Text_Embeddings.__name__
                )

        return results

    def Searchlight_brain_difference_Text_Embeddings_RDM(self, set_name, query):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """

        results = self.Searchlight_Text_Embeddings(set_name, query, plotting=False)

        file_name = f"{self.Searchlight_brain_difference_Text_Embeddings_RDM.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}_{set_name}_{query}.pickle"

        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()

        difference_results = None

        if file_path.is_file():
            difference_results = pickle.load(open(file_path, "rb"))
        else:
            print(f"There are no saved RSA results {file_path}. Executing results.")
            difference_results = self.Searchlight_brain_difference(results)
            with open(file_path, "wb") as output:
                pickle.dump(difference_results, output)

        for smoothed_img, brain_k, brain_l in difference_results:
            title = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Text Embeddings RDM, {set_name} {query}".replace(
                "  ", " "
            )
            self.plot_brain_image(
                smoothed_img,
                title,
                self.Searchlight_brain_difference_Text_Embeddings_RDM.__name__,
            )

    def Searchlight_Video_Embeddings(self, set_name, query, plotting=True):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """
        self.brain.current_labels = self.brain.subject_labels

        file_name = f"{self.Searchlight_Video_Embeddings.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}_{set_name}_{query}.pickle"
        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()

        results = None

        if file_path.is_file():
            results = pickle.load(open(file_path, "rb"))
        else:
            print(f"There are no saved RSA results {file_path}. Executing results.")

            if self.rsa_config.normalize:
                x = self.brain.normalize_data(
                    self.brain.voxels, self.rsa_config.strategy
                )
                self.brain.voxels = x

            subject_unary_data = self.brain.unary_fmri_subject_or_image(
                self.data_config
            )

            embeddings = pickle.load(
                open(
                    Path(f"PickleFiles/MCQ_Embeddings_{set_name}_{query}.pickle")
                    .absolute()
                    .resolve(),
                    "rb",
                )
            )

            video_embeddings = self.average_embeddings(embeddings, 4)
            del embeddings
            video_rdm = RepresentationalSimilarityAnalysis().create_embedding_RDM(
                video_embeddings
            )
            results = RepresentationalSimilarityAnalysis().run_RSA(
                subject_unary_data,
                video_rdm,
                self.rsa_config.radius,
                self.rsa_config.radius_adjustment,
                self.Talairach_MNI_space,
                self.brain.NIfTI,
            )
            with open(file_path, "wb") as output:
                pickle.dump(results, output)

        if plotting:
            for brain, smoothed_img, rsa_result in results:
                title = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} on top of Video Embeddings RDM, {set_name} {query}".replace(
                    "  ", " "
                )
                self.plot_brain_image(
                    smoothed_img, title, self.Searchlight_Video_Embeddings.__name__
                )

        return results

    def Searchlight_brain_difference_Video_Embeddings_RDM(self, set_name, query):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """

        results = self.Searchlight_Video_Embeddings(set_name, query, plotting=False)

        file_name = f"{self.Searchlight_brain_difference_Video_Embeddings_RDM.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}_{set_name}_{query}.pickle"

        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()

        difference_results = None

        if file_path.is_file():
            difference_results = pickle.load(open(file_path, "rb"))
        else:
            print(f"There are no saved RSA results {file_path}. Executing results.")
            difference_results = self.Searchlight_brain_difference(results)
            with open(file_path, "wb") as output:
                pickle.dump(difference_results, output)

        for smoothed_img, brain_k, brain_l in difference_results:
            title = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Video Embeddings RDM, {set_name} {query}".replace(
                "  ", " "
            )
            self.plot_brain_image(
                smoothed_img,
                title,
                self.Searchlight_brain_difference_Video_Embeddings_RDM.__name__,
            )

    def Searchlight_Bridge_Embeddings(self, set_name, query, plotting=True):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """
        self.brain.current_labels = self.brain.subject_labels

        file_name = f"{self.Searchlight_Bridge_Embeddings.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}_{set_name}_{query}.pickle"
        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()

        results = None

        if file_path.is_file():
            results = pickle.load(open(file_path, "rb"))
        else:
            print(f"There are no saved RSA results {file_path}. Executing results.")

            if self.rsa_config.normalize:
                x = self.brain.normalize_data(
                    self.brain.voxels, self.rsa_config.strategy
                )
                self.brain.voxels = x

            subject_unary_data = self.brain.unary_fmri_subject_or_image(
                self.data_config
            )

            embeddings = pickle.load(
                open(
                    Path(f"PickleFiles/MCQ_Embeddings_{set_name}_{query}.pickle")
                    .absolute()
                    .resolve(),
                    "rb",
                )
            )

            bridge_embeddings = self.average_embeddings(embeddings, 3)
            del embeddings
            bridge_rdm = RepresentationalSimilarityAnalysis().create_embedding_RDM(
                bridge_embeddings
            )
            results = RepresentationalSimilarityAnalysis().run_RSA(
                subject_unary_data,
                bridge_rdm,
                self.rsa_config.radius,
                self.rsa_config.radius_adjustment,
                self.Talairach_MNI_space,
                self.brain.NIfTI,
            )
            with open(file_path, "wb") as output:
                pickle.dump(results, output)

        if plotting:
            for brain, smoothed_img, rsa_result in results:
                title = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} on top of Bridge Embeddings RDM, {set_name} {query}".replace(
                    "  ", " "
                )
                self.plot_brain_image(
                    smoothed_img, title, self.Searchlight_Bridge_Embeddings.__name__
                )

        return results

    def Searchlight_brain_difference_Bridge_Embeddings_RDM(self, set_name, query):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """

        results = self.Searchlight_Bridge_Embeddings(set_name, query, plotting=False)

        file_name = f"{self.Searchlight_brain_difference_Bridge_Embeddings_RDM.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}_{set_name}_{query}.pickle"

        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()

        difference_results = None

        if file_path.is_file():
            difference_results = pickle.load(open(file_path, "rb"))
        else:
            print(f"There are no saved RSA results {file_path}. Executing results.")
            difference_results = self.Searchlight_brain_difference(results)
            with open(file_path, "wb") as output:
                pickle.dump(difference_results, output)

        for smoothed_img, brain_k, brain_l in difference_results:
            title = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Bridge Embeddings RDM, {set_name} {query}".replace(
                "  ", " "
            )
            self.plot_brain_image(
                smoothed_img,
                title,
                self.Searchlight_brain_difference_Bridge_Embeddings_RDM.__name__,
            )

    def average_embeddings(self, embeddings, index):
        """
        key => 0;
        text_embed => 1;
        answer_embed => 2;
        bridge_embed => 3;
        vid_embed => 4;
        en_de_txt => 5;
        """
        if index < 1 or index > 4:
            raise IndexError("To process embeddings index range is [1-4]")

        abstract_related = "mp_dt"
        abstract_unrelated = "fm_dt"
        concrete_related = "ik_dt"
        concrete_unrelated = "fi_dt"
        ar_embed = []
        au_embed = []
        cr_embed = []
        cu_embed = []
        for embed in embeddings:
            # key,text_embed,answer_embed,bridge_embed,vid_embed,en_de_txt = embed
            if abstract_related in embed[0]:
                ar_embed.append(embed[index].cpu())
            elif abstract_unrelated in embed[0]:
                au_embed.append(embed[index].cpu())
            elif concrete_related in embed[0]:
                cr_embed.append(embed[index].cpu())
            elif concrete_unrelated in embed[0]:
                cu_embed.append(embed[index].cpu())

        # a = [1.0, 2.0, 3.0, 4.0]
        # ta = torch.tensor(a)
        # b = [5.0, 6.0, 7.0, 8.0]
        # tb = torch.tensor(b)
        # f = torch.stack([ta, tb]).mean(dim=0)
        ar_embed_m = torch.stack(ar_embed).mean(dim=0)
        au_embed_m = torch.stack(au_embed).mean(dim=0)
        cr_embed_m = torch.stack(cr_embed).mean(dim=0)
        cu_embed_m = torch.stack(cu_embed).mean(dim=0)
        avg_embeddings = np.array([ar_embed_m, au_embed_m, cr_embed_m, cu_embed_m])

        return avg_embeddings

    def Searchlight_Abstract_Concrete_RDM(self, plotting=True):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """
        self.brain.current_labels = self.brain.subject_labels

        file_name = f"{self.Searchlight_Abstract_Concrete_RDM.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}.pickle"
        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()

        results = None

        if file_path.is_file():
            results = pickle.load(open(file_path, "rb"))
        else:
            print(f"There are no saved RSA results {file_path}. Executing results.")
            if self.rsa_config.normalize:
                x = self.brain.normalize_data(
                    self.brain.voxels, self.rsa_config.strategy
                )
                self.brain.voxels = x

            subject_unary_data = self.brain.unary_fmri_subject_or_image(
                self.data_config
            )
            results = RepresentationalSimilarityAnalysis().run_RSA(
                subject_unary_data,
                self.rsa_config.abstract_concrete_RDM,
                self.rsa_config.radius,
                self.rsa_config.radius_adjustment,
                self.Talairach_MNI_space,
                self.brain.NIfTI,
            )
            with open(file_path, "wb") as output:
                pickle.dump(results, output)

        for brain, smoothed_img, rsa_result in results:
            if plotting:
                title = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} on top of Abstract Concrete RDM".replace(
                    "  ", " "
                )
                self.plot_brain_image(
                    smoothed_img, title, self.Searchlight_Abstract_Concrete_RDM.__name__
                )

        return results

    def Searchlight_brain_difference_Abstract_Concrete_RDM(self):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """

        results = self.Searchlight_Abstract_Concrete_RDM(plotting=False)

        file_name = f"{self.Searchlight_brain_difference_Abstract_Concrete_RDM.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}.pickle"

        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()

        difference_results = None

        if file_path.is_file():
            difference_results = pickle.load(open(file_path, "rb"))
        else:
            print(f"There are no saved RSA results {file_path}. Executing results.")
            difference_results = self.Searchlight_brain_difference(results)
            with open(file_path, "wb") as output:
                pickle.dump(difference_results, output)

        for smoothed_img, brain_k, brain_l in difference_results:
            title = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Abstract Concrete RDM".replace(
                "  ", " "
            )
            self.plot_brain_image(
                smoothed_img,
                title,
                self.Searchlight_brain_difference_Abstract_Concrete_RDM.__name__,
            )

    def Searchlight_brain_difference_Related_Unrelated_RDM(self):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """

        results = self.Searchlight_Related_Unrelated_RDM(plotting=False)

        file_name = f"{self.Searchlight_brain_difference_Related_Unrelated_RDM.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}.pickle"

        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()

        difference_results = None

        if file_path.is_file():
            difference_results = pickle.load(open(file_path, "rb"))
        else:
            print(f"There are no saved RSA results {file_path}. Executing results.")
            difference_results = self.Searchlight_brain_difference(results)
            with open(file_path, "wb") as output:
                pickle.dump(difference_results, output)

        for smoothed_img, brain_k, brain_l in difference_results:
            title = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Related Unrelated RDM".replace(
                "  ", " "
            )
            self.plot_brain_image(
                smoothed_img,
                title,
                self.Searchlight_brain_difference_Related_Unrelated_RDM.__name__,
            )

    def Searchlight_brain_difference(self, results):
        difference_results = []
        combinations = itertools.combinations(list(range(len(results))), 2)
        for k, l in combinations:
            smoothed_img = image.smooth_img(self.brain.NIfTI, None)
            smoothed_img._dataobj = np.zeros(smoothed_img._dataobj.shape)
            brain_k, smoothed_img_k, rsa_result_k = results[k]
            brain_l, smoothed_img_l, rsa_result_l = results[l]
            assert len(rsa_result_k) == len(rsa_result_l)
            i = 0
            while i < len(rsa_result_k):
                sph_cntr_k, vox_indices_k, r_k, aal_coors_k = rsa_result_k[i]
                sph_cntr_l, vox_indices_l, r_l, aal_coors_l = rsa_result_l[i]

                assert sph_cntr_k == sph_cntr_l
                assert vox_indices_k == vox_indices_l
                assert aal_coors_k == aal_coors_l

                for vox_index in vox_indices_k:
                    for aal_coo in aal_coors_k:
                        smoothed_img._dataobj[aal_coo] = abs(r_k - r_l)

                i += 1

            smoothed_img._data_cache = smoothed_img._dataobj
            difference_results.append(
                (
                    smoothed_img,
                    brain_k,
                    brain_l,
                )
            )
        return difference_results

    def Searchlight_Related_Unrelated_RDM(self, plotting=True):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """
        self.brain.current_labels = self.brain.subject_labels

        file_name = f"{self.Searchlight_Related_Unrelated_RDM.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_{self.is_normalized().replace(' ', '_')}.pickle"

        file_path = Path(f"PickleFiles/{file_name}").absolute().resolve()
        results = None

        if file_path.is_file():
            results = pickle.load(open(file_path, "rb"))
        else:
            print(
                f"There are no saved RSA results {file_path}. RSA function will be executed."
            )
            if self.rsa_config.normalize:
                x = self.brain.normalize_data(
                    self.brain.voxels, self.rsa_config.strategy
                )
                self.brain.voxels = x

            subject_unary_data = self.brain.unary_fmri_subject_or_image(
                self.data_config
            )
            results = RepresentationalSimilarityAnalysis().run_RSA(
                subject_unary_data,
                self.rsa_config.abstract_concrete_RDM,
                self.rsa_config.radius,
                self.rsa_config.radius_adjustment,
                self.Talairach_MNI_space,
                self.brain.NIfTI,
            )
            with open(file_path, "wb") as output:
                pickle.dump(results, output)

        for brain, smoothed_img, rsa_result in results:
            if plotting:
                title = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} on top of Related Unrelated RDM".replace(
                    "  ", " "
                )
                self.plot_brain_image(
                    smoothed_img, title, self.Searchlight_Related_Unrelated_RDM.__name__
                )

        return results

    def plot_brain_image(self, smoothed_img, title, directory, show=False):
        # rdm_typ = f"{self.rsa_config.related_unrelated_RDM=}".split("=")[0].split(".")[2]

        directory_path = Helper.ensure_dir("Searchlight_Graphs", directory)
        # directory_path = Path(f"Searchlight_Graphs/{directory}").absolute().resolve()
        # Path(directory_path).mkdir(parents=True, exist_ok=True)

        # display, axes = plotting.plot_img_on_surf( smoothed_img,surf_mesh="fsaverage", views=["lateral", "medial"],hemispheres=["left", "right"],inflate=False,colorbar=True,bg_on_data=True,cmap="hsv_r")
        display = plotting.plot_glass_brain(
            smoothed_img, threshold=0, title=title, display_mode="lzry", colorbar=True
        )
        # display = plotting.plot_stat_map(smoothed_img, threshold=0)
        # display.savefig("pretty_brain.png")
        # plotting.plot_glass_brain(smoothed_img, threshold=0)
        time.sleep(1)

        graph_name = ExportData.get_graph_name(".png", title.replace(" ", "_"))

        file_path = Path(directory_path).joinpath(graph_name)

        plt.savefig(file_path)

        if show:
            plotting.show()
            display.close()
            plt.close()
        # display = plotting.plot_glass_brain(None, plot_abs=False, display_mode="lzry", title=title)
        # display.add_contours(smoothed_img, filled=True)
        # plotting.show()

        # graph_name = ExportData.get_file_name(".png", title.replace(" ", "_"))
        # plt.savefig(graph_name)
        # display.close()
        # plt.close()

    def plot_brain_image_test(self, smoothed_img, title, directory, show=False):
        # rdm_typ = f"{self.rsa_config.related_unrelated_RDM=}".split("=")[0].split(".")[2]
        from nilearn import datasets, surface

        atlas = datasets.fetch_atlas_talairach("ba")

        directory_path = Helper.ensure_dir("Searchlight_Graphs", directory)

        # display, axes = plotting.plot_img_on_surf( smoothed_img,surf_mesh="fsaverage", views=["lateral", "medial"],hemispheres=["left", "right"],inflate=False,colorbar=True,bg_on_data=True,cmap="hsv_r")
        display = plotting.plot_glass_brain(
            smoothed_img, threshold=0, title=title, display_mode="lzry", colorbar=True
        )
        # display = plotting.plot_stat_map(smoothed_img, threshold=0)
        # display.savefig("pretty_brain.png")
        # plotting.plot_glass_brain(smoothed_img, threshold=0)
        time.sleep(1)

        graph_name = ExportData.get_graph_name(".png", title.replace(" ", "_"))

        file_path = Path(directory_path).joinpath(graph_name)

        plt.savefig(file_path)

        if show:
            plotting.show()
            display.close()
            plt.close()
        # display = plotting.plot_glass_brain(None, plot_abs=False, display_mode="lzry", title=title)
        # display.add_contours(smoothed_img, filled=True)
        # plotting.show()

        # graph_name = ExportData.get_file_name(".png", title.replace(" ", "_"))
        # plt.savefig(graph_name)
        # display.close()
        # plt.close()

    def tes(
        self,
    ):
        import cmasher as cmr
        import nilearn
        import numpy as np
        from matplotlib import cm
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        from nilearn import datasets, plotting, surface

        # Initialization
        color = "hsv_r"
        range_cmap = cmr.get_sub_cmap(color, 0.3, 1)

        # Custom colormap

        upper = cm.get_cmap(range_cmap, 999)
        lower = cm.get_cmap(range_cmap, 999)

        combine = np.vstack(
            (upper(np.linspace(0, 1, 999)), lower(np.linspace(0, 1, 999)))
        )
        custom_cmap = ListedColormap(combine, name="custom_map")

        # Data
        nifti = nilearn.image.load_img("load-your-nifti-file")

        # Cortical mesh
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")

        # Sample the 3D data around each node of the mesh
        texture = surface.vol_to_surf(nifti, fsaverage.pial_right)

        # Plot
        fig = plotting.plot_surf_stat_map(
            surf_mesh=fsaverage.pial_right,
            stat_map=texture,
            bg_map=fsaverage.sulc_right,
            bg_on_data=True,
            alpha=1,
            vmax=8.64,
            threshold=False,
            hemi="right",
            title="Surface right hemisphere",
            colorbar=True,
            symmetric_cbar=False,
            cmap=custom_cmap,
        )
        fig.show()

    """
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
    'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 
    'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 
    'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 
    'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn',
    'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r',
    'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
    'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'black_blue', 'black_blue_r', 'black_green', 'black_green_r', 
    'black_pink', 'black_pink_r', 'black_purple', 'black_purple_r', 'black_red', 'black_red_r', 'blue_orange', 'blue_orange_r',
    'blue_red', 'blue_red_r', 'blue_transparent', 'blue_transparent_full_alpha_range', 'bone', 'bone_r', 'brg', 'brg_r', 
    'brown_blue', 'brown_blue_r', 'brown_cyan', 'brown_cyan_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cold_hot', 'cold_hot_r',
    'cold_white_hot', 'cold_white_hot_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 
    'cubehelix', 'cubehelix_r', 'cyan_copper', 'cyan_copper_r', 'cyan_orange', 'cyan_orange_r', 'flag', 'flag_r', 'flare', 
    'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 
    'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 
    'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'green_transparent', 'green_transparent_full_alpha_range', 
    'grey', 'hot', 'hot_black_bone', 'hot_black_bone_r', 'hot_r', 'hot_white_bone', 'hot_white_bone_r', 'hsv', 'hsv_r', 'icefire',
    'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r',
    'ocean', 'ocean_hot', 'ocean_hot_r', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'purple_blue',
    'purple_blue_r', 'purple_green', 'purple_green_r', 'rainbow', 'rainbow_r', 'red_transparent', 'red_transparent_full_alpha_range',
    'rocket', 'rocket_r', 'roy_big_bl', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r',
    'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 
    'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'videen_style', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter',
    'winter_r'
    """

    def is_normalized(self):
        normalized = ""
        if self.rsa_config.normalize:
            normalized = f"{self.rsa_config.strategy} imputation"
        return normalized

    def lobe_name(self, brain):
        lobe = brain.lobe.name
        if brain.lobe == Lobe.ALL:
            lobe = f"{lobe} Lobes"
        return lobe

    def plot_detailed_bars_data_labels(self, directory, all_export_data):
        """
        This method plot the detailed graphs, with Image labels and subject labels, std and significant
        """
        nested_dict = self.groupby_strategy(all_export_data)

        subject_l = "subject"
        image_l = "image"

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_labels(subject_l, image_l, clasiifiers)

            models = []
            for label, y in bar_dict.items():
                for j, v in y.items():
                    if j not in models:
                        models.append(j)

            data_stat = {
                f"{subject_l}": {"data": [], "std": [], "result": []},
                f"{image_l }": {"data": [], "std": [], "result": []},
            }

            for label, resu in bar_dict.items():
                for classi, res in resu.items():
                    for it in res:
                        data_stat[label]["std"].append(it.standard_deviation)
                        data_stat[label]["data"].append(it.mean)
                        data_stat[label]["result"].append(it.result[0])

            self.plot_diagram_per_strategy(
                strategy,
                models,
                data_stat,
                directory,
                legends=[
                    f"Subject {chr(10)} (N, D, S)",
                    f"Image {chr(10)} (AR, AU, CR, CU)",
                ],
                legend_font=13,
                legend_title="Labels",
            )

    def plot_detailed_bars(self, directory, all_export_data):
        """
        This method plot the detailed graphs, with binary 6 combinations, std and significant
        """
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

            # for mod in models:
            # if "Linear" in mod:
            # ind = models.index("LinearDiscriminantAnalysis")
            # models[ind] = "LDA"

            data_stat = {
                f"{N}_AR-AU": {"data": [], "std": [], "result": []},
                f"{N}_AR-CR": {"data": [], "std": [], "result": []},
                f"{N}_AR-CU": {"data": [], "std": [], "result": []},
                f"{N}_AU-CR": {"data": [], "std": [], "result": []},
                f"{N}_AU-CU": {"data": [], "std": [], "result": []},
                f"{N}_CR-CU": {"data": [], "std": [], "result": []},
                f"{D}_AR-AU": {"data": [], "std": [], "result": []},
                f"{D}_AR-CR": {"data": [], "std": [], "result": []},
                f"{D}_AR-CU": {"data": [], "std": [], "result": []},
                f"{D}_AU-CR": {"data": [], "std": [], "result": []},
                f"{D}_AU-CU": {"data": [], "std": [], "result": []},
                f"{D}_CR-CU": {"data": [], "std": [], "result": []},
                f"{S}_AR-AU": {"data": [], "std": [], "result": []},
                f"{S}_AR-CR": {"data": [], "std": [], "result": []},
                f"{S}_AR-CU": {"data": [], "std": [], "result": []},
                f"{S}_AU-CR": {"data": [], "std": [], "result": []},
                f"{S}_AU-CU": {"data": [], "std": [], "result": []},
                f"{S}_CR-CU": {"data": [], "std": [], "result": []},
            }

            for patient, resu in bar_dict.items():
                for classi, res in resu.items():
                    for it in res:
                        k = f"{patient}_{it.column_name}"
                        data_stat[k]["std"].append(it.standard_deviation)
                        data_stat[k]["data"].append(it.mean)
                        data_stat[k]["result"].append(it.result[0])

            self.plot_diagram_per_strategy(
                strategy, models, data_stat, directory, legends=["N", "D", "S"]
            )

    def plot_images(self, directory, all_export_data):
        nested_dict = self.groupby_strategy(all_export_data)

        N = self.data_config.neurotypical
        D = self.data_config.depressive_disorder
        S = self.data_config.schizophrenia_spectrum

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_patients(N, D, S, clasiifiers)

            models, bar_dictc = self.merge_results(N, D, S, bar_dict)

            # for mod in models:
            # if "Linear" in mod:
            # ind = models.index("LinearDiscriminantAnalysis")
            # models[ind] = "LDA"

            self.plot_diagram(strategy, models, bar_dictc, directory)

    def plot_diagram_per_strategy(
        self,
        strategy,
        models,
        bar_data,
        directory,
        legends,
        legend_title="Mental Disorders",
        legend_font=18,
    ):
        bar_types_per_model = len(legends)
        barWidth = 0.5
        i = 0
        br_pre_pos = None
        all_br_positions = []
        color_len = int(len(bar_data) / bar_types_per_model)
        colors = ["tomato" for x in range(color_len)]
        colors.extend(["limegreen" for x in range(color_len)])
        colors.extend(["dodgerblue" for x in range(color_len)])

        bar_labels = [
            j.split("_")[-1] for i in range(len(models)) for j in list(bar_data.keys())
        ]

        br_position = None
        legend_bars = []
        fig, ax = plt.subplots(figsize=(25, 10))
        plt.rcParams.update({"ytick.labelsize": 15})
        plt.rcParams.update({"legend.title_fontsize": 15})
        for key, br_data in bar_data.items():
            if i > 0:
                br_position = [x + barWidth for x in br_pre_pos]
                br_pre_pos = br_position
            else:
                nu = [0]
                # br_pre_pos = [0, int(len(bar_data) * barWidth) + 1]
                # br_position = [0, int(len(bar_data) * barWidth) + 1]
                k = 0
                for _ in range(len(br_data["data"]) - 1):
                    k = int(len(bar_data) * barWidth) + k + 1
                    nu.append(k)
                br_pre_pos = nu
                br_position = nu

            all_br_positions.extend(br_position)
            a = plt.bar(
                br_position,
                br_data["data"],
                color=colors[i],
                width=barWidth,
                edgecolor="grey",
                label=key,
            )
            for index, d in enumerate(br_data["data"]):
                plt.text(
                    br_position[index],
                    0.5 * d,
                    bar_labels[i],
                    ha="center",
                    va="top",
                    color="white",
                    rotation="vertical",
                    fontsize=17,
                )
            for index, res in enumerate(br_data["result"]):
                if "Not" not in res:
                    plt.text(
                        br_position[index],
                        0,
                        "*",
                        ha="center",
                        va="baseline",
                        color="k",
                        fontsize=25,
                    )

            plt.errorbar(
                br_position, br_data["data"], yerr=br_data["std"], fmt="o", color="k"
            )
            if i % (int(len(bar_data) / bar_types_per_model)) == 0:
                legend_bars.append(a)
            i = i + 1
            # Adding Xticks

        lobe_n = self.brain.lobe.name
        if "All" in lobe_n:
            lobe_n = "All lobes"

        title = f"{lobe_n} Results, {strategy} as data imputation"

        plt.xlabel(title, fontweight="bold", fontsize=22)
        plt.ylabel("Accuracy", fontweight="bold", fontsize=15)

        all_br_positions.sort()
        tick_pos = []
        bar_types_per_model = int((len(all_br_positions) / len(models)))
        end = 0
        start = 0
        while end < len(all_br_positions):
            end = end + bar_types_per_model
            seg = all_br_positions[start:end]
            tick_pos.append(seg[int(len(seg) / 2)] - barWidth / 2)
            start = end

        plt.xticks(tick_pos, models, fontsize=20)

        plt.legend(
            legend_bars,
            legends,
            fontsize=legend_font,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title=legend_title,
        )
        # plt.legend(legend_bars, ["N", "D", "S"], fontsize=18, loc='upper left', bbox_to_anchor=(1, 1) ,title_fontsize=14,title="Mental Disorders")
        gname = f"{self.brain.lobe.name}_{strategy}_{directory}"
        graph_name = ExportData.get_file_name(".png", gname)
        # plt.savefig(graph_name, dpi=1200)

        # directory_path = Path(f"MlGraphs/{directory}").absolute().resolve()
        # Path(directory_path).mkdir(parents=True, exist_ok=True)

        directory_path = Helper.ensure_dir("Ml_Graphs", directory)
        file_path = Path(directory_path).joinpath(graph_name)
        plt.savefig(file_path)
        # plt.show()
        plt.close()

    def plot_diagram(self, strategy, models, bar_dictc, directory):
        barWidth = 0.25
        i = 0
        br_pre = None
        colors = ["tomato", "limegreen", "dodgerblue"]
        br_p = None
        fig, ax = plt.subplots(figsize=(25, 10))
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
        name = f"{self.brain.lobe.name} Results, {strategy} as normalization"

        plt.xlabel(name, fontweight="bold", fontsize=15)
        plt.ylabel("Accuracy", fontweight="bold", fontsize=15)
        plt.xticks([r + barWidth for r in range(len(br_p))], models, fontsize=20)
        plt.legend(fontsize=18, title="Mental Disorders", loc="upper right")
        gname = f"{self.brain.lobe.name}_{strategy}_{self.unary_subject_binary_image_classification.__name__}"
        graph_name = ExportData.get_file_name(".png", gname)
        # plt.savefig(graph_name, dpi=1200)

        directory_path = Helper.ensure_dir("Ml_Graphs", directory)

        file_path = Path(directory_path).joinpath(graph_name)
        plt.savefig(file_path)
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

    def separate_results_by_labels(self, subject_l, image_l, clasiifiers):
        bar_dict = {
            subject_l: {},
            image_l: {},
        }
        for classifier, list_per_classifier in clasiifiers.items():
            for p in list_per_classifier:
                if subject_l in p.column_name:
                    if bar_dict[subject_l].get(classifier) is None:
                        bar_dict[subject_l][classifier] = [p]
                    else:
                        bar_dict[subject_l][classifier].append(p)
                if image_l in p.column_name:
                    if bar_dict[image_l].get(classifier) is None:
                        bar_dict[image_l][classifier] = [p]
                    else:
                        bar_dict[image_l][classifier].append(p)
        return bar_dict

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
