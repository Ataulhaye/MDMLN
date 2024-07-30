import copy
import itertools
import pickle
import time
from heapq import nlargest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nilearn import image

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from DataTraining import DataTraining
from Enums import Lobe
from ExportData import ExportData
from Helper import Helper
from PlotData import Visualization
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

        optional_txt = None
        if self.training_config.dimension_reduction == True:
            optional_txt = "PCA"
        if self.training_config.use_autoencoder == True:
            optional_txt = "Autoencoder"

        visu = Visualization()
        visu.plot_merged_bars(
            directory=self.unary_subject_binary_image_classification.__name__,
            all_data=all_export_data,
            lobe_name=self.brain.lobe.name,
            opt_info=optional_txt,
        )
        time.sleep(5)
        visu.plot_detailed_bars(
            directory=self.unary_subject_binary_image_classification.__name__,
            all_data=all_export_data,
            lobe_name=self.brain.lobe.name,
            opt_info=optional_txt,
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

        optional_txt = None
        if self.training_config.dimension_reduction == True:
            optional_txt = "PCA"
        if self.training_config.use_autoencoder == True:
            optional_txt = "Autoencoder"

        visu = Visualization()
        visu.plot_merged_bars(
            self.binary_subject_binary_image_classification.__name__,
            all_data,
            self.brain.lobe.name,
            N="N-D",
            D="N-S",
            S="D-S",
            legend_text=[
                f"Neurotypical & {chr(10)} Depressive",
                f"Neurotypical & {chr(10)} Schizophrenia",
                f"Depressive & {chr(10)} Schizophrenia",
            ],
            opt_info=optional_txt,
        )
        time.sleep(5)
        visu.plot_detailed_bars(
            directory=self.binary_subject_binary_image_classification.__name__,
            all_data=all_data,
            lobe_name=self.brain.lobe.name,
            N="N-D",
            D="N-S",
            S="D-S",
            legend_text=[
                f"Neurotypical & {chr(10)} Depressive",
                f"Neurotypical & {chr(10)} Schizophrenia",
                f"Depressive & {chr(10)} Schizophrenia",
            ],
            opt_info=optional_txt,
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
        optional_txt = None
        if self.training_config.dimension_reduction == True:
            optional_txt = "PCA"
        if self.training_config.use_autoencoder == True:
            optional_txt = "Autoencoder"

        Visualization().plot_detailed_bars_data_labels(
            directory=self.subject_and_image_classification.__name__,
            lobe_name=self.brain.lobe.name,
            all_data=export_data,
            opt_info=optional_txt,
        )

    def Searchlight_Text_Embeddings(self, set_name="Set1", query="verb", plotting=True):
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
                pdf, png, csv = ExportData.create_figure_names(
                    title.replace(" ", "_"), opt=".csv"
                )
                self.export_top_similarities(
                    rsa_result,
                    title,
                    csv,
                    self.Searchlight_Text_Embeddings.__name__,
                )
                Visualization().plot_brain_image(
                    smoothed_img,
                    title,
                    self.Searchlight_Text_Embeddings.__name__,
                    pdf,
                    png,
                )

        return results

    def Searchlight_brain_difference_Text_Embeddings_RDM(
        self, set_name="Set1", query="verb"
    ):
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

        for smoothed_img, brain_k, brain_l, diffrences in difference_results:
            title = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Text Embeddings RDM, {set_name} {query}".replace(
                "  ", " "
            )
            pdf, png, csv = ExportData.create_figure_names(
                title.replace(" ", "_"), opt=".csv"
            )
            self.export_top_similarities(
                diffrences,
                title,
                csv,
                self.Searchlight_brain_difference_Text_Embeddings_RDM.__name__,
            )
            Visualization().plot_brain_image(
                smoothed_img,
                title,
                self.Searchlight_brain_difference_Text_Embeddings_RDM.__name__,
                pdf,
                png,
            )

    def Searchlight_Video_Embeddings(
        self, set_name="Set1", query="verb", plotting=True
    ):
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
                pdf, png, csv = ExportData.create_figure_names(
                    title.replace(" ", "_"), opt=".csv"
                )
                self.export_top_similarities(
                    rsa_result,
                    title,
                    csv,
                    self.Searchlight_Video_Embeddings.__name__,
                )
                Visualization().plot_brain_image(
                    smoothed_img,
                    title,
                    self.Searchlight_Video_Embeddings.__name__,
                    pdf,
                    png,
                )

        return results

    def Searchlight_brain_difference_Video_Embeddings_RDM(
        self, set_name="Set1", query="verb"
    ):
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

        for smoothed_img, brain_k, brain_l, diffrences in difference_results:
            title = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Video Embeddings RDM, {set_name} {query}".replace(
                "  ", " "
            )
            pdf, png, csv = ExportData.create_figure_names(
                title.replace(" ", "_"), opt=".csv"
            )
            self.export_top_similarities(
                diffrences,
                title,
                csv,
                self.Searchlight_brain_difference_Video_Embeddings_RDM.__name__,
            )
            Visualization().plot_brain_image(
                smoothed_img,
                title,
                self.Searchlight_brain_difference_Video_Embeddings_RDM.__name__,
                pdf,
                png,
            )

    def Searchlight_Bridge_Embeddings(
        self, set_name="Set1", query="verb", plotting=True
    ):
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
                pdf, png, csv = ExportData.create_figure_names(
                    title.replace(" ", "_"), opt=".csv"
                )
                self.export_top_similarities(
                    rsa_result,
                    title,
                    csv,
                    self.Searchlight_Bridge_Embeddings.__name__,
                )
                Visualization().plot_brain_image(
                    smoothed_img,
                    title,
                    self.Searchlight_Bridge_Embeddings.__name__,
                    pdf,
                    png,
                )

        return results

    def Searchlight_brain_difference_Bridge_Embeddings_RDM(
        self, set_name="Set1", query="verb"
    ):
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

        for smoothed_img, brain_k, brain_l, diffrences in difference_results:
            title = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Bridge Embeddings RDM, {set_name} {query}".replace(
                "  ", " "
            )
            pdf, png, csv = ExportData.create_figure_names(
                title.replace(" ", "_"), opt=".csv"
            )
            self.export_top_similarities(
                diffrences,
                title,
                csv,
                self.Searchlight_brain_difference_Bridge_Embeddings_RDM.__name__,
            )
            Visualization().plot_brain_image(
                smoothed_img,
                title,
                self.Searchlight_brain_difference_Bridge_Embeddings_RDM.__name__,
                pdf,
                png,
            )

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
                fig_name = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} on top of Abstract Concrete RDM".replace(
                    "  ", " "
                )
                pdf, png, csv = ExportData.create_figure_names(
                    fig_name.replace(" ", "_"), opt=".csv"
                )
                self.export_top_similarities(
                    rsa_result,
                    fig_name,
                    csv,
                    self.Searchlight_Abstract_Concrete_RDM.__name__,
                )
                stat, patient = self.get_strategy_patient_names(brain)

                fig_title = f"Searchlight RSA Analysis of Abstract/Concrete RDM using {stat} Imputation in {patient}"
                Visualization().plot_brain_image(
                    smoothed_img,
                    fig_title,
                    self.Searchlight_Abstract_Concrete_RDM.__name__,
                    pdf,
                    png,
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

        for smoothed_img, brain_k, brain_l, diffrences in difference_results:
            fig_name = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Abstract Concrete RDM".replace(
                "  ", " "
            )
            pdf, png, csv = ExportData.create_figure_names(
                fig_name.replace(" ", "_"), opt=".csv"
            )
            self.export_top_similarities(
                diffrences,
                fig_name,
                csv,
                self.Searchlight_brain_difference_Abstract_Concrete_RDM.__name__,
            )
            stat, patient_k = self.get_strategy_patient_names(brain_k)
            stat, patient_l = self.get_strategy_patient_names(brain_l)
            # Searchlight RSA Analysis of Related/Unrelated RDM using MICE imputation in Neurotypicals
            # Neurotypical-Depressive Differences of Related/Unrelated RDM using MICE Imputation
            #Abstractness = abstract/concrete
            fig_title = f"{patient_k[:-1]}-{patient_l[:-1]} Differences in Conceptual Abstractness using {stat} Imputation"

            Visualization().plot_brain_image(
                smoothed_img,
                fig_title,
                self.Searchlight_brain_difference_Abstract_Concrete_RDM.__name__,
                pdf,
                png,
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

        for smoothed_img, brain_k, brain_l, diffrences in difference_results:
            fig_name = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} on top of Related Unrelated RDM".replace(
                "  ", " "
            )
            pdf, png, csv = ExportData.create_figure_names(
                fig_name.replace(" ", "_"), opt=".csv"
            )
            self.export_top_similarities(
                diffrences,
                fig_name,
                csv,
                self.Searchlight_brain_difference_Related_Unrelated_RDM.__name__,
            )
            stat, patient_k = self.get_strategy_patient_names(brain_k)
            stat, patient_l = self.get_strategy_patient_names(brain_l)
            # Searchlight RSA Analysis of Related/Unrelated RDM using MICE imputation in Neurotypicals
            # Neurotypical-Depressive Differences of Related/Unrelated RDM using MICE Imputation
            # Relatedness = related/unrelated
            fig_title = f"{patient_k[:-1]}-{patient_l[:-1]} Differences in Conceptual Relatedness using {stat} Imputation"

            Visualization().plot_brain_image(
                smoothed_img,
                fig_title,
                self.Searchlight_brain_difference_Related_Unrelated_RDM.__name__,
                pdf,
                png,
            )

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
                fig_name = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} on top of Related Unrelated RDM".replace(
                    "  ", " "
                )
                pdf, png, csv = ExportData.create_figure_names(
                    fig_name.replace(" ", "_"), opt=".csv"
                )
                self.export_top_similarities(
                    rsa_result,
                    fig_name,
                    csv,
                    self.Searchlight_Related_Unrelated_RDM.__name__,
                )
                stat, patient = self.get_strategy_patient_names(brain)

                fig_title = f"Searchlight RSA Analysis of Related/Unrelated RDM using {stat} Imputation in {patient}"

                # Searchlight RSA Analysis of Related/Unrelated RDM using MICE imputation in Neurotypicals
                Visualization().plot_brain_image(
                    smoothed_img,
                    fig_title,
                    self.Searchlight_Related_Unrelated_RDM.__name__,
                    pdf,
                    png,
                )

        return results

    def plot_NaNs(self):
        Visualization().visualize_nans(self.brain)

    def export_top_similarities(self, diffrences, title, csv, directory):
        top_n = nlargest(20, diffrences, key=lambda x: x[2])
        # csv_name = ExportData.get_graph_name(".csv", title.replace(" ", "_"))
        directory_path = Helper.ensure_dir("Searchlight_Graphs", directory)
        csv_file = Path(directory_path).joinpath(csv)
        f = open(csv_file, "w")
        f.write(
            f"Spermanr;Sphere_Centres;Voxel_Indicies;AAL_Coordinates;Talairach_Coordinates\n"
        )
        for sph_cntr, vox_indices, r, aal_coors, taal_xyz_coo in top_n:
            coords_maping = []
            for x, y, z in taal_xyz_coo:
                row = self.brain.Talairach_MNI_space[
                    (self.brain.Talairach_MNI_space["TALX"] == x)
                    & (self.brain.Talairach_MNI_space["TALY"] == y)
                    & (self.brain.Talairach_MNI_space["TALZ"] == z)
                ]
                coords_maping.append([(x, y, z), row.iloc[0]["BA"]])
            f.write(f"{r};{sph_cntr};{vox_indices};{aal_coors};{coords_maping}\n")
        f.close()

    def get_strategy_patient_names(self, brain):
        stat = None
        if "mice" in self.rsa_config.strategy:
            stat = "MICE"
        elif "mean" in self.rsa_config.strategy:
            stat = "Mean"
        patient = None
        if "N" in brain.current_labels.name.split("_")[-1]:
            patient = "Neurotypicals"
        elif "S" in brain.current_labels.name.split("_")[-1]:
            patient = "Schizophrenics"
        elif "D" in brain.current_labels.name.split("_")[-1]:
            patient = "Depressives"
        return stat, patient

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

    def Searchlight_brain_difference(self, results):
        difference_results = []
        combinations = itertools.combinations(list(range(len(results))), 2)
        for k, l in combinations:
            diffrences = []
            smoothed_img = image.smooth_img(self.brain.NIfTI, None)
            smoothed_img._dataobj = np.zeros(smoothed_img._dataobj.shape)
            brain_k, smoothed_img_k, rsa_result_k = results[k]
            brain_l, smoothed_img_l, rsa_result_l = results[l]
            assert len(rsa_result_k) == len(rsa_result_l)
            i = 0
            while i < len(rsa_result_k):
                sph_cntr_k, vox_indices_k, r_k, aal_coors_k, taal_xyz_coo_k = (
                    rsa_result_k[i]
                )
                sph_cntr_l, vox_indices_l, r_l, aal_coors_l, taal_xyz_coo_l = (
                    rsa_result_l[i]
                )

                assert sph_cntr_k == sph_cntr_l
                assert vox_indices_k == vox_indices_l
                assert aal_coors_k == aal_coors_l
                assert taal_xyz_coo_k == taal_xyz_coo_l

                r_diff = abs(r_k - r_l)
                for vox_index in vox_indices_k:
                    for aal_coo in aal_coors_k:
                        smoothed_img._dataobj[aal_coo] = r_diff

                diffrences.append(
                    (sph_cntr_k, vox_indices_k, r_diff, aal_coors_k, taal_xyz_coo_l)
                )

                i += 1

            smoothed_img._data_cache = smoothed_img._dataobj
            difference_results.append((smoothed_img, brain_k, brain_l, diffrences))
        return difference_results

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

    def __get_note(self, mean, method):
        return f"Popmean:{mean}, Conditions:{self.data_config.conditions}, PCA fix components:{self.training_config.has_fix_components}, Executed By:{method.__name__}(...)"

    """
    def plot_brain_image(self, smoothed_img, title_txt, directory, show=False):
        # rdm_typ = f"{self.rsa_config.related_unrelated_RDM=}".split("=")[0].split(".")[2]

        directory_path = Helper.ensure_dir("Searchlight_Graphs", directory)
        fig = plt.figure(figsize=(18, 7))
        # display, axes = plotting.plot_img_on_surf( smoothed_img,surf_mesh="fsaverage", views=["lateral", "medial"],hemispheres=["left", "right"],inflate=False,colorbar=True,bg_on_data=True,cmap="hsv_r")
        display = plotting.plot_glass_brain(
            smoothed_img,
            threshold=0,
            # title=title,
            display_mode="lzry",
            colorbar=True,
            figure=fig,
        )
        display.title(
            text=title_txt,
            # x=0.01,
            # y=0.99,
            x=0.02,
            y=0.02,
            size=20,
            # color="green",
            bgcolor=None,
            alpha=1,
            va="bottom",
        )
        # display = plotting.plot_stat_map(smoothed_img, threshold=0)
        # display.savefig("pretty_brain.png")
        # plotting.plot_glass_brain(smoothed_img, threshold=0)
        time.sleep(1)

        pdf_name, png_name = ExportData.create_figure_names(title_txt.replace(" ", "_"))
        # png_name = ExportData.get_file_name(".png", title_txt.replace(" ", "_"))
        # pdf_name = ExportData.get_file_name(".pdf", title_txt.replace(" ", "_"))

        directory_path = Helper.ensure_dir("Searchlight_Graphs", directory)
        png_path = Path(directory_path).joinpath(png_name)
        pdf_path = Path(directory_path).joinpath(pdf_name)
        plt.savefig(pdf_path)
        time.sleep(1)
        plt.savefig(png_path, dpi=600)

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
    """
    """
    def plot_detailed_bars_data_labels(self, directory, all_export_data):
        #This method plot the detailed graphs, with Image labels and subject labels, std and significant
        nested_dict = self.groupby_strategy(all_export_data)

        subject_l = "Subject"
        image_l = "Speech-Gesture"

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
                    f"Subjects {chr(10)} (N, D, S)",
                    f"Speech-Gesture {chr(10)} Combinations {chr(10)}(AR, AU, CR, CU)",
                ],
                legend_font=13,
                legend_title="Conditions",
            )
    """
    """
    def plot_detailed_bars(
        self,
        directory,
        all_export_data,
        N="N",
        D="D",
        S="S",
        legend_text=["Neurotypical", "Depressive", "Schizophrenia"],
    ):
        #This method plot the detailed graphs, with binary 6 combinations, std and significant
        
        nested_dict = self.groupby_strategy(all_export_data)

        # N = self.data_config.neurotypical
        # D = self.data_config.depressive_disorder
        # S = self.data_config.schizophrenia_spectrum

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_patients(N, D, S, clasiifiers)

            models = []
            for label, y in bar_dict.items():
                for j, v in y.items():
                    if j not in models:
                        models.append(j)
                    for i in v:
                        i.column_name = i.column_name.split("_")[-1]

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
                strategy,
                models,
                data_stat,
                directory,
                legends=legend_text,
                legend_font=13,
            )
    """
    """
    def plot_images(
        self,
        directory,
        all_export_data,
        N="N",
        D="D",
        S="S",
        legend_title="Mental Disorders",
        legend_text=[
            "Neurotypical",
            "Depressive",
            "Schizophrenia",
        ],
    ):
        nested_dict = self.groupby_strategy(all_export_data)

        for strategy, clasiifiers in nested_dict.items():
            bar_dict = self.separate_results_by_patients(N, D, S, clasiifiers)

            models, bar_dictc = self.merge_results(N, D, S, bar_dict)

            self.plot_diagram(
                strategy,
                models,
                bar_dictc,
                directory,
                legend_title=legend_title,
                legend_text=legend_text,
            )
    """
    """
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
        plt.subplots(figsize=(25, 10))
        plt.rcParams.update({"ytick.labelsize": 18})
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
                txt_pos = 0.5 * d
                if "Speech-Gesture" in bar_labels[i]:
                    txt_pos = 0.6 * d
                plt.text(
                    br_position[index],
                    txt_pos,
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
            lobe_n = "Whole Brain"
        else:
            lobe_n = f"{lobe_n} Lobe"

        sta_name = strategy

        if "mice" in sta_name:
            sta_name = "MICE Imputation"
        elif "mean" in sta_name:
            sta_name = "Mean Imputation"
        elif "remove" in sta_name:
            sta_name = "Voxel Deletion"

        title = f"{lobe_n} Analysis with {sta_name}"

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

        pdf_name, png_name = ExportData.create_figure_names(gname)
        # png_name = ExportData.get_file_name(".png", gname)
        # pdf_name = ExportData.get_file_name(".pdf", gname)

        directory_path = Helper.ensure_dir("Ml_Graphs", directory)
        png_path = Path(directory_path).joinpath(png_name)
        pdf_path = Path(directory_path).joinpath(pdf_name)
        plt.savefig(pdf_path)
        time.sleep(1)
        plt.savefig(png_path, dpi=600)

        # plt.show()
        plt.close()
    """
    """
    def plot_diagram(
        self,
        strategy,
        models,
        bar_dictc,
        directory,
        legend_title,
        legend_text,
    ):
        barWidth = 0.25
        i = 0
        br_pre = None
        colors = ["tomato", "limegreen", "dodgerblue"]
        br_p = None
        plt.subplots(figsize=(25, 10))
        plt.rcParams.update({"ytick.labelsize": 18})
        plt.rcParams.update({"legend.title_fontsize": 15})
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

        lobe_n = self.brain.lobe.name
        if "All" in lobe_n:
            lobe_n = "Whole Brain"
        else:
            lobe_n = f"{lobe_n} Lobe"

        sta_name = strategy
        if "mice" in sta_name:
            sta_name = "MICE Imputation"
        elif "mean" in sta_name:
            sta_name = "Mean Imputation"
        elif "remove" in sta_name:
            sta_name = "Voxel Deletion"

        name = f"{lobe_n} Analysis with {sta_name}"

        plt.xlabel(name, fontweight="bold", fontsize=22)
        plt.ylabel("Accuracy", fontweight="bold", fontsize=15)
        plt.xticks([r + barWidth for r in range(len(br_p))], models, fontsize=20)
        # plt.legend(legend_bars,legends,fontsize=13,loc="upper left",bbox_to_anchor=(1, 1),title=legend_title,)
        # plt.legend(fontsize=18, title="Mental Disorders", loc="upper right")
        l = plt.legend(
            fontsize=13,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title=legend_title,
        )

        for lidx, text in enumerate(legend_text):
            l.get_texts()[lidx].set_text(text)

        gname = f"{self.brain.lobe.name}_{strategy}_{self.unary_subject_binary_image_classification.__name__}"

        pdf_name, png_name = ExportData.create_figure_names(gname)
        # png_name = ExportData.get_file_name(".png", gname)
        # pdf_name = ExportData.get_file_name(".pdf", gname)

        directory_path = Helper.ensure_dir("Ml_Graphs", directory)
        png_path = Path(directory_path).joinpath(png_name)
        pdf_path = Path(directory_path).joinpath(pdf_name)
        plt.savefig(pdf_path)
        time.sleep(1)
        plt.savefig(png_path, dpi=600)

        # plt.show()
        plt.close()
    """
    """
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
    """

    """
    def separate_results_by_labels(self, subject_l, image_l, clasiifiers):
        bar_dict = {
            subject_l: {},
            image_l: {},
        }
        for classifier, list_per_classifier in clasiifiers.items():
            for p in list_per_classifier:
                if "subject" in p.column_name:
                    if bar_dict[subject_l].get(classifier) is None:
                        bar_dict[subject_l][classifier] = [p]
                    else:
                        bar_dict[subject_l][classifier].append(p)
                if "image" in p.column_name:
                    if bar_dict[image_l].get(classifier) is None:
                        bar_dict[image_l][classifier] = [p]
                    else:
                        bar_dict[image_l][classifier].append(p)
        return bar_dict
    """

    """
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
    """
    """
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
    """
