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
        else:
            self.classifiers = classifiers

        if strategies is None:
            self.strategies = ["mice", "mean", "remove_voxels"]
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
        optional_txt = None
        if self.training_config.dimension_reduction == True:
            optional_txt = "PCA"
        if self.training_config.use_autoencoder == True:
            optional_txt = "Autoencoder"
        xl_name = f"{self.brain.lobe.name}-Results"

        if optional_txt is not None:
            xl_name = f"{optional_txt}_{xl_name}"

        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=xl_name,
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
            transpose=True,
            single_label=True,
        )

        visu = Visualization()
        visu.plot_merged_bars(
            directory=self.unary_subject_binary_image_classification.__name__,
            all_data=all_export_data,
            lobe_name=self.brain.lobe.name,
            opt_info=optional_txt,
            legend_font=18,
        )
        time.sleep(5)
        visu.plot_detailed_bars(
            directory=self.unary_subject_binary_image_classification.__name__,
            all_data=all_export_data,
            lobe_name=self.brain.lobe.name,
            opt_info=optional_txt,
            legend_font=18,
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
        optional_txt = None
        if self.training_config.dimension_reduction == True:
            optional_txt = "PCA"
        if self.training_config.use_autoencoder == True:
            optional_txt = "Autoencoder"
        xl_name = f"{self.brain.lobe.name}-Results"

        if optional_txt is not None:
            xl_name = f"{optional_txt}_{xl_name}"
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=xl_name,
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            directory=directory_path,
            transpose=True,
            single_label=True,
        )

        visu = Visualization()
        visu.plot_merged_bars(
            self.binary_subject_binary_image_classification.__name__,
            all_data,
            self.brain.lobe.name,
            N="N-D",
            D="N-S",
            S="D-S",
            legend_text=[
                f"Neurotypicals & {chr(10)} Depressives",
                f"Neurotypicals & {chr(10)} Schizophrenics",
                f"Depressives & {chr(10)} Schizophrenics",
            ],
            opt_info=optional_txt,
            legend_font=17,
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
                f"Neurotypicals & {chr(10)} Depressives",
                f"Neurotypicals & {chr(10)} Schizophrenics",
                f"Depressives & {chr(10)} Schizophrenics",
            ],
            opt_info=optional_txt,
            legend_font=17,
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

        with open(
            "subject_and_image_classification_autoencoder.pickle", "wb"
        ) as output:
            pickle.dump(export_data, output)

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        note = export.create_note([self.training_config, ""])
        directory_path = Helper.ensure_dir(
            "Automated_Excel_files", self.subject_and_image_classification.__name__
        )
        optional_txt = None
        if self.training_config.dimension_reduction == True:
            optional_txt = "PCA"
        if self.training_config.use_autoencoder == True:
            optional_txt = "Autoencoder"
        xl_name = f"{self.brain.lobe.name}-Results"

        if optional_txt is not None:
            xl_name = f"{optional_txt}_{xl_name}"

        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=xl_name,
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
                fig_name = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} Text Embeddings RDM {set_name} {query}".replace(
                    "  ", " "
                )
                pdf, png, csv = ExportData.create_figure_names(
                    fig_name.replace(" ", "_"), opt=".csv"
                )
                self.export_top_similarities(
                    rsa_result,
                    fig_name,
                    csv,
                    self.Searchlight_Text_Embeddings.__name__,
                )
                stat, patient = self.get_strategy_patient_names(brain)
                fig_title = f"Searchlight RSA Analysis of Audio Embeddings with {stat} Imputation in {patient}"

                Visualization().plot_brain_image(
                    smoothed_img,
                    fig_title,
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
            fig_name = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} Text Embeddings RDM {set_name} {query}".replace(
                "  ", " "
            )
            pdf, png, csv = ExportData.create_figure_names(
                fig_name.replace(" ", "_"), opt=".csv"
            )
            self.export_top_similarities(
                diffrences,
                fig_name,
                csv,
                self.Searchlight_brain_difference_Text_Embeddings_RDM.__name__,
            )
            stat, patient_k = self.get_strategy_patient_names(brain_k)
            stat, patient_l = self.get_strategy_patient_names(brain_l)

            fig_title = f"{patient_k[:-1]}-{patient_l[:-1]} Differences in Auditory Processing with {stat} Imputation"

            Visualization().plot_brain_image(
                smoothed_img,
                fig_title,
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
                fig_name = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} Video Embeddings RDM {set_name} {query}".replace(
                    "  ", " "
                )
                pdf, png, csv = ExportData.create_figure_names(
                    fig_name.replace(" ", "_"), opt=".csv"
                )
                self.export_top_similarities(
                    rsa_result,
                    fig_name,
                    csv,
                    self.Searchlight_Video_Embeddings.__name__,
                )

                stat, patient = self.get_strategy_patient_names(brain)
                fig_title = f"Searchlight RSA Analysis of Video Embeddings with {stat} Imputation in {patient}"

                Visualization().plot_brain_image(
                    smoothed_img,
                    fig_title,
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
            title = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} Video Embeddings RDM {set_name} {query}".replace(
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
            stat, patient_k = self.get_strategy_patient_names(brain_k)
            stat, patient_l = self.get_strategy_patient_names(brain_l)

            fig_title = f"{patient_k[:-1]}-{patient_l[:-1]} Differences in Visual Processing with {stat} Imputation"

            Visualization().plot_brain_image(
                smoothed_img,
                fig_title,
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
                fig_name = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} Bridge Embeddings RDM {set_name} {query}".replace(
                    "  ", " "
                )
                pdf, png, csv = ExportData.create_figure_names(
                    fig_name.replace(" ", "_"), opt=".csv"
                )
                self.export_top_similarities(
                    rsa_result,
                    fig_name,
                    csv,
                    self.Searchlight_Bridge_Embeddings.__name__,
                )
                stat, patient = self.get_strategy_patient_names(brain)
                fig_title = f"Searchlight RSA Analysis of Bridge Embeddings with {stat} Imputation in {patient}"

                Visualization().plot_brain_image(
                    smoothed_img,
                    fig_title,
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
            fig_name = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} Bridge Embeddings RDM {set_name} {query}".replace(
                "  ", " "
            )
            pdf, png, csv = ExportData.create_figure_names(
                fig_name.replace(" ", "_"), opt=".csv"
            )
            self.export_top_similarities(
                diffrences,
                fig_name,
                csv,
                self.Searchlight_brain_difference_Bridge_Embeddings_RDM.__name__,
            )

            stat, patient_k = self.get_strategy_patient_names(brain_k)
            stat, patient_l = self.get_strategy_patient_names(brain_l)

            fig_title = f"{patient_k[:-1]}-{patient_l[:-1]} Differences in Audiovisual Processing with {stat} Imputation"

            Visualization().plot_brain_image(
                smoothed_img,
                fig_title,
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
                fig_name = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} Abstract Concrete RDM".replace(
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

                fig_title = f"Searchlight RSA Analysis of Abstract/Concrete RDM with {stat} Imputation in {patient}"
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
            fig_name = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} Abstract Concrete RDM".replace(
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
            # Abstractness = abstract/concrete
            fig_title = f"{patient_k[:-1]}-{patient_l[:-1]} Differences in Conceptual Abstractness with {stat} Imputation"

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
            fig_name = f"{self.lobe_name(self.brain)} difference between {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} {self.is_normalized()} Related Unrelated RDM".replace(
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
            fig_title = f"{patient_k[:-1]}-{patient_l[:-1]} Differences in Conceptual Relatedness with {stat} Imputation"

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
                fig_name = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} Related Unrelated RDM".replace(
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

                fig_title = f"Searchlight RSA Analysis of Related/Unrelated RDM with {stat} Imputation in {patient}"

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
