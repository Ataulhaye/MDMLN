import copy
import itertools
import pickle
import statistics

import matplotlib.pyplot as plt
import nibabel as nib
import nilearn as ni
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import torch
from nilearn import datasets, image, plotting
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
        rsa_config: RSAConfig = None,
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
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
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
        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()

        print_config = self.__get_note(
            stg_image_binary_data[0].current_labels.popmean,
            self.binary_image_classification,
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
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
        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.unary_subject_binary_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.binary_subject_concatenated_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.binary_subject_binary_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.binary_subject_unary_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
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
        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(
            mean, self.subject_concatenated_image_classification
        )
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        print_config = self.__get_note(mean, self.subject_binary_image_classification)
        note = export.create_note([self.training_config, print_config])
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
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
            self.training_config.lobe = self.brain.lobe.name
            export = ExportData()
            note = export.create_note([self.training_config])
            export.create_and_write_datasheet(
                data=export_data,
                sheet_name=f"{self.brain.lobe.name}-Results",
                title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds-{split}-Clf",
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
            self.training_config.lobe = self.brain.lobe.name
            export = ExportData()
            note = export.create_note([self.training_config])
            export.create_and_write_datasheet(
                data=export_data,
                sheet_name=f"{self.brain.lobe.name}-Results",
                title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds-{split}-Clf",
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

        self.training_config.lobe = self.brain.lobe.name
        export = ExportData()
        note = export.create_note([self.training_config, ""])
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{self.brain.lobe.name}-Results",
            title=f"{self.brain.lobe.name}-{self.training_config.folds}-Folds",
            notes=note,
            transpose=True,
        )

    def RSA_Audio_RDM(self, plotting=True):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """
        self.brain.current_labels = self.brain.subject_labels

        if self.rsa_config.normalize:
            x = self.brain.normalize_whole_data(self.brain.voxels)
            self.brain.voxels = x

        file_name = f"{self.RSA_Audio_RDM.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_Normalized-{str(self.rsa_config.normalize)}.pickle"

        results = None
        try:
            results = pickle.load(open(file_name, "rb"))
        except FileNotFoundError as err:
            print("There are no saved RSA results. RSA function will be executed.", err)

        if results is None:
            subject_unary_data = self.brain.unary_fmri_subject_or_image(
                self.data_config
            )
            results = RepresentationalSimilarityAnalysis().run_RSA(
                subject_unary_data,
                self.rsa_config.audio_RDM,
                self.rsa_config.radius,
                self.Talairach_MNI_space,
                self.brain.NIfTI,
            )
            with open(file_name, "wb") as output:
                pickle.dump(results, output)

        for brain, smoothed_img, rsa_result in results:
            if plotting:
                title = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} Audio RDM".replace(
                    "  ", " "
                )
                self.plot_brain_image(smoothed_img, title)

        return results

    def RSA_brain_difference_Audio_RDM(self):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """

        results = self.RSA_Audio_RDM(plotting=False)

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

            title = f"{self.lobe_name(self.brain)} difference of {self.is_normalized()} {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} on top of Audio RDM".replace(
                "  ", " "
            )
            self.plot_brain_image(smoothed_img, title)

    def RSA_brain_difference_related_unrelated_RDM(self):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """

        results = self.RSA_related_unrelated_RDM(plotting=False)

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

            title = f"{self.lobe_name(self.brain)} difference of {self.is_normalized()} {brain_k.current_labels.name.split('_')[-1]} and {brain_l.current_labels.name.split('_')[-1]} on top of related unrelated RDM".replace(
                "  ", " "
            )
            self.plot_brain_image(smoothed_img, title)

    def RSA_related_unrelated_RDM(self, plotting=False):
        """
        Unarize the fMRI data based on subjects, then for every unarized instance RSA takes place
        i.e unary_subject_labels_N
        """
        self.brain.current_labels = self.brain.subject_labels

        if self.rsa_config.normalize:
            x = self.brain.normalize_whole_data(self.brain.voxels)
            self.brain.voxels = x

        file_name = f"{self.RSA_related_unrelated_RDM.__name__}_{self.brain.lobe.name}_{self.brain.current_labels.name}_Normalized-{str(self.rsa_config.normalize)}.pickle"

        results = None
        try:
            results = pickle.load(open(file_name, "rb"))
        except FileNotFoundError as err:
            print("There are no saved RSA results. RSA function will be executed.", err)

        if results is None:
            subject_unary_data = self.brain.unary_fmri_subject_or_image(
                self.data_config
            )
            results = RepresentationalSimilarityAnalysis().run_RSA(
                subject_unary_data,
                self.rsa_config.audio_RDM,
                self.rsa_config.radius,
                self.Talairach_MNI_space,
                self.brain.NIfTI,
            )
            with open(file_name, "wb") as output:
                pickle.dump(results, output)

        for brain, smoothed_img, rsa_result in results:
            if plotting:
                title = f"{self.lobe_name(brain)} {self.is_normalized()} {brain.current_labels.name.split('_')[-1]} Related Unrelated RDM".replace(
                    "  ", " "
                )
                self.plot_brain_image(smoothed_img, title)

        return results

    def plot_brain_image(self, smoothed_img, title):
        # rdm_typ = f"{self.rsa_config.related_unrelated_RDM=}".split("=")[0].split(".")[2]
        # atlas = datasets.fetch_atlas_talairach("ba")

        display = plotting.plot_glass_brain(
            smoothed_img, threshold=0, title=title, display_mode="lzry"
        )
        # display = plotting.plot_stat_map(smoothed_img, threshold=0)
        # display.savefig("pretty_brain.png")
        # plotting.plot_glass_brain(smoothed_img, threshold=0)
        plotting.show()

        graph_name = ExportData.get_file_name(".png", title.replace(" ", "_"))
        plt.savefig(graph_name)
        display.close()
        plt.close()
        # display = plotting.plot_glass_brain(None, plot_abs=False, display_mode="lzry", title=title)
        # display.add_contours(smoothed_img, filled=True)
        # plotting.show()

        # graph_name = ExportData.get_file_name(".png", title.replace(" ", "_"))
        # plt.savefig(graph_name)
        # display.close()
        # plt.close()

    def is_normalized(self):
        normalized = ""
        if self.rsa_config.normalize:
            normalized = "Normalized"
        return normalized

    def lobe_name(self, brain):
        lobe = brain.lobe.name
        if brain.lobe == Lobe.ALL:
            lobe = f"{lobe} Lobes"
        return lobe

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

            self.plot_diagram_per_strategy(strategy, models, data_stat)

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

    def plot_diagram_per_strategy(self, strategy, models, bar_data, patients=3):
        barWidth = 0.5
        i = 0
        br_pre_pos = None
        all_br_positions = []
        color_len = int(len(bar_data) / patients)
        colors = ["tomato" for x in range(color_len)]
        colors.extend(["limegreen" for x in range(color_len)])
        colors.extend(["dodgerblue" for x in range(color_len)])

        bar_labels = [
            j.split("_")[-1] for i in range(len(models)) for j in list(bar_data.keys())
        ]

        br_position = None
        legend_bars = []
        fig, ax = plt.subplots(figsize=(25, 10))
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
            if i % (int(len(bar_data) / patients)) == 0:
                legend_bars.append(a)
            i = i + 1
            # Adding Xticks
        name = f"{self.brain.lobe.name} Results, {strategy} as data normalization"

        plt.xlabel(name, fontweight="bold", fontsize=15)
        plt.ylabel("Accuracy", fontweight="bold", fontsize=15)

        all_br_positions.sort()
        tick_pos = []
        bars_per_model = int((len(all_br_positions) / len(models)))
        end = 0
        start = 0
        while end < len(all_br_positions):
            end = end + bars_per_model
            seg = all_br_positions[start:end]
            tick_pos.append(seg[int(len(seg) / 2)] - barWidth / 2)
            start = end

        plt.xticks(tick_pos, models, fontsize=20)

        plt.legend(legend_bars, ["N", "D", "S"], fontsize=18, title="Mental Disorders")
        # plt.legend(legend_bars, ["N", "D", "S"], fontsize=18, loc='upper left', bbox_to_anchor=(1, 1) ,title_fontsize=14,title="Mental Disorders")
        gname = f"{self.brain.lobe.name}_{strategy}_{self.unary_subject_binary_image_classification.__name__}"
        graph_name = ExportData.get_file_name(".png", gname)
        # plt.savefig(graph_name, dpi=1200)
        plt.savefig(graph_name)
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
        name = f"{self.brain.lobe.name} Results, {strategy} as Norm."

        plt.xlabel(name, fontweight="bold", fontsize=15)
        plt.ylabel("Accuracy", fontweight="bold", fontsize=15)
        plt.xticks(
            [r + barWidth for r in range(len(br_p))],
            models,
        )
        plt.legend()
        gname = f"{self.brain.lobe.name}_{strategy}_{self.unary_subject_binary_image_classification.__name__}"
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
