import copy

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from DataTraining import DataTraining
from Enums import Lobe
from ExportData import ExportData
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
                load_labels=True,
                load_int_labels=True,
            )
        elif lobe is Lobe.IFG:
            self.brain = Brain(
                area=self.data_config.IFG,
                data_path=self.data_config.IFG_path,
                load_labels=True,
                load_int_labels=True,
            )
        else:
            self.brain = brain

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
