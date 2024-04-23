from AnalyserBase import AnalyserBase
from Brain import Brain
from BrainDataConfig import BrainDataConfig
from DataTraining import DataTraining
from ExportData import ExportData
from TrainingConfig import TrainingConfig


class STGAnalyser(AnalyserBase):

    def __init__(
        self,
        brain: Brain = None,
        classifiers=None,
        strategies=None,
        training_config: TrainingConfig = None,
        data_config: BrainDataConfig = BrainDataConfig(),
    ):
        if brain is None:
            self.brain = Brain(
                area=data_config.STG,
                data_path=data_config.STG_path,
                load_labels=True,
                load_int_labels=True,
            )
        else:
            self.brain = brain

        super(STGAnalyser, self).__init__(
            self.brain, classifiers, strategies, training_config, data_config
        )

    def stg_subject_binary_classification(self):

        split = "r_split"
        if self.training_config.predefined_split:
            split = "cr_split"

        self.brain.current_labels = self.brain.subject_labels
        stg_subject_binary_data = self.brain.binary_data(self.data_config)

        # self.training_config.dimension_reduction = True
        # t_config.explain = True
        # t_config.folds = 1
        # t_config.predefined_split = False
        all_export_data = []

        for bd in stg_subject_binary_data:
            training = DataTraining()
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
            )
            all_export_data.extend(export_data)
        self.training_config.brain_area = self.brain.area
        export = ExportData()
        note = export.create_note(self.training_config)
        export.create_and_write_datasheet(
            data=all_export_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def stg_concatenated_trails_classification(self):

        self.brain.current_labels = self.brain.subject_labels
        split = "r_split"
        if self.training_config.predefined_split:
            split = "cr_split"

        stg_subject_binary_data = self.brain.concatenate_fmri_data_trails()

        self.training_config.analyze_concatenated_trails = True
        self.training_config.analyze_binary_trails = False

        all_data = []

        for bd in stg_subject_binary_data:
            training = DataTraining()
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
            )
            all_data.extend(export_data)
        self.training_config.brain_area = self.brain.area
        export = ExportData()
        note = export.create_note(self.training_config)
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def stg_binary_trails_classification(self):
        self.brain.current_labels = self.brain.subject_labels
        split = "r_split"
        if self.training_config.predefined_split:
            split = "cr_split"

        stg_subject_binary_data = self.brain.binary_fmri_data_trails()

        self.training_config.analyze_binary_trails = True
        self.training_config.analyze_concatenated_trails = False

        all_data = []

        for bd in stg_subject_binary_data:
            training = DataTraining()
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
            )
            all_data.extend(export_data)

        self.training_config.brain_area = self.brain.area
        export = ExportData()
        note = export.create_note(self.training_config)
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def stg_concatenated_trails_classification(self):
        self.brain.current_labels = self.brain.subject_labels
        split = "r_split"
        if self.training_config.predefined_split:
            split = "cr_split"

        stg_subject_binary_data = self.brain.concatenate_fmri_data_trails()

        self.training_config.analyze_binary_trails = False
        self.training_config.analyze_concatenated_trails = True

        all_data = []

        for bd in stg_subject_binary_data:
            training = DataTraining()
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
            )
            all_data.extend(export_data)
        self.training_config.brain_area = self.brain.area
        export = ExportData()
        note = export.create_note(self.training_config)
        export.create_and_write_datasheet(
            data=all_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
            single_label=True,
        )

    def stg_binary_classification_with_shap(self):

        split = "r_split"
        if self.training_config.predefined_split:
            split = "cr_split"

        self.brain.current_labels = self.brain.subject_labels_int
        stg_subject_binary_data = self.brain.binary_data(self.data_config)

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
            )
            self.training_config.brain_area = self.brain.area
            export = ExportData()
            note = export.create_note(self.training_config)
            export.create_and_write_datasheet(
                data=export_data,
                sheet_name=f"{self.brain.area}-Results",
                title=f"{self.brain.area}-{self.training_config.folds}-Folds-{split}-Clf",
                notes=note,
                transpose=True,
            )
        self.brain.current_labels = self.brain.image_labels_int
        stg_image_binary_data = self.brain.binary_data(self.data_config)

        for bd in stg_image_binary_data:
            training = DataTraining()
            export_data = training.brain_data_classification(
                bd,
                self.training_config,
                self.strategies,
                self.classifiers,
            )
            self.training_config.brain_area = self.brain.area
            export = ExportData()
            note = export.create_note(self.training_config)
            export.create_and_write_datasheet(
                data=export_data,
                sheet_name=f"{self.brain.area}-Results",
                title=f"{self.brain.area}-{self.training_config.folds}-Folds-{split}-Clf",
                notes=note,
                transpose=True,
            )

    def stg_classification(self):
        self.brain.current_labels = self.brain.subject_labels_int

        training = DataTraining()

        export_data = training.brain_data_classification(
            self.brain, self.training_config, self.strategies, self.classifiers
        )

        self.brain.current_labels = self.brain.image_labels_int
        e_data = training.brain_data_classification(
            self.brain, self.training_config, self.strategies, self.classifiers
        )
        export_data.extend(e_data)

        split = "r_split"
        if self.training_config.predefined_split:
            split = "cr_split"

        self.training_config.brain_area = self.brain.area
        export = ExportData()
        note = export.create_note(self.training_config)
        export.create_and_write_datasheet(
            data=export_data,
            sheet_name=f"{self.brain.area}-Results",
            title=f"{self.brain.area}-{self.training_config.folds}-Folds-{split}-Clf",
            notes=note,
            transpose=True,
        )
