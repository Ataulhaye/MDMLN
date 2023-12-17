class BrainDataConfig:
    def __init__(self):
        self.conditions = 4

        self.neurotypical_patients = 43
        self.depressive_disorder_patients = 33
        self.schizophrenia_spectrum_patients = 46

        # subject labels
        self.neurotypical = "N"
        self.depressive_disorder = "D"
        self.schizophrenia_spectrum = "S"

        # subject labels
        self.neurotypical_int = 0
        self.depressive_disorder_int = 1
        self.schizophrenia_spectrum_int = 2

        # image labels
        self.abstract_related = "AR"
        self.abstract_unrelated = "AU"
        self.concrete_related = "CR"
        self.concrete_unrelated = "CU"

        # image labels
        self.abstract_related_int = 0
        self.abstract_unrelated_int = 1
        self.concrete_related_int = 2
        self.concrete_unrelated_int = 3

        # this order has to be maintained
        self.patients = [
            self.neurotypical_patients,
            self.depressive_disorder_patients,
            self.schizophrenia_spectrum_patients,
        ]

        self.subject_labels = [
            self.neurotypical,
            self.depressive_disorder,
            self.schizophrenia_spectrum,
        ]

        self.image_labels = [
            self.abstract_related,
            self.abstract_unrelated,
            self.concrete_related,
            self.concrete_unrelated,
        ]
        self.subject_label_popmean = round(((100 / len(self.subject_labels)) / 100), 2)
        self.image_label_popmean = round(((100 / len(self.image_labels)) / 100), 2)

        self.STG_path = "./left_STG_MTG_AALlable_ROI.rex.mat"
        self.STG = "STG"

        self.IFG_path = "./ROI_aal_wfupick_left44_45.rex.mat"
        self.IFG = "IFG"
