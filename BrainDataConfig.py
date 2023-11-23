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

        # image labels
        self.abstract_related = "AR"
        self.abstract_unrelated = "AU"
        self.concrete_related = "CR"
        self.concrete_unrelated = "CU"

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
