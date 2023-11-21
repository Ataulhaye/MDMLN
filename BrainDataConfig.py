class BrainDataConfig:
    def __init__(self):
        self.trails = 4

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
