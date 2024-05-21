from pathlib import Path

import nilearn.decoding
import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.image import get_data, index_img, load_img, new_img_like
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_img, plot_stat_map, show
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from DataTraining import DataTraining
from TrainingConfig import TrainingConfig

brain_area = "all_lobes"

bd_config = BrainDataConfig()
data_path = ""
area = ""
if "IFG" in brain_area:
    data_file = "ROI_aal_wfupick_left44_45.rex.mat"
    area = bd_config.IFG
elif "all_lobes" in brain_area:
    data_file = r"AAL_all_lobes_ROI.rex.mat"
    area = bd_config.all_lobes
else:
    data_file = "left_STG_MTG_AALlable_ROI.rex.mat"
    area = bd_config.STG

data_path = Path(data_file).absolute()

brain = Brain(
    area=area,
    # data_path=bd_config.STG_path,
    data_path=data_path,
)
brain.current_labels = brain.subject_labels_int


nni_path = Path("All_Brain_Data_Raw//m1.nii").absolute()
nni_gz_path = Path("All_Brain_Data_Raw//m1.nii.gz").absolute()


brain.mask = nni_gz_path
brain.niimg = nni_path
# Prepare masks
mask_img = load_img(brain.mask)

# how to get the 4d image?
# fmri_img = index_img(r"C:\Users\ataul\source\Uni\BachelorThesis\poc\All_Brain_Data_Raw\AAL_all_lobes_ROI.nii.gz",brain.current_labels.labels,)
# fmri_img = index_img(brain.mask, brain.current_labels.labels)

# Searchlight computation
# Make processing parallel
# /!\ As each thread will print its progress, n_jobs > 1 could mess up the
#     information output.
n_jobs = 2

# Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the run, which corresponds to
# splitting the samples in 4 folds and make 4 runs using each fold as a test
# set once and the others as learning sets


cv = KFold(n_splits=4)


# The radius is the one of the Searchlight sphere that will scan the volume
searchlight = nilearn.decoding.SearchLight(
    mask_img,
    process_mask_img=None,
    radius=5.6,
    n_jobs=n_jobs,
    verbose=1,
    cv=cv,
)


# smoothed_img = image.smooth_img(brain.niimg, None)
# searchlight.fit(smoothed_img, brain.current_labels.labels)


searchlight.fit(brain.niimg, brain.current_labels.labels)

print("End")
