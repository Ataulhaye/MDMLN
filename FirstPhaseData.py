import numpy as np
import plotly.express as px
import scipy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# left IFG (BA44/45; 523 voxels) and the left STG/MTG (7238 voxels) as both are relevant
# for the task but the left IFG should differentiate between groups (N,D,S)
# 43 N 33 D 46 S x 4 images [AR, AU, CR, CU]
print((43 + 33 + 46) * 4)

# Labels
subject_labels = ["N"] * 4 * 43 + ["D"] * 4 * 33 + ["S"] * 4 * 46
# subject_labels 172=N, 132=D, 184=S sumup to 488
image_labels = ["AR", "AU", "CR", "CU"] * (43 + 33 + 46)

# Ata:
# Test different classifiers with the MTG and IFG
# Try to classify the subjects labes and image labels
# We predict IFG can classify both, MTG only IMG label
# Tuthera:
# Download and familiarize yourself with Net2Brain and RSA
# claculate RDMs for each Subject + image condition pair and each ROI
# Calculate correlation (do RSA) using the RDMs

# Tuhera: Unraveling The Computational Basis Of Mental Disorders Using Task Specific Artificial Nueral NEtworks
# Ata: Using Machine Learning to Diagnose Mental Disorders from Neuroimaging Data


allL = [sb + im for sb, im in zip(subject_labels, image_labels)]

print(set(allL))

STG_data = scipy.io.loadmat("./left_STG_MTG_AALlable_ROI.rex.mat")

STG_data["R"].shape  # (488, 7238)

STG_raw = STG_data["R"]

STG_without_nan = STG_raw[:, ~np.isnan(STG_raw).any(axis=0)]


rows = []
for row in STG_raw:
    print(len(row))
    row = [x for x in row if ~np.isnan(x)]
    rows.append(row)
    print(len(row))
    print("----------------")

# STG = np.array(
#    rows,
# )


# what are exactly AR, AU, CR and CU
# 172, 132 and 184
# nans are errors or?
# hardcoded means it known for that data

IFG = scipy.io.loadmat("./ROI_aal_wfupick_left44_45.rex.mat")

IFG["R"].shape
