import nilearn as ni
import numpy as np
import pandas as pd
import scipy.io
import torch
from nilearn import image, plotting
from scipy.spatial import KDTree
from scipy.stats import spearmanr

from BrainDataConfig import BrainDataConfig


class SearchLight:

    def make_RDMs(self, data):
        rdms = []
        for i in range(data.shape[0]):
            rdms.append(1 - torch.corrcoef(torch.from_numpy(data[i])))
        return np.stack(rdms, axis=0)

    def get_sphere_centers(self, df, radius=7):

        bdc = BrainDataConfig()

        talx = df[bdc.TALX].tolist()
        taly = df[bdc.TALY].tolist()
        talz = df[bdc.TALZ].tolist()
        minx = min(talx)
        maxx = max(talx)
        miny = min(taly)
        maxy = max(taly)
        minz = min(talz)
        maxz = max(talz)

        sphere_centers = []
        for xx in range(minx, maxx, radius):
            for yy in range(miny, maxy, radius):
                for zz in range(minz, maxz, radius):
                    sphere_centers.append(np.array([xx, yy, zz]))

        return np.array(sphere_centers)
