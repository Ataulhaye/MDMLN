import itertools
import pickle
from heapq import nlargest

import nilearn as ni
import numpy as np
import pandas as pd
import scipy.io
import torch
from nilearn import image, plotting
from scipy.spatial import KDTree
from scipy.stats import spearmanr

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from RSAConfig import RSAConfig


class RepresentationalSimilarityAnalysis:

    def RSA(self, fmri_data: list[Brain], RDM, radius, coordinates):
        # Creating the sphere centers with some radius

        sphere_centers = self.get_sphere_centers(coordinates, radius)
        # creating xyz points by providing the TALX, TALY, and TALZ
        xyz_points = np.array(coordinates[["TALX", "TALY", "TALZ"]])

        kdtree = KDTree(xyz_points)
        # getting the sphere voxels
        sphere_vox = kdtree.query_ball_point(sphere_centers, radius)
        # this will remove the empty spheres and map to the sphere centres and return the list of tuples (sphere_centre_dims, sphere_Voxels)
        # (The voxel dimension in the brain and the voxel indices)
        final_spheres = [
            (sphere_centers[i].tolist(), j) for i, j in enumerate(sphere_vox) if j
        ]

        all_r_means = []
        # subject_unary_data list conatins all three subject voxels. N, D, and S
        for un_brain in fmri_data:
            r_means = []
            for sphere in final_spheres:
                # combine voxels and sphere: voxels for a specific coordinates sphere[-1] has the voxels indices
                voxels = un_brain.voxels[:, sphere[-1]]
                new_shape = (
                    int(voxels.shape[0] / 4),
                    4,
                    voxels.shape[1],
                )
                rvoxels = np.reshape(voxels, new_shape)
                # make RDM from combined voxels and sphere
                RDMs = self.make_RDMs(rvoxels)
                # calculating the rank
                r = [
                    spearmanr(RDM.ravel(), RDMs[i].ravel()).statistic
                    for i in range(RDMs.shape[0])
                ]
                # saving the rank per sphere as mean
                r_means.append(
                    (un_brain.current_labels.name, np.nanmean(r), sphere, rvoxels)
                )
            all_r_means.append(r_means)
        return all_r_means

    def generate_RSA_results(self, r_means):
        k_max = []
        for means in r_means:
            k_max.append(nlargest(10, means))

        combis = itertools.combinations(list(range(len(r_means))), 2)

        diff_means = []
        length = len(r_means[0])
        for comb in combis:
            m_means = []
            i = 0
            while i < length:
                m_means.append(
                    (
                        f"{r_means[comb[0]][i][0]} - {r_means[comb[1]][i][0]}",
                        abs(r_means[comb[0]][i][1] - r_means[comb[1]][i][1]),
                        r_means[comb[1]][i][2],
                        r_means[comb[0]][i][3],
                        r_means[comb[1]][i][3],
                    )
                )
                i += 1

            diff_means.append(m_means)

        max_from_pairs = []
        for means in diff_means:
            max_from_pairs.append(max(means, key=lambda item: item[1]))

        return k_max, max_from_pairs

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
