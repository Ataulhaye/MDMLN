import copy
import itertools
import pickle
from heapq import nlargest

import nilearn as ni
import numpy as np
import pandas as pd
import scipy.io
import scipy.special as special
import torch
from matplotlib import pyplot as plt
from nilearn import image, plotting
from scipy._lib._bunch import _make_tuple_bunch
from scipy.spatial import KDTree
from scipy.stats import spearmanr

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from ExportData import ExportData
from RSAConfig import RSAConfig


class RepresentationalSimilarityAnalysis:

    def get_spheres(self, radius, coordinates: pd.DataFrame):

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
        return final_spheres

    def RSA(self, fmri_data: list[Brain], RDM, radius, coordinates):
        # Creating the sphere centers with some radius

        # sphere_centers = self.get_sphere_centers(coordinates, radius)
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

    def run_RSA(self, fmri_data: list[Brain], RDM, radius, tal_MNI_space, NIfTI):
        # Creating the sphere centers with some radius

        # sphere_centers = self.get_sphere_centers(coordinates, radius)
        sphere_centers = self.get_sphere_centers(tal_MNI_space, radius)
        # creating xyz points by providing the TALX, TALY, and TALZ
        xyz_points = np.array(tal_MNI_space[["TALX", "TALY", "TALZ"]])

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
        for data in fmri_data:
            all_data = []
            smoothed_img = image.smooth_img(NIfTI, None)
            new_nii = copy.deepcopy(smoothed_img)
            new_nii._dataobj = np.zeros(new_nii._dataobj.shape)
            coord2xyz_dict = self.get_xyz_coordinates(smoothed_img)

            for sph_cntr, vox_indices in final_spheres:
                r = self.calculate_spermanr(data, vox_indices, RDM)
                all_data.append((sph_cntr, vox_indices, r))
                for vox_index in vox_indices:
                    xyz_coo = (
                        tal_MNI_space.iloc[vox_index]["TALX"],
                        tal_MNI_space.iloc[vox_index]["TALY"],
                        tal_MNI_space.iloc[vox_index]["TALZ"],
                    )
                    aal_coo = coord2xyz_dict[xyz_coo]

                    new_nii._dataobj[tuple(aal_coo)] = r

            new_nii._data_cache = new_nii._dataobj
            all_r_means.append(all_data)

            # file_name = f"{data.lobe.name}_{data.current_labels.name}-sph_cntr-vox_indices-R-audio_RDM_RSA.pickle"
            # with open(file_name, "wb") as output:
            # pickle.dump(all_data, output)

            # new_nii = image.smooth_img(new_nii, "fast")
            plotting.plot_glass_brain(new_nii, threshold=0)
            gname = (
                f"{data.lobe.name}_{data.current_labels.name}-{self.run_RSA.__name__}"
            )
            graph_name = ExportData.get_file_name(".png", gname)
            plt.savefig(graph_name)
            # plotting.show()

        return all_r_means

    def get_xyz_coordinates(self, smoothed_img):
        coord2xyz_dict = {}
        xlen, ylen, zlen = smoothed_img._dataobj.shape
        for i in range(xlen):
            for j in range(ylen):
                for k in range(zlen):
                    # matmul Matrix product of two arrays
                    x, y, z, __ = np.matmul(smoothed_img.affine, np.array([i, j, k, 1]))
                    coord2xyz_dict[(x, y, z)] = [i, j, k]
        return coord2xyz_dict

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

    def calculate_spermanr(self, brain: Brain, voxel_indicies, RDM):
        # The part before the comma represents the row while the part after the comma represents the column
        voxels = brain.voxels[:, voxel_indicies]
        new_shape = (
            int(voxels.shape[0] / 4),
            4,
            voxels.shape[1],
        )
        rvoxels = np.reshape(voxels, new_shape)
        # make RDM from combined voxels and sphere
        RDMs = self.make_RDMs(rvoxels)
        # calculating the rank
        # r = []
        # for i in range(RDMs.shape[0]):
        # v = spearmanr(RDM.ravel(), RDMs[i].ravel()).statistic
        # v = m(RDM.ravel(), RDMs[i].ravel()).statistic
        # r.append(v)
        r = [
            spearmanr(RDM.ravel(), RDMs[i].ravel()).statistic
            for i in range(RDMs.shape[0])
        ]
        return np.nanmean(r)

    def make_RDMs(self, data):
        rdms = []
        # constants = []
        for i in range(data.shape[0]):
            rdm = 1 - torch.corrcoef(torch.from_numpy(data[i]))
            # ravel_rdm = rdm.ravel()
            # set_rdm = set(ravel_rdm.tolist())
            # if len(set_rdm) == 1:
            # constants.append(set_rdm)
            rdms.append(rdm)

        # if len(constants) > 0:
        # print(f"Data is constatnt: {constants}; Count: {len(constants)}")
        # diff = len(constants) - len(rdms)
        # if diff > 0:
        # print(f"Difference  Constant-RDMS - RDMS: {diff}")
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

        radis_adjus = int(1.5 * radius)

        sphere_centers = []
        for x in range(minx, maxx, radis_adjus):
            for y in range(miny, maxy, radis_adjus):
                for z in range(minz, maxz, radis_adjus):
                    sphere_centers.append(np.array([x, y, z]))

        return np.array(sphere_centers)
