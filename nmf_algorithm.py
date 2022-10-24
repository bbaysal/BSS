import time
from copy import deepcopy
from tqdm import tnrange

import numpy as np

from BSS import utils
from BSS.bss_strategy import BSSStrategy


class NMF(BSSStrategy):
    EPS = 2.0 ** -52
    __track_list = {}

    def __init__(self, thresh=1e-5, iterations=1000, component_number=4):
        super().__init__()
        print("Separation Algorithm: NMF")
        self.tresh = thresh
        self.iterations = iterations
        self.component_number = component_number

    """
        Name: NMF
        Date: Jun 2019
        Programmer: Christian Dittmar, Yiğitcan Özer

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        If you use the 'NMF toolbox' please refer to:
        [1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
            Müller
            NMF Toolbox: Music Processing Applications of Nonnegative Matrix
            Factorization
            In Proceedings of the International Conference on Digital Audio Effects
            (DAFx), 2019.

        License:
        This file is part of 'NMF toolbox'.
        https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
        'NMF toolbox' is free software: you can redistribute it and/or modify it
        under the terms of the GNU General Public License as published by the
        the Free Software Foundation, either version 3 of the License, or (at
        your option) any later version.

        'NMF toolbox' is distributed in the hope that it will be useful, but
        WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
        Public License for more details.

        You should have received a copy of the GNU General Public License along
        with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
    """

    def folder_name(self):
        return "NMF/"

    def __nmf(self, V, parameter):
        """Given a non-negative matrix V, find non-negative templates W and activations
        H that approximate V.

        References
        ----------
        [2] Lee, DD & Seung, HS. "Algorithms for Non-negative Matrix Factorization"

        [3] Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, and Shun-ichi Amari
        "Nonnegative Matrix and Tensor Factorizations: Applications to
        Exploratory Multi-Way Data Analysis and Blind Source Separation"
        John Wiley and Sons, 2009.

        Parameters
        ----------
        V: array-like
            K x M non-negative matrix to be factorized

        parameter: dict
            costFunc      Cost function used for the optimization, currently
                          supported are:
                          'EucDdist' for Euclidean Distance
                          'KLDiv' for Kullback Leibler Divergence
                          'ISDiv' for Itakura Saito Divergence
            numIter       Number of iterations the algorithm will run.
            numComp       The rank of the approximation

        Returns
        -------
        W: array-like
            K x R non-negative templates
        H: array-like
            R x M non-negative activations
        nmfV: array-like
            List with approximated component matrices
        """
        parameter = self.init_parameters(parameter)

        # get important params
        K, M = V.shape
        R = parameter['numComp']
        L = parameter['numIter']

        # initialization of W and H
        if isinstance(parameter['initW'], list):
            W = np.array(parameter['initW'])
        else:
            W = deepcopy(parameter['initW'])

        H = deepcopy(parameter['initH'])

        # create helper matrix of all ones
        onesMatrix = np.ones((K, M))

        # normalize to unit sum
        V /= (self.EPS + V.sum())

        # main iterations
        for iter in tnrange(L, desc='Processing'):

            # compute approximation
            Lambda = self.EPS + W @ H

            # switch between pre-defined update rules
            if parameter['costFunc'] == 'EucDist':  # euclidean update rules
                if not parameter['fixW']:
                    W *= (V @ H.T / (Lambda @ H.T + self.EPS))

                H *= (W.T @ V / (W.T @ Lambda + self.EPS))

            elif parameter['costFunc'] == 'KLDiv':  # Kullback Leibler divergence update rules
                if not parameter['fixW']:
                    W *= ((V / Lambda) @ H.T) / (onesMatrix @ H.T + self.EPS)

                H *= (W.T @ (V / Lambda)) / (W.T @ onesMatrix + self.EPS)

            elif parameter['costFunc'] == 'ISDiv':  # Itakura Saito divergence update rules
                if not parameter['fixW']:
                    W *= ((Lambda ** -2 * V) @ H.T) / ((Lambda ** -1) @ H.T + self.EPS)

                H *= (W.T @ (Lambda ** -2 * V)) / (W.T @ (Lambda ** -1) + self.EPS)

            else:
                raise ValueError('Unknown cost function')

            # normalize templates to unit sum
            if not parameter['fixW']:
                normVec = W.sum(axis=0)
                W *= 1.0 / (self.EPS + normVec)

        nmfV = list()

        # compute final output approximation
        for r in range(R):
            nmfV.append(W[:, r].reshape(-1, 1) @ H[r, :].reshape(1, -1))

        return W, H, nmfV

    def init_parameters(self, parameter):
        """Auxiliary function to set the parameter dictionary

        Parameters
        ----------
        parameter: dict
            See the above function inverseSTFT for further information

        Returns
        -------
        parameter: dict
        """
        parameter['costFunc'] = 'KLDiv' if 'costFunc' not in parameter else parameter['costFunc']
        parameter['numIter'] = 30 if 'numIter' not in parameter else parameter['numIter']
        parameter['fixW'] = False if 'fixW' not in parameter else parameter['fixW']

        return parameter

    def do_bss_for_track(self, reference_path, estimates_path, directory):
        [drums, bass, vocals, other, samp_rate] = utils.read_components(reference_path, directory)
        mixed = utils.get_mixture_from_components(drums, bass, vocals, other)

        params = self.init_track_parameters(mixed)
        start_time = time.time()
        W, H, nmfV = self.__nmf(mixed, params)
        end_time = time.time()
        self._separation_time = end_time - start_time
        # V_approximated = W.dot(H)
        # coefficient = 10 ** 7
        # V_approximated[0] = self.reject_outliers(V_approximated[0]) * coefficient
        # V_approximated[1] = self.reject_outliers(V_approximated[1]) * coefficient
        # V_approximated[2] = self.reject_outliers(V_approximated[2]) * coefficient
        # V_approximated[3] = self.reject_outliers(V_approximated[3]) * coefficient
        #
        # estimates = [V_approximated[0], V_approximated[1], V_approximated[2], V_approximated[3]]
        # references = {"vocals": vocals, "drums": drums, "other": other, "bass": bass}
        #
        # ordered_estimates = utils.get_ordered_estimates(references, estimates)

        # utils.write_to_file(estimates_path + self.folder_name(), directory, ordered_estimates, samp_rate)

    def init_track_parameters(self, mixed):
        K = mixed.shape[0]
        N = mixed.shape[1]
        R = self.component_number
        W_init = np.random.rand(K, R)
        W_init = W_init / np.max(W_init)
        H_init = np.random.rand(R, N)
        params = dict()
        params["numComp"] = R
        params["initW"] = W_init
        params["initH"] = H_init
        params["numIter"] = self.iterations
        params['costFunc'] = 'EucDist'
        params = self.init_parameters(params)
        return params

    def reject_outliers(self, data, m=50.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        # return data[s < m]
        return np.where(s < m, data, mdev)

    @property
    def separation_time(self):
        return self._separation_time
