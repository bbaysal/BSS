import time
from copy import deepcopy

import museval
import utils
import numpy as np
import os
from IPython import display as ipd
import csv

from BSS.bss_strategy import BSSStrategy


class FastICA(BSSStrategy):

    def folder_name(self):
        return "FastICA/"

    def __init__(self, alpha=1, thresh=1e-8, iterations=5000):
        super().__init__()
        print("Separation Algorithm: FastICA")
        self.__thresh = 1e-8
        self.__iterations = 5000
        self.__alpha = 1

    def __center(self, x):
        mean = np.mean(x, axis=1, keepdims=True)
        centered = x - mean
        return centered, mean

    def __covariance(self, x):
        mean = np.mean(x, axis=1, keepdims=True)
        n = np.shape(x)[1] - 1
        m = x - mean
        return (m.dot(m.T)) / n

    def __whiten(self, x):
        # Calculate the covariance matrix
        coVarM = self.__covariance(x)

        # Single value decoposition
        U, S, V = np.linalg.svd(coVarM)

        # Calculate diagonal matrix of eigenvalues
        d = np.diag(1.0 / np.sqrt(S))

        # Calculate whitening matrix
        whiteM = np.dot(U, np.dot(d, U.T))

        # Project onto whitening matrix
        Xw = np.dot(whiteM, x)

        return Xw, whiteM

    def __fastIca(self, signals):
        m, n = signals.shape

        # Initialize random weights
        W = np.random.rand(m, m)

        for c in range(m):
            w = W[c, :].copy().reshape(m, 1)
            w = w / np.sqrt((w ** 2).sum())

            i = 0
            lim = 100
            while (lim > self.__thresh) & (i < self.__iterations):
                # Dot product of weight and signal
                ws = np.dot(w.T, signals)

                # Pass w*s into contrast function g
                wg = np.tanh(ws * self.__alpha).T

                # Pass w*s into g prime
                wg_ = (1 - np.square(np.tanh(ws))) * self.__alpha

                # Update weights
                wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

                # Decorrelate weights
                wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.sqrt((wNew ** 2).sum())

                # Calculate limit condition
                lim = np.abs(np.abs((wNew * w).sum()) - 1)

                # Update weights
                w = wNew

                # Update counter
                i += 1

            W[c, :] = w.T
        return W

    def do_bss_for_track(self, reference_path, estimates_path, directory):
        [drums, bass, vocals, other, samp_rate] = utils.read_components(reference_path, directory)
        mixed = utils.get_mixture_from_components(drums, bass, vocals, other)
        start_time = time.time()
        unMixed = self.__bss(mixed)
        end_time = time.time()
        self._separation_time = end_time - start_time
        estimates = [unMixed[:, 0], unMixed[:, 1], unMixed[:, 2], unMixed[:, 3]]
        references = {"vocals": vocals, "drums": drums, "other": other, "bass": bass}

        ordered_estimates = utils.get_ordered_estimates(references, estimates)

        utils.write_to_file(estimates_path + self.folder_name(), directory, ordered_estimates, samp_rate)

    def __bss(self, mixed):
        # Center signals
        Xc, meanX = self.__center(mixed)

        # Whiten mixed signals
        Xw, whiteM = self.__whiten(Xc)
        # print(np.round(self.__covariance(Xw)))

        W = self.__fastIca(Xw)

        # Un-mix signals using
        unMixed = Xw.T.dot(W.T)

        # Subtract mean
        unMixed = (unMixed.T - meanX).T

        return unMixed

    @property
    def separation_time(self):
        return self._separation_time
