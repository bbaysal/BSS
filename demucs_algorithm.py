import time

import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from BSS import utils
from BSS.bss_strategy import BSSStrategy
from demucs import pretrained
from demucs.apply import apply_model

import torch


class Demucs(BSSStrategy):
    def do_bss_for_track(self, reference_path, estimates_path, directory):
        start_time = time.time()
        [drums, bass, vocals, other, samp_rate] = utils.read_components(reference_path, directory, True)
        mixture = utils.get_mixture_from_components(drums, bass, vocals, other, is_2_channel=True)
        model = pretrained.get_model('mdx')
        mixture = np.expand_dims(mixture, axis=0)
        mixture = torch.from_numpy(mixture)
        model = model.double()
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device_str)
        out = apply_model(model, mixture)[0]  # shape is [S, C, T] with S the number of sources
        end_time = time.time()
        self._separation_time = end_time - start_time
        estimates = {}
        # So let see, where is all the white noise content is going ?
        for name, source in zip(model.sources, out):
            estimates[name] = source.cpu().detach().numpy().T

        utils.write_to_file(estimates_path + self.folder_name(), directory, estimates, samp_rate)

    def folder_name(self):
        return "Demucs/"

    def __init__(self):
        super().__init__()
        print("Separation Algorithm: Demucs")

    @property
    def separation_time(self):
        return self._separation_time
