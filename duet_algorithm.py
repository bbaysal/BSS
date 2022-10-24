import numpy as np
import time
import nussl

from BSS import utils
from BSS.bss_strategy import BSSStrategy


class Duet(BSSStrategy):
    def __init__(self, num_sources=4):
        super().__init__()
        print("Separation Algorithm: DUET")
        self._num_sources = num_sources

    def folder_name(self):
        return "DUET/"

    def do_bss_for_track(self, reference_path, estimates_path, directory):
        start_time = time.time()
        nussl.utils.seed(0)
        [drums, bass, vocals, other, samp_rate] = utils.read_components(reference_path, directory, True)
        mixture = utils.get_mixture_from_components(drums, bass, vocals, other, is_2_channel=True)
        audio_signal = nussl.AudioSignal(audio_data_array=mixture, sample_rate=samp_rate)
        start_time = time.time()
        separator = nussl.separation.spatial.Duet(
            audio_signal, num_sources=self._num_sources)
        estimates = separator()
        end_time = time.time()
        self._separation_time = end_time - start_time
        estimates = {
            f'Component{i}': e
            for i, e in enumerate(estimates)
        }
        print("Seperation time for track:" + str(end_time - start_time))
        # estimates = [estimates["Component0"].audio_data.T, estimates["Component1"].audio_data.T,
        #              estimates["Component2"].audio_data.T, estimates["Component3"].audio_data.T]
        # references = {"vocals": vocals, "drums": drums, "other": other, "bass": bass}
        #
        # ordered_estimates = utils.get_ordered_estimates(references, estimates)
        #
        # utils.write_to_file(estimates_path + self.folder_name(), directory, ordered_estimates, samp_rate)

    @property
    def separation_time(self):
        return self._separation_time
