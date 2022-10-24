from BSS.bss_strategy import BSSStrategy
from BSS.spleeter.separator import Separator
from BSS.spleeter.audio.adapter import AudioAdapter
from BSS import utils

import numpy as np
import os

# This lise was used for disable gpu on tensorflow. I have 2gb gpu memory and it didn't enough for separete
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import time
import soundfile as sf
import warnings

warnings.filterwarnings('ignore')


class SpleeterStrategy(BSSStrategy):

    def __init__(self):
        super().__init__()
        self._separation_time = None
        print("Separation Algorithm: Spleeter")
        gpu_memory_fraction = 0.4  # Fraction of GPU memory to use
        config = tf.compat.v1.ConfigProto(gpu_options=
                                          tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction))
        sess = tf.compat.v1.Session(config=config)

    def do_bss_for_track(self, reference_path, estimates_path, directory):
        [drums, bass, vocals, other, samp_rate] = utils.read_components(reference_path, directory, True)
        mixture = utils.get_mixture_from_components(drums, bass, vocals, other, is_2_channel=True)
        sf.write(estimates_path + self.folder_name() + "temp.wav", mixture.T.astype(np.float32), samp_rate)
        tf.config.experimental.set_visible_devices([], 'GPU')
        separator = Separator('spleeter:4stems')

        audio_loader = AudioAdapter.default()
        waveform, _ = audio_loader.load(estimates_path + self.folder_name() + "temp.wav", sample_rate=samp_rate)

        start_time = time.time()
        prediction = separator.separate(mixture.T)
        self._separation_time = time.time() - start_time
        estimates = {}
        for target, estimate in prediction.items():
            estimates[target] = estimate

        utils.write_to_file(estimates_path + self.folder_name(), directory, estimates, samp_rate)

    def folder_name(self):
        return "Spleeter/"

    @property
    def separation_time(self):
        return self._separation_time
