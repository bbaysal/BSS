import numpy as np
import torch.cuda
import soundfile as sf

from BSS import utils
from BSS.bss_strategy import BSSStrategy
from openunmix import predict


class OpenUnmix(BSSStrategy):

    def __init__(self):
        print("Separation Algorithm: OpenUnmix")
        use_cude = torch.cuda.is_available()
        self.__device_str = "cuda" if use_cude else "cpu"
        self.__num_components = 4

    def folder_name(self):
        return "OpenUnmix/"

    def do_bss_for_track(self, reference_path, estimates_path, directory):
        [drums, bass, vocals, other, samp_rate] = utils.read_components(reference_path, directory, is_2_channel=True)
        mixed = utils.get_mixture_from_components(drums, bass, vocals, other, is_2_channel=True)

        estimates = predict.separate(
            torch.as_tensor(mixed).float(),
            targets=["vocals", "bass", "drums", "other"],
            rate=samp_rate,
            device="cpu"
        )

        estimates_cpu = {}
        for target, estimate in estimates.items():
            audio = estimate.detach().cpu().numpy()[0]
            estimates_cpu[target] = audio.T

        utils.write_to_file(estimates_path + self.folder_name(), directory, estimates_cpu, samp_rate)
