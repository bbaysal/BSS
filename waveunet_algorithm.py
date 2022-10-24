import time

import torch

from BSS import utils
from BSS.bss_strategy import BSSStrategy
from BSS.waveunet.model.waveunet import Waveunet
import BSS.waveunet.model.utils as model_utils
from BSS.waveunet.test import predict_song, predict


class WaveUNet(BSSStrategy):

    def __init__(self):
        super().__init__()
        use_cude = torch.cuda.is_available()
        self.__device_str = "cuda" if use_cude else "cpu"
        self.__args = {
            "instruments": ["bass", "drums", "other", "vocals"],
            "cuda": "cpu",
            "features": 32,
            "load_model": "./waveunet/checkpoints/waveunet/model",
            "batch_size": 4,
            "levels": 6,
            "sr": 44100,
            "depth": 1,
            "channels": 2,
            "kernel_size": 5,
            "output_size": 2.0,
            "strides": 4,
            "res": "fixed",
            "separate": 1,
            "feature_growth": "double",
            "conv_type": "gn",
        }
        print("Separation Algorithm: Wave-U-Net")
        num_features = [self.__args["features"] * i for i in range(1, self.__args["levels"] + 1)] if self.__args[
                                                                                                         "feature_growth"] == "add" else \
            [self.__args["features"] * 2 ** i for i in range(0, self.__args["levels"])]
        target_outputs = int(self.__args["output_size"] * self.__args["sr"])
        self.__model = Waveunet(self.__args["channels"], num_features, self.__args["channels"],
                                self.__args["instruments"],
                                kernel_size=self.__args["kernel_size"],
                                target_output_size=target_outputs, depth=self.__args["depth"],
                                strides=self.__args["strides"],
                                conv_type=self.__args["conv_type"], res=self.__args["res"],
                                separate=self.__args["separate"])

    def do_bss_for_track(self, reference_path, estimates_path, directory):
        print("Producing source estimates for input mixture file " + reference_path)
        # Prepare input audio as track object (in the MUSDB sense), so we can use the MUSDB-compatible prediction function
        [drums, bass, vocals, other, samp_rate] = utils.read_components(reference_path, directory, True)
        mixture = utils.get_mixture_from_components(drums, bass, vocals, other, is_2_channel=True)
        use_cuda = torch.cuda.is_available()
        self.__model = self.__model.double()
        state = model_utils.load_model(self.__model, None, "./waveunet/checkpoints/waveunet/model",
                                       cuda=False)
        start_time = time.time()
        preds = predict(mixture, self.__model)
        self._separation_time = time.time() - start_time
        estimates = {}

        for key in preds:
            estimates[key] = preds[key].T

        utils.write_to_file(estimates_path + self.folder_name(), directory, estimates, samp_rate)

    def folder_name(self):
        return "WaveUNet/"

    @property
    def separation_time(self):
        return self._separation_time
