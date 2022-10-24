from BSS.spleeter_algorithm import SpleeterStrategy
from BSS.demucs_algorithm import Demucs
from BSS.waveunet_algorithm import WaveUNet
from fastIca_algorithm import FastICA
from nmf_algorithm import NMF
from duet_algorithm import Duet
import spleeter
from BSS.bss_context import BSSContext
from BSS.open_unmix_algorithm import OpenUnmix
import os

# This lise was used for disable gpu on tensorflow. I have 2gb gpu memory and it didn't enough for separete
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

reference_path = "../../../musdb18hq/"
estimates_path = "../../../predict/"


def get_available_devices():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def main():
    strategy = FastICA()
    print(get_available_devices())
    print(tf.config.list_physical_devices('GPU'))
    if tf.test.gpu_device_name():
        print('Default GPU Device: {} '.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    bss_context = BSSContext(strategy, reference_path, estimates_path, "../../../musdb18hq/tracklist.csv")
    #bss_context.do_bss_on_valid_set()
    bss_context.strategy = NMF()
    bss_context.do_bss_on_valid_set()
    bss_context.strategy = Duet()
    bss_context.do_bss_on_valid_set()
    # bss_context.strategy = OpenUnmix()
    # bss_context.do_bss_on_valid_set()
    # bss_context.strategy= FastICA()
    # bss_context.do_bss_on_valid_set()
    # bss_context.strategy= NMF()
    # bss_context.do_bss_on_valid_set()
    # bss_context.strategy= Duet()
    # bss_context.do_btss_on_valid_set()


if __name__ == "__main__":
    main()
