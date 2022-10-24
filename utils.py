from copy import deepcopy

import librosa
import os
import numpy as np
from IPython import display as ipd
import sys
from pathlib import Path
import soundfile as sf
import csv

sys.path.append('../')
import museval

drums_path = "drums.wav"
bass_path = "bass.wav"
vocals_path = "vocals.wav"
other_path = "other.wav"
mixture_path = "mixture.wav"


def read_components(subdir, directory, is_2_channel=False):
    if is_2_channel:
        drums, samp_rate1 = sf.read(os.path.join(subdir + directory, drums_path))
        bass, samp_rate2 = sf.read(os.path.join(subdir + directory, bass_path))
        vocals, samp_rate3 = sf.read(os.path.join(subdir + directory, vocals_path))
        other, samp_rate4 = sf.read(os.path.join(subdir + directory, other_path))
    else:
        drums, samp_rate1 = librosa.load(os.path.join(subdir + directory, drums_path))
        bass, samp_rate2 = librosa.load(os.path.join(subdir + directory, bass_path))
        vocals, samp_rate3 = librosa.load(os.path.join(subdir + directory, vocals_path))
        other, samp_rate4 = librosa.load(os.path.join(subdir + directory, other_path))
    return drums, bass, vocals, other, samp_rate1


def get_mixture(subdir, directory, apply_noise=False, is_2_channel=False):
    if is_2_channel:
        [drums, bass, vocals, other, samp_rate] = read_components(subdir, directory, True)
        return mix_2_channel_sources(bass, drums, other, vocals)
    else:
        [drums, bass, vocals, other, samp_rate] = read_components(subdir, directory)
        return mix_sources([drums, bass, vocals, other], apply_noise=False), samp_rate


def get_mixture_from_components(drums, bass, vocals, other, apply_noise=False, is_2_channel=False):
    if is_2_channel:
        return mix_2_channel_sources(bass, drums, other, vocals)
    else:
        return mix_sources([drums, bass, vocals, other], apply_noise)


def mix_2_channel_sources(bass, drums, other, vocals):
    channel_1 = mix_sources([drums[:, 0], bass[:, 0], vocals[:, 0], other[:, 0]])
    channel_2 = mix_sources([drums[:, 1], bass[:, 1], vocals[:, 1], other[:, 1]])
    channel_1 = np.mean(channel_1, axis=0)
    channel_2 = np.mean(channel_2, axis=0)
    return np.array([channel_1, channel_2])


def mix_sources(sources, apply_noise=False):
    for i in range(len(sources)):
        max_val = np.max(sources[i])
        if max_val > 1 or np.min(sources[i]) < 1:
            sources[i] = sources[i] / (max_val / 2) - 0.5

    mixture = np.c_[[source for source in sources]]

    if apply_noise:
        mixture += 0.02 * np.random.normal(size=mixture.shape)

    return mixture


def load_all_component_audios(X, samp_rate):
    ipd.display(ipd.Audio(X[0, :], rate=samp_rate))
    ipd.display(ipd.Audio(X[1, :], rate=samp_rate))
    ipd.display(ipd.Audio(X[2, :], rate=samp_rate))
    ipd.display(ipd.Audio(X[3, :], rate=samp_rate))


def evaluate(reference_path, estimates_path, directory):
    # evaluate an existing estimate folder with wav files
    return museval.eval_dir(
        reference_dir=reference_path + directory,  # path to estimate folder
        estimates_dir=estimates_path + directory  # set a folder to write eval json files
    )


def get_ordered_estimates(reference, estimates):
    ordered_estimates = {}
    for key in reference:
        axes_to_expand = 0
        if reference[key].ndim == 1:
            axes_to_expand = [0, 2]

        SDR, ISR, SIR, SAR = museval.evaluate(np.expand_dims(reference[key], axis=axes_to_expand),
                                              np.expand_dims(estimates[0], axis=axes_to_expand))
        max_SAR = np.nanmedian(SAR, axis=1)
        print(key + " - Component 0: " + str(max_SAR))
        min_index = 0
        for index in range(1, len(estimates)):
            SDR, ISR, SIR, SAR = museval.evaluate(np.expand_dims(reference[key], axis=axes_to_expand),
                                                  np.expand_dims(estimates[index], axis=axes_to_expand))
            component_SAR = np.nanmedian(SAR, axis=1)
            print(key + " - Component " + str(index) + ": " + str(component_SAR))
            if max_SAR < component_SAR:
                max_SAR = component_SAR
                min_index = index
        print(key + " - Selected Component: " + str(min_index) + " - Selected Maximum SAR: " + str(max_SAR))
        ordered_estimates[key] = estimates[min_index]
        estimates.pop(min_index)
    return ordered_estimates


def write_to_file(estimates_path, directory, estimates, sample_rate):
    Path(estimates_path + directory).mkdir(parents=False, exist_ok=True)
    mixture = np.array([estimates["vocals"], estimates["bass"], estimates["other"], estimates["drums"]])
    sf.write(estimates_path + directory + "/mixture.wav", np.mean(mixture, axis=0).astype(np.float32),
             samplerate=sample_rate)
    for key in estimates:
        sf.write(estimates_path + directory + "/" + key + ".wav", estimates[key].astype(np.float32),
                 samplerate=sample_rate)


def write_to_csv_files(scores, genre_scores, estimates_path):
    construct_csv(scores, estimates_path, 'scores.csv')

    for genre in genre_scores:
        genre_score = genre_scores[genre]
        filename = genre.replace("/", "-") + ".csv"
        construct_csv(genre_score, estimates_path, filename)


def construct_csv(scores, estimates_path, filename):
    header = ['Track Name', "Components", 'SDR', 'SIR', 'ISR', 'SAR']
    components_specific_values = {}
    with open(estimates_path + filename, 'w', newline="") as file:
        csvwriter = csv.writer(file)  # 2. create a csvwriter object
        csvwriter.writerow(header)  # 4. write the header
        for key in scores:
            i = 0
            components = scores[key]
            for component_name in components:
                component = deepcopy(components[component_name])
                components_specific_value = components_specific_values.setdefault(component_name,
                                                                                  {"SDR": [], "SIR": [], "ISR": [],
                                                                                   "SAR": []})
                components_specific_value["SDR"].append(component["SDR"])
                components_specific_value["SIR"].append(component["SIR"])
                components_specific_value["ISR"].append(component["ISR"])
                components_specific_value["SAR"].append(component["SAR"])

                if i == 0:
                    data = [key, component_name, component["SDR"], component["SIR"], component["ISR"], component["SAR"]]
                    i += 1
                else:
                    data = ['', component_name, component["SDR"], component["SIR"], component["ISR"], component["SAR"]]
                csvwriter.writerow(data)  # 5. write the rest of the data

        for component_name in ["vocals.wav", "bass.wav", "drums.wav", "other.wav"]:
            median_sdr = np.median(components_specific_values[component_name]["SDR"])
            median_sir = np.median(components_specific_values[component_name]["SIR"])
            median_isr = np.median(components_specific_values[component_name]["ISR"])
            median_sar = np.median(components_specific_values[component_name]["SAR"])

            data = ["Mean Value: ", component_name, median_sdr, median_sir, median_isr, median_sar]
            csvwriter.writerow(data)  # 5. write the rest of the data
