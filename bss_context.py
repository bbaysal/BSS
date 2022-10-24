import csv
import os
import time
from copy import deepcopy

import numpy as np

from BSS import utils
from BSS.bss_strategy import BSSStrategy


class BSSContext:
    __track_list = {}

    def __init__(self, strategy: BSSStrategy, reference_path, estimates_path, genre_csv_path) -> None:
        self._strategy = strategy
        self.reference_path = reference_path
        self.estimates_path = estimates_path
        self.genre_csv_path = genre_csv_path
        self.set_csv_values()

    @property
    def strategy(self) -> BSSStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BSSStrategy) -> None:
        self._strategy = strategy

    def do_bss_on_valid_set(self):
        scores = {}
        genre_scores = {}
        track_scores = {}
        total_second = 0
        total_track_count = len([name for name in os.listdir(self.reference_path)])
        index = 1
        print("Total Track Count: " + str(total_track_count))
        print("Separation Period Begins")
        separation_start_time = time.time()
        for subdir, dirs, files in os.walk(self.reference_path):
            for directory in dirs:
                print(str(index) + " of " + str(total_track_count))
                print("Track: " + directory)
                print("Separation begins..")
                self._strategy.do_bss_for_track(self.reference_path, self.estimates_path, directory)
                total_second += self._strategy.separation_time
                print("Per Step: " + str(self._strategy.separation_time))
                print("Separated")
                index += 1
        separation_end_time = time.time()
        total_separation_time = separation_end_time - separation_start_time
        print("Separation Period Ends")
        print("Separation period time: " + str(total_separation_time))
        # print("Evaluation Period Begins")
        # index = 1
        # evaluation_start_time = time.time()
        # for subdir, dirs, files in os.walk(self.reference_path):
        #     for directory in dirs:
        #         print(str(index) + " of " + str(total_track_count))
        #         print("Track: " + directory)
        #         print("Evaluation begins..")
        #         results = utils.evaluate(self.reference_path, self.estimates_path + self._strategy.folder_name(),
        #                                  directory)
        #         for t in results.scores['targets']:
        #             metrics = {}
        #             for metric in ['SDR', 'SIR', 'ISR', 'SAR']:
        #                 metrics[metric] = np.nanmedian([np.float(f['metrics'][metric])
        #                                                 for f in t['frames']])
        #             track_scores[t['name']] = metrics
        #         scores[directory] = deepcopy(track_scores)
        #         if self.genre_csv_path != "":
        #             if directory in self.__track_list.keys():
        #                 genre = self.__track_list[directory]
        #             else:
        #                 genre = "other"
        #             genre_tracks = genre_scores.setdefault(genre, {})
        #             genre_tracks[directory] = deepcopy(track_scores)
        #             genre_scores[genre] = genre_tracks
        #
        #         print(results)
        #         print("Evaluation ended")
        #         index += 1
        # evaluation_end_time = time.time()
        # total_evaluation_time = evaluation_end_time - evaluation_start_time
        # print("Evaluation Period Ends")
        # print("Evaluation period time: " + str(total_evaluation_time))
        #
        # print("Separation process have ended.")
        # print("Writing csv files")
        # utils.write_to_csv_files(estimates_path=self.estimates_path + self._strategy.folder_name(),
        #                          scores=scores, genre_scores=genre_scores)
        print("The End..")
        print("Separation Time: " + str(total_separation_time))

    def set_csv_values(self):
        with open(self.genre_csv_path) as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                self.__track_list[row[0]] = row[1]
