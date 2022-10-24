from abc import ABC, abstractmethod


class BSSStrategy(ABC):

    def __init__(self):
        self._separation_time = None

    @abstractmethod
    def do_bss_for_track(self, reference_path, estimates_path, directory):
        pass

    @abstractmethod
    def folder_name(self):
        pass

    @property
    def separation_time(self):
        return self._separation_time

