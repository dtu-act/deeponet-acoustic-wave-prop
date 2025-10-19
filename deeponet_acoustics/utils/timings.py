# ==============================================================================
# Copyright 2023 Technical University of Denmark
# Author: Nikolas Borrel-Jensen
#
# All Rights Reserved.
#
# Licensed under the MIT License.
# ==============================================================================
import io
import os
import time


class TimingsWriter:
    log_dir: str
    timings_start: dict[str, float] = dict()
    timings: dict[str, tuple[int, float]] = dict()
    file_handle: io.TextIOWrapper

    def __init__(self, log_dir):
        self.log_dir = log_dir
        filepath = os.path.join(self.log_dir, "timings.txt")
        self.file_handle = open(filepath, "w")

    def startTiming(self, key):
        self.timings_start[key] = time.perf_counter()

    def endTiming(self, key):
        if key not in self.timings_start:
            raise Exception("Timing has not been started.")

        if key not in self.timings:
            self.timings[key] = (0, 0)

        total_timing = self.timings[key][1] + (
            time.perf_counter() - self.timings_start[key]
        )
        num_timings = self.timings[key][0] + 1
        self.timings[key] = (num_timings, total_timing)

    def writeTimings(self, output_str: dict[str, str]):
        self.file_handle.write("--------------------------------\n")
        for key, val in self.timings.items():
            if val[0] == 1:
                self.file_handle.write(f"{output_str[key]} {val[1] * 1000} ms\n")
            else:
                self.file_handle.write(f"{output_str[key]} total: {val[1] * 1000} ms\n")
                self.file_handle.write(
                    f"{output_str[key]} per iteration: {(val[1] / val[0]) * 1000} ms\n"
                )

        self.file_handle.write("--------------------------------\n\n")
        self.file_handle.flush()

    def resetTimings(self):
        self.timings_start.clear()
        self.timings.clear()
        self.N_iter = 0

    def __del__(self):
        self.file_handle.close()
