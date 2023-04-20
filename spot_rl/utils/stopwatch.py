import time
from collections import OrderedDict, deque

import numpy as np


class Stopwatch:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.times = OrderedDict()
        self.current_time = time.time()

    def reset(self):
        self.current_time = time.time()

    def record(self, key):
        if key not in self.times:
            self.times[key] = deque(maxlen=self.window_size)
        self.times[key].append(time.time() - self.current_time)
        self.current_time = time.time()

    def print_stats(self, latest=False):
        name2time = OrderedDict()
        for k, v in self.times.items():
            if latest:
                name2time[k] = v[-1]
            else:
                name2time[k] = np.mean(v)
        name2time["total"] = np.sum(list(name2time.values()))
        print(" ".join([f"{k}: {v:.4f}" for k, v in name2time.items()]))
