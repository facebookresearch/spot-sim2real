import time

class FrameRateCounter:
    def __init__(self):
        self.start_time = None
        self.frame_count = 0
    def update(self):
        """ Update the frame rate counter by incrementing the frame count. """
        if self.start_time is None:
            self.start_time = time.perf_counter_ns()  # Start the timer on the first frame
        self.frame_count += 1
    def reset(self):
        """ Reset the frame rate counter. """
        self.start_time = None
        self.frame_count = 0
    def avg_value(self):
        """ Calculate the average frame rate in frames per second. """
        if self.start_time is None or self.frame_count == 0:
            return 0
        end_time = time.perf_counter_ns()
        elapsed_time_ns = end_time - self.start_time
        if elapsed_time_ns > 0:
            return float(round(self.frame_count / (elapsed_time_ns * 1e-9)))
        return 0
