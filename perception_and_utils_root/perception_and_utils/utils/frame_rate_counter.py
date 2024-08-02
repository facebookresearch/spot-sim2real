import time
import logging
class FrameRateCounter:
    def __init__(self, logger):
        self.logger = logger
        self.start_time = None
        self.total_time_ns = 0
        self.total_frames = 0
        self.window_size = 5
        self.window = [0.0] * self.window_size
        self.window_index = 0
        self.start_index = 0
    def start(self):
        self.start_time = time.perf_counter_ns()
    def stop(self):
        end_time = time.perf_counter_ns()
        elapsed_time_ns = end_time - self.start_time
        if elapsed_time_ns > 0:
            self.total_frames += 1
            self.total_time_ns += elapsed_time_ns

            # Add the current frame time to the window
            self.window[self.window_index] = elapsed_time_ns
            self.window_index = (self.window_index + 1) % self.window_size

            # If the window is full, increment the start index
            if self.window_index == self.start_index:
                self.start_index = (self.start_index + 1) % self.window_size

            # Compute instant. rate
            instantaneous_frame_rate = 1 / (elapsed_time_ns * 1e-9)
            # Compute average frame rate from start
            average_frame_rate = self.total_frames / (self.total_time_ns * 1e-9)
            # Compute average frame rate over the window
            window_average = 0.0
            if self.window_index >= self.start_index:
                window_average = (self.window_index - self.start_index + 1) / (sum(self.window[self.start_index:self.window_index+1]) * 1e-9)
            else:
                window_average = (self.window_size - self.start_index + self.window_index + 1) / (sum(self.window[self.start_index:] + self.window[:self.window_index+1]) * 1e-9)

            self.logger.info(f"Averate over last {self.window_size} records: {window_average:.2f} FPS")

    def reset(self):
        self.start_time = None
        self.total_time_ns = 0
        self.total_frames = 0
        self.window_index = 0
        self.start_index = 0
        self.window = [0] * self.window_size
