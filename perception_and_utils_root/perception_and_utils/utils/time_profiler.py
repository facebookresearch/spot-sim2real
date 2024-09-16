import time

WINDOW_SIZE = 5


class TimeProfiler:
    def __init__(self, window_size=WINDOW_SIZE):
        """
        A class to measure and calculate frame rates.

        The class uses a timer to track the time elapsed between frames,
        and calculates the instantaneous and average frame rates based on this data.
        The instantaneous frame rate is the frame rate at the current moment,
        while the average frame rate is the average frame rate since the start of the timer.

        The class also provides a windowed average frame rate,
        which is the average frame rate over a specified number of frames.
        This can be useful for smoothing out fluctuations in the frame rate
        and getting a more accurate picture of the application's performance.

        How to Use:
            1. Create an instance of the TimeProfiler class, passing a window size as an argument (optional).
            2. Call the start method to start the timer.
            3. Perform some operation that you want to measure the frame rate for.
            4. Call the stop method to stop the timer and get the instantaneous and average frame rates.
            5. If you want to reset the profiler, call the reset method.

        Example:
            profiler = TimeProfiler()

            while (condition):
                # Start the profiler
                profiler.start()

                # Perform some operation here

                # Stop the profiler and get the frame rates
                instantaneous_frame_rate, average_frame_rate_from_start, average_frame_rate_over_window = profiler.stop()

        """
        self.start_time = None
        self.total_time_ns = 0
        self.total_frames = 0
        self.window_size = window_size
        self.window = [0.0] * self.window_size
        self.window_index = 0
        self.start_index = 0

    def start(self):
        self.start_time = time.perf_counter_ns()

    def stop(self):
        """
        Stop the timer and return the frame rates (instantaneous, and window average)
        Returns:
            instantaneous_frame_rate: float
            average_frame_rate_from_start: float
            average_frame_rate_over_window: float
        """
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
            average_frame_rate_from_start = self.total_frames / (
                self.total_time_ns * 1e-9
            )

            # Compute average frame rate over the window
            average_frame_rate_over_window = 0.0
            if self.window_index >= self.start_index:
                average_frame_rate_over_window = (
                    self.window_index - self.start_index + 1
                ) / (sum(self.window[self.start_index : self.window_index + 1]) * 1e-9)
            else:
                average_frame_rate_over_window = (
                    self.window_size - self.start_index + self.window_index + 1
                ) / (
                    sum(
                        self.window[self.start_index :]
                        + self.window[: self.window_index + 1]
                    )
                    * 1e-9
                )

        return (
            instantaneous_frame_rate,
            average_frame_rate_from_start,
            average_frame_rate_over_window,
        )

    def reset(self):
        self.start_time = None
        self.total_time_ns = 0
        self.total_frames = 0
        self.window_index = 0
        self.start_index = 0
        self.window = [0] * self.window_size
