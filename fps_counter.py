import time

class FPSCounter:
    def __init__(self):
        self.frame_times = []
        self.max_history = 60

    def tick(self):
        t = time.perf_counter()
        self.frame_times.append(t)
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)
        if len(self.frame_times) > 1:
            fps = 1.0 / (self.frame_times[-1] - self.frame_times[0]) * len(self.frame_times)
            return round(fps)
        return 0

    def reset(self):
        self.frame_times = []

