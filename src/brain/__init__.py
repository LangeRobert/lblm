import threading
import time
from multiprocessing import Queue, Event

class Brain(threading.Thread):
    def __init__(self):
        super().__init__()
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.stop_event = Event()

    def run(self):
        while True:
            time.sleep(20)  # Simulate processing time