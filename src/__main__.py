from queue import Queue

from src.vis import Visualizer
from src.util import load_animations
from src.controller import Controller



if __name__ == '__main__':
    queue = Queue()
    options = load_animations()

    vis = Visualizer(options=options)

    # init and start the controller
    controller = Controller(options=options, output_queue=queue)
    controller.start()

    vis.run()