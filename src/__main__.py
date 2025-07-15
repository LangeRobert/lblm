from src.animations import load_animations
from src.brain import Brain
from src.controller import Controller
from src.detection.live_detector import Detector
from src.vis import Visualizer

if __name__ == '__main__':
    options = load_animations(base_path="animations", animation_folder="GLBs")

    brain = Brain()

    vis = Visualizer(options=options, queue=brain.output_queue)
    controller = Controller(options=options.keys(), output_queue=brain.input_queue)
    detector = Detector(data_queue=brain.input_queue,stop_event=brain.stop_event)

    brain.start()
    controller.start()
    detector.start()
    vis.start()

    brain.join()
    controller.join()
    detector.join()
    vis.join()