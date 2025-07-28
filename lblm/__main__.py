from brain import Brain
from detection.live_detector import Detector
from vis import Visualizer

if __name__ == '__main__':

    brain = Brain()

    vis = Visualizer(options=brain.options, queue=brain.output_queue,is_outputting_event=brain.is_outputting_event)
    detector = Detector(data_queue=brain.input_queue,stop_event=brain.stop_event)

    brain.start()
    detector.start()
    vis.start()

    brain.join()
    detector.join()
    vis.join()