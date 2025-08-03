from .brain import Brain
from .detector import Detector
from .visualizer import Visualizer
import click

@click.command()
@click.option('--light',is_flag=True, default=False, help="If the visualisation should run in light mode")
@click.option('--loop',is_flag=True, default=False, help="If the visualiser should only loop through the animations")
def main(light:bool,loop:bool):
    """
    Main entry point for the LBLM application
    """
    brain = Brain()

    vis = Visualizer(options=brain.options, queue=brain.output_queue, is_outputting_event=brain.is_outputting_event, light=light,loop=loop)

    if not loop:
        detector = Detector(data_queue=brain.input_queue, stop_event=brain.stop_event)
    else:
        detector = None
    brain.start()
    vis.start()
    if detector:
        detector.start()
    brain.join()
    if detector:
        detector.join()
    vis.join()


if __name__ == '__main__':
    main()