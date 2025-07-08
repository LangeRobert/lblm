

if __name__ == '__main__':
    from src.vis.vis import BodyModelVisualizer
    from src.model import BodyModel
    from src.detection import Detector

    # init the model and the shared memory
    model = BodyModel.default()
    #model.save()

    detector = Detector(model)
    detector.start()

    visualizer = BodyModelVisualizer(model)
    visualizer.run()