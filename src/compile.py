from src.animations import load_animations
options = load_animations(base_path="animations", animation_folder="GLBs")

if __name__ == '__main__':
    from src.animations.glb2mp4 import Converter
    converter = Converter(options)
    converter.run()