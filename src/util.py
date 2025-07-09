import os

def load_animations(base_path="vis/models", animation_folder="animations"):
    animations = {
        "idle": os.path.join(base_path, "model.glb")
    }

    animation_dir = os.path.join(base_path, animation_folder)
    if not os.path.isdir(animation_dir):
        return animations  # return only idle if folder doesn't exist

    for file in os.listdir(animation_dir):
        if file.lower().endswith(".glb"):
            name = os.path.splitext(file)[0]
            animations[name] = os.path.join(animation_dir, file)

    print("loaded: ", animations.keys())
    return animations