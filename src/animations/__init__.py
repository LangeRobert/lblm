import os

def load_animations(base_path="animations", animation_folder="transformed"):
    animations = {}

    animation_dir = os.path.join(base_path, animation_folder)
    if not os.path.isdir(animation_dir):
        return animations  # return only idle if folder doesn't exist

    for file in os.listdir(animation_dir):
        if file.lower().endswith(".glb"):
            name = os.path.splitext(file)[0]
            animations[name] = os.path.join(animation_dir, file)

    if "idle" in animations:
        idle_anim = animations.pop("idle")
        animations = {"idle": idle_anim, **animations}
    else:
        raise ValueError("No idle animation found in options. Please provide an idle animation.")

    print("loaded: ", animations.keys())
    return animations