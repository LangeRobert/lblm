import os

def load_animations(animations_path="data/animations") -> dict[str,str]:
    """
    Loads all .gbl animations from the specified path.
    :param animations_path: The path to the animations folder.
    :return: A dictionary with animation names as keys and their file paths as values.
    """

    animations = {}

    if not os.path.isdir(animations_path):
        return animations  # return only idle if folder doesn't exist

    for file in os.listdir(animations_path):
        if file.lower().endswith(".glb"):
            name = os.path.splitext(file)[0]
            animations[name] = os.path.join(animations_path, file)

    if "idle" in animations:
        idle_anim = animations.pop("idle")
        animations = {"idle": idle_anim, **animations}
    else:
        raise ValueError("No idle animation found in options. Please provide an idle animation.")

    print("loaded: ", animations.keys())
    return animations