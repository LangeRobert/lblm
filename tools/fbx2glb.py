"""
    Converts FBX Animations to GLB format using Blender's Python API.
    This script is intended to be run within Blender's scripting environment. A quick look into the documentation or
    a quick search online will help you to run this script in Blender. I recommend using the Blender CLI.
    :usage:
    1. Run the script in Blender's scripting environment. (blender --background --python fbx2glb.py)
    2. Specify the input and output folders. Absolute paths are required.
    3. The script will convert all FBX files in the input folder to GLB.

    This is a quick, dirty and hacky solution to convert FBX animations to GLB format.
    It should be noted that this script is not optimized for performance and may take a while to run and has only a
    basic error handling.
"""

if __name__ == '__main__':
    import bpy
    import os

    # Set input and output folders
    input_folder = input("Please enter the input folder: ")
    output_folder = input("Please enter the output folder: ")

    os.makedirs(output_folder, exist_ok=True)

    # Clear Blender scene
    def full_cleanup() -> None:
        """
        Clears the Blender scene by deleting all objects, actions, armatures, and meshes.
        :return: None
        """
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        for block in bpy.data.objects:
            bpy.data.objects.remove(block, do_unlink=True)
        for block in bpy.data.actions:
            bpy.data.actions.remove(block, do_unlink=True)
        for block in bpy.data.armatures:
            bpy.data.armatures.remove(block, do_unlink=True)
        for block in bpy.data.meshes:
            bpy.data.meshes.remove(block, do_unlink=True)


    index = 0

    already_converted = [tarfile.replace(".glb", "") for tarfile in os.listdir(output_folder) if
                         tarfile.lower().endswith(".glb")]
    print("Already converted animations:", len(already_converted))
    print(already_converted)

    # Batch convert FBX to GLB
    for filename in os.listdir(input_folder):
        index += 1
        animation_name = filename.replace(".fbx", "")
        print(f"Processing {index}: {filename} as {animation_name}")
        # Check if the GLB already exists
        if animation_name in already_converted:
            print(f"Skipping {animation_name} with index {index}, already converted.")
            continue

        elif filename.lower().endswith(".fbx"):
            fbx_path = os.path.join(input_folder, filename)
            glb_name = os.path.splitext(filename)[0] + ".glb"
            glb_path = os.path.join(output_folder, glb_name)

            print(f"Converting {filename} to {glb_name}")

            full_cleanup()
            bpy.ops.import_scene.fbx(filepath=fbx_path)

            # Delete everything except the armature
            for obj in bpy.data.objects:
                if obj.type != 'ARMATURE':
                    bpy.data.objects.remove(obj, do_unlink=True)

            if bpy.data.actions:
                bpy.context.object.animation_data.action = bpy.data.actions[0]

            bpy.ops.export_scene.gltf(
                filepath=glb_path,
                export_format='GLB',
                export_apply=True,
                export_animations=True,
                export_yup=True,
                export_force_sampling=True,  # Force baked animation (good for compatibility)
                use_selection=False
            )

        else:
            print(f"------- Skipping {filename}, not an FBX file.")

    print("Conversion complete.")

