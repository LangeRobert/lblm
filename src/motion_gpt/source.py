"""Import pinned inference modules without executing upstream training initializers."""

import importlib
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType


def import_source_module(source: Path, name: str) -> ModuleType:
    """Import model architecture or geometry under an isolated package namespace.

    HumanML3D's root initializer eagerly imports training datasets and spaCy.
    Its geometry modules only need relative imports within their own directory.
    Keep those imports intact without loading unrelated training dependencies.

    :param source: Pinned MotionGPT source snapshot or explicit trusted checkout.
    :param name: Architecture path or geometry-prefixed path within that source.
    :returns: Imported module; later calls reuse the same source and namespace.
    :raises ValueError: If a different source was already loaded in the process.
    """
    if name.startswith("geometry."):
        root = source / "mGPT" / "data" / "humanml"
        namespace = "_lblm_motiongpt_geometry"
        relative_name = name.removeprefix("geometry.")
    else:
        root = source / "mGPT"
        namespace = "_lblm_motiongpt"
        relative_name = name
    root = root.resolve()
    existing = sys.modules.get(namespace)
    if existing is None:
        package = ModuleType(namespace)
        package.__path__ = [str(root)]
        package.__spec__ = ModuleSpec(namespace, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = [str(root)]
        sys.modules[namespace] = package
    elif list(existing.__path__) != [str(root)]:
        raise ValueError("A different MotionGPT source is already loaded in this process")
    return importlib.import_module(f"{namespace}.{relative_name}")
