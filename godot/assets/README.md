# Mannequiny

`mannequiny.glb` is the unmodified `mannequiny-0.3.0.glb` from the official
[GDQuest 3D mannequin repository](https://github.com/gdquest-demos/godot-3d-mannequin).

- Source revision: `d48a4fdb49a3930f38cdbeb854cf5be3ae4b6b2a`
- Source path: `godot/assets/3d/mannequiny/mannequiny-0.3.0.glb`
- Attribution: **Mannequiny — GDQuest, Luciano Muñoz, and contributors**.
- Art license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Upstream license notice: [MANNEQUINY-LICENSE.txt](MANNEQUINY-LICENSE.txt).

LBLM overrides all materials with matte white at runtime, reconstructs the skin
bind pose, and drives its bones from streamed canonical motion. These runtime
adaptations are in `../humanoid.gd`; the bundled source model is unchanged.
