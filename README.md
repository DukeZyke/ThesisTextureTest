# Water Molecule + Texture Swap Utilities

This workspace has two separate workflows:

1. **Viewer app** (root `main.py`): renders a simple H₂O scene using OBJ/MTL assets.
2. **Texture swap utility** (`texture_swap_only/swap_textures.py`): edits `map_Kd` texture paths in MTL files (no viewer).

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Viewer app

Run:

```powershell
python main.py
```

Main files:
- `main.py` (viewer + water molecule scene)
- `mesh_obj_loader.py` (OBJ/MTL loader for rendering)

Edit these two lines in `main.py` to swap assets:

```python
ATOM_MODEL_PATH = os.path.join("assets", "atomv2", "rdmblk.obj")
BOND_MODEL_PATH = os.path.join("assets", "bond", "bond.obj")
```

Optional molecule size controls in `main.py`:
- `OXYGEN_SCALE`
- `HYDROGEN_SCALE`
- `BOND_THICKNESS`

Viewer controls:
- **Left mouse drag on atom**: move atom in camera plane (can affect X/Y/Z depending on view angle)
- **Drag color swatch from left panel**: add new atom (red/blue/white)
- **Right click atom**: menu with `remove molecule`, `connect`, `resize +`, `resize -`, `reset all sizes`
- **Right click bond**: menu with `remove`, `reset all sizes`
- **Middle mouse drag**: rotate camera
- **Shift + Middle mouse drag**: pan camera (inverse to drag direction)
- **Mouse wheel**: zoom
- **R**: reset camera
- **Esc**: quit

Live update behavior:
- While app is running, edits in `texture_swap_only/swap_textures.py` (size/thickness values) are reloaded automatically.
- Use `reset all sizes` to apply updated defaults to existing atoms and bonds.

## 2) Texture swap utility (no rendering)

Run:

```powershell
python .\texture_swap_only\swap_textures.py
```

Main files:
- `texture_swap_only/swap_textures.py` (entry script)
- `texture_swap_only/mtl_texture_editor.py` (MTL `map_Kd` editor)

Edit these values in `texture_swap_only/swap_textures.py`:

```python
OBJ_PATH = os.path.join("assets", "atomv2", "rdmblk.obj")
TARGET_MATERIAL = None
NEW_TEXTURE_PATH = "rdmblk_texture.png"
UPDATE_ALL_MATERIALS = True
DRY_RUN = False
```

Notes:
- Set `DRY_RUN = True` to preview changes without writing files.
- `NEW_TEXTURE_PATH` is written into the material file as `map_Kd ...`.
- To update both atom and bond textures, run the script once per target OBJ (or change `OBJ_PATH` and run again).

## Blender export checklist (recommended)

From Blender OBJ export:
- **UV Coordinates** enabled
- **Normals** enabled
- **Triangulated Mesh** enabled
- **Materials** enabled
- **Material Groups** enabled (under Grouping)

Keep exported `.obj`, `.mtl`, and texture image files in consistent relative paths.

## Troubleshooting

- If terminal says running but no window appears, run from VS Code **Integrated Terminal**, not Output panel.
- If the object appears black, check normals and `map_Kd` texture path in `.mtl`.
