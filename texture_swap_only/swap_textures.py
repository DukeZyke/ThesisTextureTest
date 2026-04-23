import os

try:
    from .mtl_texture_editor import MtlTextureEditor
except ImportError:
    from mtl_texture_editor import MtlTextureEditor


# ---------------------------------------------------------
# SHARED SETTINGS (single source of truth)
# ---------------------------------------------------------
ATOM_MODEL_PATH = os.path.join("assets", "atom", "atom.obj")
BOND_MODEL_PATH = os.path.join("assets", "bond", "bond.obj")

MATERIAL_LIBRARY_ROOT = "assets/atom"
USE_ATOM_KIND_OVERRIDES = True
USE_ATOM_MTL_OVERRIDES = True

CPK_BASE_RADIUS = 1.7
BOND_THICKNESS = 0.085

OXYGEN_SCALE = 1.52 # CPK O
HYDROGEN_SCALE = 1.10 # CPK H
# Auto-applied via molecular_engine.cpk_radius()


# ---------------------------------------------------------
# TEXTURE SWAP SETTINGS
# ---------------------------------------------------------
OBJ_PATH = ATOM_MODEL_PATH
TARGET_MATERIAL = None  # e.g. "AtomMat"; use None to update all materials
NEW_TEXTURE_PATH = "rdmblk_texture.png"  # relative path written into .mtl
UPDATE_ALL_MATERIALS = True
DRY_RUN = False  # True = preview only (no file write)


def resolve_from_project(path_value: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    return os.path.abspath(os.path.join(project_root, path_value))


def main() -> None:
    obj_file = resolve_from_project(OBJ_PATH)
    editor = MtlTextureEditor(obj_file)

    print(f"OBJ: {obj_file}")
    print("Materials found:")
    for mat in editor.list_materials():
        print(f"- {mat.name} | current map_Kd: {mat.texture_path} | file: {mat.mtl_file}")

    if UPDATE_ALL_MATERIALS or TARGET_MATERIAL is None:
        if DRY_RUN:
            print(f"\n[DRY RUN] Would update all materials to map_Kd: {NEW_TEXTURE_PATH}")
        else:
            editor.set_texture_for_all(NEW_TEXTURE_PATH)
            print(f"\nUpdated all materials to map_Kd: {NEW_TEXTURE_PATH}")
    else:
        if DRY_RUN:
            print(f"\n[DRY RUN] Would update material '{TARGET_MATERIAL}' to map_Kd: {NEW_TEXTURE_PATH}")
        else:
            editor.set_texture(TARGET_MATERIAL, NEW_TEXTURE_PATH)
            print(f"\nUpdated material '{TARGET_MATERIAL}' to map_Kd: {NEW_TEXTURE_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()
