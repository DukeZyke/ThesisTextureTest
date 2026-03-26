from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MaterialInfo:
    name: str
    texture_path: Optional[str]
    mtl_file: Path


class MtlTextureEditor:
    """
    Small utility for editing texture links in OBJ/MTL files.

    What it does:
    - Reads OBJ and its referenced MTL file(s)
    - Finds material names (`newmtl`)
    - Reads/writes `map_Kd` texture paths
    """

    def __init__(self, obj_path: str) -> None:
        self.obj_path = Path(obj_path).resolve()
        if not self.obj_path.exists():
            raise FileNotFoundError(f"OBJ not found: {self.obj_path}")

        self.obj_dir = self.obj_path.parent
        self.mtl_files = self._read_mtllibs()
        if not self.mtl_files:
            raise ValueError("No `mtllib` found in OBJ. Export with materials enabled.")

        self.materials: Dict[str, MaterialInfo] = self._read_materials()
        if not self.materials:
            raise ValueError("No materials found in referenced MTL file(s).")

    def _read_mtllibs(self) -> List[Path]:
        files: List[Path] = []
        with self.obj_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("mtllib "):
                    # OBJ can list multiple MTL files on one line
                    names = line.split()[1:]
                    for name in names:
                        p = (self.obj_dir / name).resolve()
                        if p.exists():
                            files.append(p)
        return files

    def _read_materials(self) -> Dict[str, MaterialInfo]:
        materials: Dict[str, MaterialInfo] = {}

        for mtl_file in self.mtl_files:
            current_name: Optional[str] = None
            current_texture: Optional[str] = None

            with mtl_file.open("r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue

                    if line.startswith("newmtl "):
                        if current_name is not None:
                            materials[current_name] = MaterialInfo(
                                name=current_name,
                                texture_path=current_texture,
                                mtl_file=mtl_file,
                            )
                        current_name = line.split(maxsplit=1)[1].strip()
                        current_texture = None

                    elif line.startswith("map_Kd ") and current_name is not None:
                        current_texture = line.split(maxsplit=1)[1].strip()

            if current_name is not None:
                materials[current_name] = MaterialInfo(
                    name=current_name,
                    texture_path=current_texture,
                    mtl_file=mtl_file,
                )

        return materials

    def list_materials(self) -> List[MaterialInfo]:
        return list(self.materials.values())

    def set_texture(self, material_name: str, new_texture_path: str) -> None:
        if material_name not in self.materials:
            available = ", ".join(sorted(self.materials.keys()))
            raise KeyError(f"Material '{material_name}' not found. Available: {available}")

        material = self.materials[material_name]
        self._update_map_kd_in_mtl(material.mtl_file, material_name, new_texture_path)
        material.texture_path = new_texture_path

    def set_texture_for_all(self, new_texture_path: str) -> None:
        for material_name in list(self.materials.keys()):
            self.set_texture(material_name, new_texture_path)

    def _update_map_kd_in_mtl(self, mtl_file: Path, target_material: str, new_texture_path: str) -> None:
        with mtl_file.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        in_target = False
        found_map_kd = False
        insert_index: Optional[int] = None

        for i, raw in enumerate(lines):
            stripped = raw.strip()

            if stripped.startswith("newmtl "):
                name = stripped.split(maxsplit=1)[1].strip()
                if in_target and not found_map_kd:
                    insert_at = i
                    lines.insert(insert_at, f"map_Kd {new_texture_path}\n")
                    found_map_kd = True
                    break

                in_target = (name == target_material)
                insert_index = i + 1
                continue

            if in_target and stripped.startswith("map_Kd "):
                lines[i] = f"map_Kd {new_texture_path}\n"
                found_map_kd = True
                break

        if in_target and not found_map_kd:
            if insert_index is None:
                lines.append(f"map_Kd {new_texture_path}\n")
            else:
                lines.insert(insert_index, f"map_Kd {new_texture_path}\n")

        with mtl_file.open("w", encoding="utf-8") as f:
            f.writelines(lines)
