import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pygame
from OpenGL.GL import (
    GL_COMPILE,
    GL_LINEAR,
    GL_RGB,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glDisable,
    glEnable,
    glEnd,
    glEndList,
    glGenLists,
    glGenTextures,
    glCallList,
    glNewList,
    glNormal3f,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glVertex3f,
)

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
FaceVertex = Tuple[int, int, int]  # (v_idx, vt_idx or -1, vn_idx or -1)


@dataclass
class Material:
    name: str
    texture_path: Optional[str] = None
    texture_id: Optional[int] = None


@dataclass
class MeshTriangle:
    material_name: Optional[str]
    vertices: Tuple[FaceVertex, FaceVertex, FaceVertex]


@dataclass
class ObjModel:
    vertices: List[Vec3]
    texcoords: List[Vec2]
    normals: List[Vec3]
    triangles: List[MeshTriangle]
    materials: Dict[str, Material] = field(default_factory=dict)
    display_list_id: Optional[int] = None

    def upload_textures(self) -> None:
        for material in self.materials.values():
            if not material.texture_path or material.texture_id is not None:
                continue

            if not os.path.exists(material.texture_path):
                continue

            surface = pygame.image.load(material.texture_path)
            width, height = surface.get_size()

            has_alpha = surface.get_masks()[3] != 0
            image_format = GL_RGBA if has_alpha else GL_RGB
            image_bytes = pygame.image.tobytes(surface, "RGBA" if has_alpha else "RGB", True)

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                image_format,
                width,
                height,
                0,
                image_format,
                GL_UNSIGNED_BYTE,
                image_bytes,
            )
            glBindTexture(GL_TEXTURE_2D, 0)

            material.texture_id = texture_id

    def _apply_material(self, material_name: Optional[str]) -> None:
        if material_name is None:
            glDisable(GL_TEXTURE_2D)
            return

        material = self.materials.get(material_name)
        if material is None or material.texture_id is None:
            glDisable(GL_TEXTURE_2D)
            return

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, material.texture_id)

    def draw(self) -> None:
        if self.display_list_id is not None:
            glCallList(self.display_list_id)
            return

        self.display_list_id = glGenLists(1)
        glNewList(self.display_list_id, GL_COMPILE)
        try:
            self._draw_immediate()
        finally:
            glEndList()

    def _draw_immediate(self) -> None:
        current_material: Optional[str] = None

        for triangle in self.triangles:
            if triangle.material_name != current_material:
                self._apply_material(triangle.material_name)
                current_material = triangle.material_name

            p0 = self.vertices[triangle.vertices[0][0]]
            p1 = self.vertices[triangle.vertices[1][0]]
            p2 = self.vertices[triangle.vertices[2][0]]
            face_normal = normalize(cross(subtract(p1, p0), subtract(p2, p0)))

            glBegin(GL_TRIANGLES)
            for v_idx, vt_idx, vn_idx in triangle.vertices:
                if 0 <= vn_idx < len(self.normals):
                    nx, ny, nz = self.normals[vn_idx]
                else:
                    nx, ny, nz = face_normal

                if 0 <= vt_idx < len(self.texcoords):
                    u, v = self.texcoords[vt_idx]
                    glTexCoord2f(u, v)

                x, y, z = self.vertices[v_idx]
                glNormal3f(nx, ny, nz)
                glVertex3f(x, y, z)
            glEnd()

        glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)


def subtract(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(v: Vec3) -> Vec3:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length == 0:
        return (0.0, 0.0, 1.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def resolve_obj_index(raw: str, count: int) -> int:
    idx = int(raw)
    return idx - 1 if idx > 0 else count + idx


def parse_face_token(token: str, v_count: int, vt_count: int, vn_count: int) -> FaceVertex:
    parts = token.split("/")

    v_idx = resolve_obj_index(parts[0], v_count)

    vt_idx = -1
    if len(parts) >= 2 and parts[1] != "":
        vt_idx = resolve_obj_index(parts[1], vt_count)

    vn_idx = -1
    if len(parts) >= 3 and parts[2] != "":
        vn_idx = resolve_obj_index(parts[2], vn_count)

    return (v_idx, vt_idx, vn_idx)


def triangulate(vertices: List[FaceVertex]) -> List[Tuple[FaceVertex, FaceVertex, FaceVertex]]:
    triangles: List[Tuple[FaceVertex, FaceVertex, FaceVertex]] = []
    for i in range(1, len(vertices) - 1):
        triangles.append((vertices[0], vertices[i], vertices[i + 1]))
    return triangles


def load_mtl(mtl_path: str) -> Dict[str, Material]:
    materials: Dict[str, Material] = {}
    current: Optional[Material] = None

    if not os.path.exists(mtl_path):
        return materials

    mtl_dir = os.path.dirname(mtl_path)
    with open(mtl_path, "r", encoding="utf-8", errors="replace") as mtl_file:
        for raw in mtl_file:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("newmtl "):
                name = line.split(maxsplit=1)[1].strip()
                current = Material(name=name)
                materials[name] = current
            elif line.startswith("map_Kd ") and current is not None:
                texture_rel = line.split(maxsplit=1)[1].strip()
                current.texture_path = os.path.join(mtl_dir, texture_rel)

    return materials


def load_obj(obj_path: str) -> ObjModel:
    vertices: List[Vec3] = []
    texcoords: List[Vec2] = []
    normals: List[Vec3] = []
    triangles: List[MeshTriangle] = []
    materials: Dict[str, Material] = {}

    current_material: Optional[str] = None
    obj_dir = os.path.dirname(obj_path)

    with open(obj_path, "r", encoding="utf-8", errors="replace") as obj_file:
        for raw in obj_file:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("mtllib "):
                mtl_rel = line.split(maxsplit=1)[1].strip()
                mtl_path = os.path.join(obj_dir, mtl_rel)
                materials.update(load_mtl(mtl_path))

            elif line.startswith("usemtl "):
                current_material = line.split(maxsplit=1)[1].strip()

            elif line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    texcoords.append((float(parts[1]), float(parts[2])))

            elif line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    normals.append(normalize((float(parts[1]), float(parts[2]), float(parts[3]))))

            elif line.startswith("f "):
                tokens = line.split()[1:]
                face_vertices: List[FaceVertex] = []

                for token in tokens:
                    try:
                        parsed = parse_face_token(token, len(vertices), len(texcoords), len(normals))
                        if 0 <= parsed[0] < len(vertices):
                            face_vertices.append(parsed)
                    except (ValueError, IndexError):
                        continue

                if len(face_vertices) < 3:
                    continue

                for tri in triangulate(face_vertices):
                    triangles.append(MeshTriangle(material_name=current_material, vertices=tri))

    if not vertices or not triangles:
        raise ValueError(f"OBJ empty or unsupported: {obj_path}")

    return ObjModel(
        vertices=vertices,
        texcoords=texcoords,
        normals=normals,
        triangles=triangles,
        materials=materials,
    )
