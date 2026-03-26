import math
import os
import sys
import importlib
from dataclasses import dataclass
from typing import Optional, Tuple

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import (
    GL_AMBIENT_AND_DIFFUSE,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_DEPTH_TEST,
    GL_DEPTH_BUFFER_BIT,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_MODELVIEW,
    GL_NORMALIZE,
    GL_POSITION,
    GL_PROJECTION,
    GL_QUADS,
    GL_SMOOTH,
    GL_TEXTURE_2D,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glColorMaterial,
    glDisable,
    glEnd,
    glEnable,
    glLoadIdentity,
    glLightfv,
    glMatrixMode,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glScalef,
    glShadeModel,
    glTranslatef,
    glVertex2f,
)
from OpenGL.GLU import gluPerspective

from mesh_obj_loader import ObjModel, load_obj
import texture_swap_only.swap_textures as swap_config


# Shared model paths and size controls are defined in:
# texture_swap_only/swap_textures.py


Vec3 = Tuple[float, float, float]

# 3x5 bitmap font for robust OpenGL overlay text (no texture flipping issues).
BITMAP_FONT: dict[str, list[str]] = {
    "A": ["010", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"],
    "C": ["011", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "110", "100", "111"],
    "F": ["111", "100", "110", "100", "100"],
    "G": ["011", "100", "101", "101", "011"],
    "H": ["101", "101", "111", "101", "101"],
    "I": ["111", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "101", "010"],
    "K": ["101", "101", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"],
    "N": ["101", "111", "111", "111", "101"],
    "O": ["010", "101", "101", "101", "010"],
    "P": ["110", "101", "110", "100", "100"],
    "Q": ["010", "101", "101", "111", "011"],
    "R": ["110", "101", "110", "101", "101"],
    "S": ["011", "100", "010", "001", "110"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "010"],
    "W": ["101", "101", "111", "111", "101"],
    "X": ["101", "101", "010", "101", "101"],
    "Y": ["101", "101", "010", "010", "010"],
    "Z": ["111", "001", "010", "100", "111"],
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "+": ["000", "010", "111", "010", "000"],
    "-": ["000", "000", "111", "000", "000"],
    " ": ["000", "000", "000", "000", "000"],
    "?": ["111", "001", "010", "000", "010"],
}


@dataclass
class AtomState:
    position: list[float]
    scale: float
    default_scale: float
    tint: Vec3


@dataclass
class BondLink:
    atom_a: int
    atom_b: int
    thickness: float
    default_thickness: float
    tint: Vec3


@dataclass
class BondMeshProfile:
    x_min: float
    x_max: float

    @property
    def span(self) -> float:
        return max(self.x_max - self.x_min, 0.000001)


@dataclass
class PaletteItem:
    name: str
    tint: Vec3
    scale: float
    center: Tuple[int, int]
    radius: int


def setup_opengl(width: int, height: int) -> None:
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, width / max(height, 1), 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glLightfv(GL_LIGHT0, GL_POSITION, (6.0, 8.0, 10.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))

    glShadeModel(GL_SMOOTH)
    glClearColor(0.08, 0.08, 0.10, 1.0)


def draw_atom(atom_model: ObjModel, atom: AtomState) -> None:
    glPushMatrix()
    glTranslatef(atom.position[0], atom.position[1], atom.position[2])
    glScalef(atom.scale, atom.scale, atom.scale)
    glColor3f(atom.tint[0], atom.tint[1], atom.tint[2])
    atom_model.draw()
    glPopMatrix()


def get_bond_mesh_profile(bond_model: ObjModel) -> BondMeshProfile:
    x_values = [v[0] for v in bond_model.vertices]
    if not x_values:
        return BondMeshProfile(-0.5, 0.5)
    return BondMeshProfile(min(x_values), max(x_values))


def draw_bond_between(
    bond_model: ObjModel,
    profile: BondMeshProfile,
    a: AtomState,
    b: AtomState,
    thickness: float,
    tint: Vec3,
) -> None:
    dx = b.position[0] - a.position[0]
    dy = b.position[1] - a.position[1]
    dz = b.position[2] - a.position[2]
    length = max(math.sqrt(dx * dx + dy * dy + dz * dz), 0.001)

    # Align local +X axis to A->B direction in full 3D.
    ux, uy, uz = dx / length, dy / length, dz / length
    dot = max(-1.0, min(1.0, ux))
    angle = math.degrees(math.acos(dot))
    axis_x, axis_y, axis_z = 0.0, -uz, uy
    axis_len = math.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)

    glPushMatrix()
    glTranslatef(a.position[0], a.position[1], a.position[2])
    if axis_len > 0.000001:
        glRotatef(angle, axis_x / axis_len, axis_y / axis_len, axis_z / axis_len)
    elif ux < 0.0:
        glRotatef(180.0, 0.0, 1.0, 0.0)
    glScalef(length / profile.span, thickness, thickness)
    glTranslatef(-profile.x_min, 0.0, 0.0)
    glColor3f(tint[0], tint[1], tint[2])
    bond_model.draw()
    glPopMatrix()


def draw_overlay_rect(x: float, y: float, w: float, h: float, color: Vec3) -> None:
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def draw_overlay_text(text: str, x: float, y: float, color: Tuple[int, int, int] = (240, 240, 245), pixel: int = 3) -> None:
    r, g, b = color
    color_f = (r / 255.0, g / 255.0, b / 255.0)
    cursor_x = x

    for ch in text.upper():
        glyph = BITMAP_FONT.get(ch, BITMAP_FONT["?"])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    draw_overlay_rect(cursor_x + col * pixel, y + row * pixel, pixel, pixel, color_f)
        cursor_x += 4 * pixel


def draw_2d_overlay(
    width: int,
    height: int,
    palette: list[PaletteItem],
    context_menu: Optional[dict],
    dragging_preview: Optional[Tuple[int, int, Vec3]],
) -> None:
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_TEXTURE_2D)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Palette background strip.
    draw_overlay_rect(16, 24, 112, 300, (0.12, 0.12, 0.16))
    draw_overlay_text("PALETTE", 26, 34)
    for item in palette:
        size = item.radius * 2
        draw_overlay_rect(item.center[0] - item.radius, item.center[1] - item.radius, size, size, item.tint)
        draw_overlay_text(item.name, 26, item.center[1] + 22)

    # Optional drag preview square.
    if dragging_preview is not None:
        px, py, tint = dragging_preview
        draw_overlay_rect(px - 12, py - 12, 24, 24, tint)

    # Context menu boxes.
    if context_menu is not None:
        mx, my = context_menu["pos"]
        option_count = len(context_menu["options"])
        for i in range(option_count):
            color = (0.20, 0.24, 0.30) if i % 2 == 0 else (0.16, 0.20, 0.26)
            draw_overlay_rect(mx, my + i * 30, 150, 28, color)
            draw_overlay_text(context_menu["options"][i], mx + 10, my + i * 30 + 6)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


def atom_position(angle_deg: float, bond_len: float) -> Vec3:
    a = math.radians(angle_deg)
    return (math.sin(a) * bond_len, math.cos(a) * bond_len, 0.0)


def create_water_molecule(oxygen_scale: float, hydrogen_scale: float, bond_thickness: float) -> Tuple[list[AtomState], list[BondLink]]:
    oxygen_pos = (0.0, 0.0, 0.0)
    bond_len = 1.30
    half_angle = 52.25  # 104.5 / 2

    h1 = atom_position(+half_angle, bond_len)
    h2 = atom_position(-half_angle, bond_len)

    atoms = [
        AtomState([oxygen_pos[0], oxygen_pos[1], oxygen_pos[2]], oxygen_scale, oxygen_scale, (1.0, 0.25, 0.25)),
        AtomState([h1[0], h1[1], h1[2]], hydrogen_scale, hydrogen_scale, (0.9, 0.95, 1.0)),
        AtomState([h2[0], h2[1], h2[2]], hydrogen_scale, hydrogen_scale, (0.9, 0.95, 1.0)),
    ]

    bonds = [
        BondLink(atom_a=0, atom_b=1, thickness=bond_thickness, default_thickness=bond_thickness, tint=(0.8, 0.85, 0.95)),
        BondLink(atom_a=0, atom_b=2, thickness=bond_thickness, default_thickness=bond_thickness, tint=(0.8, 0.85, 0.95)),
    ]

    return atoms, bonds


def rotate_around_y(v: Vec3, angle_deg: float) -> Vec3:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)


def rotate_around_x(v: Vec3, angle_deg: float) -> Vec3:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return (v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c)


def world_to_screen(
    point: Vec3,
    width: int,
    height: int,
    rotate_x: float,
    rotate_y: float,
    zoom: float,
    camera_pan: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    # Matches current camera order in render loop: translate -> rotateX -> rotateY.
    p = rotate_around_y(point, rotate_y)
    p = rotate_around_x(p, rotate_x)
    cam_x, cam_y, cam_z = p[0] + camera_pan[0], p[1] + camera_pan[1], p[2] + zoom

    if cam_z >= -0.1:
        return None

    fov = math.radians(60.0)
    aspect = width / max(height, 1)
    x_ndc = (cam_x / -cam_z) / (math.tan(fov * 0.5) * aspect)
    y_ndc = (cam_y / -cam_z) / math.tan(fov * 0.5)

    screen_x = (x_ndc + 1.0) * 0.5 * width
    screen_y = (1.0 - y_ndc) * 0.5 * height
    return (screen_x, screen_y)


def pick_atom(
    atoms: list[AtomState],
    mouse_pos: Tuple[int, int],
    width: int,
    height: int,
    rotate_x: float,
    rotate_y: float,
    zoom: float,
    camera_pan: Tuple[float, float],
) -> Optional[int]:
    mx, my = mouse_pos
    best_idx: Optional[int] = None
    best_dist = 999999.0
    pick_radius_px = 28.0

    for idx, atom in enumerate(atoms):
        screen = world_to_screen((atom.position[0], atom.position[1], atom.position[2]), width, height, rotate_x, rotate_y, zoom, camera_pan)
        if screen is None:
            continue
        dx = screen[0] - mx
        dy = screen[1] - my
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < pick_radius_px and dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx


def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 0.000001:
        return math.sqrt(apx * apx + apy * apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + abx * t
    cy = ay + aby * t
    dx = px - cx
    dy = py - cy
    return math.sqrt(dx * dx + dy * dy)


def pick_bond(
    atoms: list[AtomState],
    bonds: list[BondLink],
    mouse_pos: Tuple[int, int],
    width: int,
    height: int,
    rotate_x: float,
    rotate_y: float,
    zoom: float,
    camera_pan: Tuple[float, float],
) -> Optional[int]:
    mx, my = mouse_pos
    best_idx: Optional[int] = None
    best_dist = 999999.0

    for idx, bond in enumerate(bonds):
        a = atoms[bond.atom_a]
        b = atoms[bond.atom_b]
        sa = world_to_screen((a.position[0], a.position[1], a.position[2]), width, height, rotate_x, rotate_y, zoom, camera_pan)
        sb = world_to_screen((b.position[0], b.position[1], b.position[2]), width, height, rotate_x, rotate_y, zoom, camera_pan)
        if sa is None or sb is None:
            continue
        dist = point_to_segment_distance(mx, my, sa[0], sa[1], sb[0], sb[1])
        if dist < 14.0 and dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx


def world_units_per_pixel(zoom: float, height: int) -> float:
    distance = max(abs(zoom), 0.1)
    view_height = 2.0 * distance * math.tan(math.radians(60.0) * 0.5)
    return view_height / max(height, 1)


def camera_to_world_vector(v: Vec3, rotate_x: float, rotate_y: float) -> Vec3:
    # Inverse of world->camera rotation used in world_to_screen.
    p = rotate_around_x(v, -rotate_x)
    p = rotate_around_y(p, -rotate_y)
    return p


def screen_to_world_camera_plane(
    mouse_pos: Tuple[int, int],
    width: int,
    height: int,
    zoom: float,
    rotate_x: float,
    rotate_y: float,
    camera_pan: Tuple[float, float],
) -> Vec3:
    scale = world_units_per_pixel(zoom, height)
    cam_x = (mouse_pos[0] - width * 0.5) * scale - camera_pan[0]
    cam_y = (height * 0.5 - mouse_pos[1]) * scale - camera_pan[1]
    return camera_to_world_vector((cam_x, cam_y, 0.0), rotate_x, rotate_y)


def remove_atom(atoms: list[AtomState], bonds: list[BondLink], atom_idx: int) -> None:
    del atoms[atom_idx]
    kept: list[BondLink] = []
    for bond in bonds:
        if bond.atom_a == atom_idx or bond.atom_b == atom_idx:
            continue
        a = bond.atom_a - 1 if bond.atom_a > atom_idx else bond.atom_a
        b = bond.atom_b - 1 if bond.atom_b > atom_idx else bond.atom_b
        kept.append(BondLink(a, b, bond.thickness, bond.default_thickness, bond.tint))
    bonds[:] = kept


def has_bond(bonds: list[BondLink], a: int, b: int) -> bool:
    for bond in bonds:
        if (bond.atom_a == a and bond.atom_b == b) or (bond.atom_a == b and bond.atom_b == a):
            return True
    return False


def absolute_from_project(relative_path: str) -> str:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_dir, relative_path)


def load_runtime_settings() -> dict:
    return {
        "atom_model_path": swap_config.ATOM_MODEL_PATH,
        "bond_model_path": swap_config.BOND_MODEL_PATH,
        "oxygen_scale": float(swap_config.OXYGEN_SCALE),
        "hydrogen_scale": float(swap_config.HYDROGEN_SCALE),
        "bond_thickness": float(swap_config.BOND_THICKNESS),
    }


def apply_default_sizes(atoms: list[AtomState], bonds: list[BondLink]) -> None:
    for atom in atoms:
        atom.scale = atom.default_scale
    for bond in bonds:
        bond.thickness = bond.default_thickness


def refresh_defaults_from_settings(
    atoms: list[AtomState],
    bonds: list[BondLink],
    palette: list[PaletteItem],
    settings: dict,
) -> None:
    palette[0].scale = settings["oxygen_scale"]
    palette[1].scale = settings["hydrogen_scale"]
    palette[2].scale = settings["hydrogen_scale"]

    # Keep existing atoms in place, but update what "default" means.
    for atom in atoms:
        if atom.tint == palette[0].tint:
            atom.default_scale = settings["oxygen_scale"]
        else:
            atom.default_scale = settings["hydrogen_scale"]

    for bond in bonds:
        bond.default_thickness = settings["bond_thickness"]


def main() -> None:
    settings = load_runtime_settings()
    atom_obj = absolute_from_project(settings["atom_model_path"])
    bond_obj = absolute_from_project(settings["bond_model_path"])

    if not os.path.exists(atom_obj):
        print(f"Missing atom OBJ: {atom_obj}")
        sys.exit(1)

    if not os.path.exists(bond_obj):
        print(f"Missing bond OBJ: {bond_obj}")
        sys.exit(1)

    pygame.init()
    width, height = 1280, 720
    try:
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    except pygame.error as exc:
        print("Failed to create OpenGL window.")
        print(f"Pygame error: {exc}")
        print("Run in VS Code integrated terminal and verify graphics/OpenGL drivers.")
        sys.exit(1)

    pygame.display.set_caption("Water Molecule - OBJ + MTL Textures")
    setup_opengl(width, height)

    atom_model = load_obj(atom_obj)
    bond_model = load_obj(bond_obj)
    bond_profile = get_bond_mesh_profile(bond_model)
    atom_model.upload_textures()
    bond_model.upload_textures()

    print(f"Atom OBJ: {atom_obj}")
    print(f"Bond OBJ: {bond_obj}")
    print("Left-drag atom to move it. Bonds stretch automatically with distance.")
    print("Right-drag to orbit camera. Mouse wheel to zoom.")

    atoms, bonds = create_water_molecule(
        settings["oxygen_scale"],
        settings["hydrogen_scale"],
        settings["bond_thickness"],
    )

    palette = [
        PaletteItem("red", (1.0, 0.25, 0.25), settings["oxygen_scale"], (72, 84), 18),
        PaletteItem("blue", (0.35, 0.55, 1.0), settings["hydrogen_scale"], (72, 144), 18),
        PaletteItem("white", (0.95, 0.95, 0.95), settings["hydrogen_scale"], (72, 204), 18),
    ]

    config_path = absolute_from_project(os.path.join("texture_swap_only", "swap_textures.py"))
    config_mtime = os.path.getmtime(config_path) if os.path.exists(config_path) else 0.0
    last_reload_check_ms = 0

    clock = pygame.time.Clock()
    rotate_x = 16.0
    rotate_y = -22.0
    zoom = -5.8
    camera_pan = [0.0, 0.0]
    rotating_camera = False
    panning_camera = False
    dragging_atom_idx: Optional[int] = None
    dragging_new_atom: Optional[PaletteItem] = None
    connect_source_idx: Optional[int] = None
    context_menu: Optional[dict] = None
    overlay_drag_preview: Optional[Tuple[int, int, Vec3]] = None
    last_mouse = (0, 0)

    running = True
    while running:
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Handle context menu click first.
                    if context_menu is not None:
                        mx, my = context_menu["pos"]
                        handled_menu = False
                        for i, option in enumerate(context_menu["options"]):
                            if mx <= event.pos[0] <= mx + 150 and my + i * 30 <= event.pos[1] <= my + i * 30 + 28:
                                handled_menu = True
                                if context_menu["type"] == "atom":
                                    atom_idx = context_menu["target"]
                                    if option == "remove molecule" and 0 <= atom_idx < len(atoms):
                                        remove_atom(atoms, bonds, atom_idx)
                                        connect_source_idx = None
                                    elif option == "connect":
                                        connect_source_idx = atom_idx
                                    elif option == "resize +" and 0 <= atom_idx < len(atoms):
                                        atoms[atom_idx].scale *= 1.15
                                    elif option == "resize -" and 0 <= atom_idx < len(atoms):
                                        atoms[atom_idx].scale = max(atoms[atom_idx].scale * 0.87, 0.02)
                                    elif option == "reset all sizes":
                                        apply_default_sizes(atoms, bonds)
                                elif context_menu["type"] == "bond":
                                    bond_idx = context_menu["target"]
                                    if option == "remove" and 0 <= bond_idx < len(bonds):
                                        del bonds[bond_idx]
                                    elif option == "reset all sizes":
                                        apply_default_sizes(atoms, bonds)
                                context_menu = None
                                pygame.display.set_caption("Water Molecule - OBJ + MTL Textures")
                                break
                        if handled_menu:
                            continue
                        context_menu = None

                    # If waiting for connection target, choose target atom via left click.
                    if connect_source_idx is not None:
                        target_idx = pick_atom(atoms, event.pos, width, height, rotate_x, rotate_y, zoom, (camera_pan[0], camera_pan[1]))
                        if target_idx is not None and target_idx != connect_source_idx and not has_bond(bonds, connect_source_idx, target_idx):
                            bonds.append(
                                BondLink(
                                    connect_source_idx,
                                    target_idx,
                                    settings["bond_thickness"],
                                    settings["bond_thickness"],
                                    (0.8, 0.85, 0.95),
                                )
                            )
                        connect_source_idx = None
                        pygame.display.set_caption("Water Molecule - OBJ + MTL Textures")
                        continue

                    # Start dragging from side palette.
                    picked_palette = None
                    for item in palette:
                        dx = event.pos[0] - item.center[0]
                        dy = event.pos[1] - item.center[1]
                        if dx * dx + dy * dy <= item.radius * item.radius:
                            picked_palette = item
                            break
                    if picked_palette is not None:
                        dragging_new_atom = picked_palette
                        overlay_drag_preview = (event.pos[0], event.pos[1], picked_palette.tint)
                        continue

                    # Drag existing atom.
                    dragging_atom_idx = pick_atom(atoms, event.pos, width, height, rotate_x, rotate_y, zoom, (camera_pan[0], camera_pan[1]))
                    last_mouse = event.pos
                elif event.button == 3:
                    # Context menu: atom has remove/connect, bond has remove.
                    atom_idx = pick_atom(atoms, event.pos, width, height, rotate_x, rotate_y, zoom, (camera_pan[0], camera_pan[1]))
                    if atom_idx is not None:
                        context_menu = {
                            "type": "atom",
                            "target": atom_idx,
                            "pos": event.pos,
                            "options": ["remove molecule", "connect", "resize +", "resize -", "reset all sizes"],
                        }
                        pygame.display.set_caption("Atom menu opened")
                        continue

                    bond_idx = pick_bond(atoms, bonds, event.pos, width, height, rotate_x, rotate_y, zoom, (camera_pan[0], camera_pan[1]))
                    if bond_idx is not None:
                        context_menu = {
                            "type": "bond",
                            "target": bond_idx,
                            "pos": event.pos,
                            "options": ["remove", "reset all sizes"],
                        }
                        pygame.display.set_caption("Bond menu opened")
                        continue

                    context_menu = None
                elif event.button == 4:
                    zoom += 0.35
                elif event.button == 5:
                    zoom -= 0.35
                elif event.button == 2:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        panning_camera = True
                        rotating_camera = False
                    else:
                        rotating_camera = True
                        panning_camera = False
                    last_mouse = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_atom_idx = None
                if dragging_new_atom is not None:
                    world = screen_to_world_camera_plane(
                        event.pos,
                        width,
                        height,
                        zoom,
                        rotate_x,
                        rotate_y,
                        (camera_pan[0], camera_pan[1]),
                    )
                    atoms.append(
                        AtomState(
                            [world[0], world[1], world[2]],
                            dragging_new_atom.scale,
                            dragging_new_atom.scale,
                            dragging_new_atom.tint,
                        )
                    )
                    dragging_new_atom = None
                    overlay_drag_preview = None

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                rotating_camera = False
                panning_camera = False

            elif event.type == pygame.MOUSEMOTION and dragging_atom_idx is not None:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]

                move_scale = world_units_per_pixel(zoom, height)
                d_world = camera_to_world_vector((dx * move_scale, -dy * move_scale, 0.0), rotate_x, rotate_y)
                atoms[dragging_atom_idx].position[0] += d_world[0]
                atoms[dragging_atom_idx].position[1] += d_world[1]
                atoms[dragging_atom_idx].position[2] += d_world[2]

                last_mouse = event.pos

            elif event.type == pygame.MOUSEMOTION and dragging_new_atom is not None:
                overlay_drag_preview = (event.pos[0], event.pos[1], dragging_new_atom.tint)

            elif event.type == pygame.MOUSEMOTION and rotating_camera:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                rotate_y += dx * 0.45
                rotate_x += dy * 0.45
                last_mouse = event.pos

            elif event.type == pygame.MOUSEMOTION and panning_camera:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                pan_scale = world_units_per_pixel(zoom, height)
                # Inverse camera drag: dragging right moves camera left.
                camera_pan[0] += dx * pan_scale
                camera_pan[1] -= dy * pan_scale
                last_mouse = event.pos

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    rotate_x, rotate_y, zoom = 16.0, -22.0, -5.8
                    camera_pan[0], camera_pan[1] = 0.0, 0.0
                elif event.key == pygame.K_c and context_menu is not None and context_menu["type"] == "atom":
                    connect_source_idx = context_menu["target"]
                    context_menu = None
                    pygame.display.set_caption("Connect mode: left-click another atom to create bond")
                elif event.key == pygame.K_DELETE and context_menu is not None:
                    if context_menu["type"] == "atom":
                        atom_idx = context_menu["target"]
                        if 0 <= atom_idx < len(atoms):
                            remove_atom(atoms, bonds, atom_idx)
                            connect_source_idx = None
                    elif context_menu["type"] == "bond":
                        bond_idx = context_menu["target"]
                        if 0 <= bond_idx < len(bonds):
                            del bonds[bond_idx]
                    context_menu = None
                    pygame.display.set_caption("Water Molecule - OBJ + MTL Textures")

        now_ms = pygame.time.get_ticks()
        if now_ms - last_reload_check_ms > 400:
            last_reload_check_ms = now_ms
            if os.path.exists(config_path):
                new_mtime = os.path.getmtime(config_path)
                if new_mtime != config_mtime:
                    config_mtime = new_mtime
                    importlib.reload(swap_config)
                    settings = load_runtime_settings()
                    refresh_defaults_from_settings(atoms, bonds, palette, settings)
                    pygame.display.set_caption("Config reloaded from swap_textures.py")

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(camera_pan[0], camera_pan[1], 0.0)
        glTranslatef(0.0, 0.0, zoom)
        glRotatef(rotate_x, 1.0, 0.0, 0.0)
        glRotatef(rotate_y, 0.0, 1.0, 0.0)

        for bond in bonds:
            draw_bond_between(bond_model, bond_profile, atoms[bond.atom_a], atoms[bond.atom_b], bond.thickness, bond.tint)

        for atom in atoms:
            draw_atom(atom_model, atom)

        draw_2d_overlay(width, height, palette, context_menu, overlay_drag_preview)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
