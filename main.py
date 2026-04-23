import math
import os
import sys
import importlib
from dataclasses import dataclass
from typing import Optional, Tuple

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import (
    GL_BLEND,
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
    GL_SPECULAR,
    GL_SMOOTH,
    GL_TRIANGLES,
    GL_TEXTURE_2D,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    glColor3f,
    glColor4f,
    glBegin,
    glBlendFunc,
    glClear,
    glClearColor,
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
    glNormal3f,
    glTexCoord2f,
    glTranslatef,
    glVertex2f,
    glVertex3f,
)
from OpenGL.GLU import gluPerspective

from mesh_obj_loader import MeshTriangle, ObjModel, load_mtl, load_obj
import texture_swap_only.swap_textures as swap_config
import numpy as np
from hdri_skybox import HDRISkybox
from molecular_engine import MolecularEngine, MoleculePresets, cpk_color, cpk_radius
from molecule_hud_2d import MoleculeHUD2D
from live_angles import get_visible_angles




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
    ".": ["000", "000", "000", "000", "010"],
    "°": ["110", "110", "000", "000", "000"],
    " ": ["000", "000", "000", "000", "000"],
}

PALETTE_LABELS = {
    "red": "OXYGEN",
    "blue": "NITROGEN",
    "white": "HYDROGEN",
    "black": "CARBON",
}


@dataclass
class AtomState:
    position: list[float]
    scale: float
    default_scale: float
    tint: Vec3
    kind: str


@dataclass
class BondLink:
    atom_a: int
    atom_b: int
    thickness: float
    default_thickness: float
    tint: Vec3
    kind: str


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


@dataclass
class BottomNotification:
    duration_ms: int = 2000
    message: str = ""
    expires_at_ms: int = 0

    def trigger(self, message: str, now_ms: int) -> None:
        self.message = str(message)
        self.expires_at_ms = int(now_ms) + int(self.duration_ms)

    def draw(self, width: int, height: int, now_ms: int) -> None:
        if not self.message or int(now_ms) >= self.expires_at_ms:
            return

        remaining = max(self.expires_at_ms - int(now_ms), 0)
        t = remaining / max(self.duration_ms, 1)
        alpha = 0.20 + (0.72 * t)

        bar_w = min(520, width - 24)
        bar_h = 34
        x = 12
        y = height - bar_h - 10

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        draw_overlay_rect_alpha(x, y, bar_w, bar_h, (0.82, 0.12, 0.12), alpha)
        draw_overlay_text(self.message, x + 10, y + 10)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)


def set_projection(width: int, height: int, zoom: float) -> None:
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = width / max(height, 1)
    gluPerspective(45.0, aspect, 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)


def setup_opengl(width: int, height: int) -> None:
    set_projection(width, height, -5.8)

    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glLightfv(GL_LIGHT0, GL_POSITION, (6.0, 8.0, 10.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))

    glShadeModel(GL_SMOOTH)
    glClearColor(0.08, 0.08, 0.10, 1.0)


def hud_toggle_button_rect(width: int) -> Tuple[int, int, int, int]:
    btn_w, btn_h = 260, 26
    x = width - btn_w - 16
    y = 24
    return (x, y, btn_w, btn_h)


def view_mode_button_rect(width: int) -> Tuple[int, int, int, int]:
    btn_w, btn_h = 260, 26
    x = width - btn_w - 16
    _, hud_y, _, hud_h = hud_toggle_button_rect(width)
    y = hud_y + hud_h + 8
    return (x, y, btn_w, btn_h)


def rearrange_button_rect(width: int) -> Tuple[int, int, int, int]:
    btn_w, btn_h = 260, 28
    x = width - btn_w - 16
    _, view_y, _, view_h = view_mode_button_rect(width)
    y = view_y + view_h + 8
    return (x, y, btn_w, btn_h)


def drag_mode_button_rect(width: int) -> Tuple[int, int, int, int]:
    btn_w, btn_h = 260, 26
    x = width - btn_w - 16
    _, rearrange_y, _, _ = rearrange_button_rect(width)
    y = rearrange_y + 36
    return (x, y, btn_w, btn_h)


def preset_button_rects(width: int) -> dict[str, Tuple[int, int, int, int]]:
    btn_w, btn_h = 260, 26
    x = width - btn_w - 16
    _, drag_y, _, drag_h = drag_mode_button_rect(width)
    y0 = drag_y + drag_h + 10
    gap = 6
    names = ["water", "methane", "ammonia", "hydrogen"]
    return {
        name: (x, y0 + i * (btn_h + gap), btn_w, btn_h)
        for i, name in enumerate(names)
    }


def clear_button_rect(width: int) -> Tuple[int, int, int, int]:
    btn_w, btn_h = 260, 26
    x = width - btn_w - 16
    preset_rects = preset_button_rects(width)
    _, last_y, _, last_h = preset_rects["hydrogen"]
    y = last_y + last_h + 8
    return (x, y, btn_w, btn_h)


def clouds_toggle_rect(height: int) -> Tuple[int, int, int, int]:
    btn_w, btn_h = 90, 26
    x = 16
    y = height - 120
    return (x, y, btn_w, btn_h)

def pairs_toggle_rect(height: int) -> Tuple[int, int, int, int]:
    btn_w, btn_h = 90, 26
    x = 16
    y = height - 100
    return (x, y, btn_w, btn_h)

def hybrid_toggle_rect(height: int) -> Tuple[int, int, int, int]:
    btn_w, btn_h = 90, 26
    x = 16
    y = height - 72
    return (x, y, btn_w, btn_h)

def angles_toggle_rect(height: int) -> Tuple[int, int, int, int]:
    """Toggle button for bond angle display"""
    btn_w, btn_h = 90, 26
    x = 16
    y = height - 48
    return (x, y, btn_w, btn_h)

def live_angles_toggle_rect(height: int) -> Tuple[int, int, int, int]:
    """Live 3D angle labels toggle"""
    btn_w, btn_h = 90, 26
    x = 16
    y = height - 32  # Lower position
    return (x, y, btn_w, btn_h)

def vsepr_clouds_button_rect(height: int) -> Tuple[int, int, int, int]:
    """clouds_toggle_rect stacked pattern - bottom-up from bottom of screen"""
    btn_w, btn_h = 90, 26
    x = 16
    y = height - 142
    return (x, y, btn_w, btn_h)

def vsepr_pairs_button_rect(height: int, clouds_y: int, clouds_h: int) -> Tuple[int, int, int, int]:
    y = clouds_y - clouds_h - 8
    btn_w, btn_h = 90, 26
    x = 16
    return (x, y, btn_w, btn_h)

def vsepr_hybrid_button_rect(height: int, pairs_y: int, pairs_h: int) -> Tuple[int, int, int, int]:
    y = pairs_y - pairs_h - 8
    btn_w, btn_h = 90, 26
    x = 16
    return (x, y, btn_w, btn_h)

def vsepr_angles_button_rect(height: int, hybrid_y: int, hybrid_h: int) -> Tuple[int, int, int, int]:
    y = hybrid_y - hybrid_h - 8
    btn_w, btn_h = 90, 26
    x = 16
    return (x, y, btn_w, btn_h)

def vsepr_live_angles_button_rect(height: int, angles_y: int, angles_h: int) -> Tuple[int, int, int, int]:
    y = angles_y - angles_h - 8
    btn_w, btn_h = 90, 26
    x = 16
    return (x, y, btn_w, btn_h)



def draw_atom(atom_model: ObjModel, atom: AtomState, scale_multiplier: float = 1.0) -> None:
    glPushMatrix()
    glTranslatef(atom.position[0], atom.position[1], atom.position[2])
    render_scale = atom.scale * scale_multiplier
    glScalef(render_scale, render_scale, render_scale)
    glColor3f(atom.tint[0], atom.tint[1], atom.tint[2])
    atom_model.draw()
    glPopMatrix()


def get_bond_mesh_profile(bond_model: ObjModel) -> BondMeshProfile:
    x_values = [v[0] for v in bond_model.vertices]
    if not x_values:
        return BondMeshProfile(-0.5, 0.5)
    return BondMeshProfile(min(x_values), max(x_values))


def draw_bond_between(
    profile: BondMeshProfile,
    a: AtomState,
    b: AtomState,
    thickness: float,
    bond_model_a: ObjModel,
    bond_model_b: ObjModel,
    offset: Vec3 = (0.0, 0.0, 0.0),
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
    glTranslatef(a.position[0] + offset[0], a.position[1] + offset[1], a.position[2] + offset[2])
    if axis_len > 0.000001:
        glRotatef(angle, axis_x / axis_len, axis_y / axis_len, axis_z / axis_len)
    elif ux < 0.0:
        glRotatef(180.0, 0.0, 1.0, 0.0)

    # Two-tone split bond: exact center join (no overlap).
    sa = a.scale
    sb = b.scale
    split = 0.5 * (sa + sb) / (sa + sb)
    segments = (
        (0.0, split, bond_model_a),
        (split, 1.0, bond_model_b),
    )
    for t0, t1, seg_model in segments:
        seg_start = length * t0
        seg_len = max(length * (t1 - t0), 0.000001)

        glPushMatrix()
        glTranslatef(seg_start, 0.0, 0.0)
        glScalef(seg_len / profile.span, thickness, thickness)
        glTranslatef(-profile.x_min, 0.0, 0.0)
        if t0 == 0.0:
            glColor3f(a.tint[0], a.tint[1], a.tint[2])
        else:
            glColor3f(b.tint[0], b.tint[1], b.tint[2])
        seg_model.draw()

        glPopMatrix()

    glColor3f(1.0, 1.0, 1.0)
    glPopMatrix()


def draw_overlay_rect(x: float, y: float, w: float, h: float, color: Vec3) -> None:
    glColor3f(color[0], color[1], color[2])
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def draw_overlay_rect_alpha(x: float, y: float, w: float, h: float, color: Vec3, alpha: float) -> None:
    clamped_alpha = max(0.0, min(1.0, float(alpha)))
    glColor4f(color[0], color[1], color[2], clamped_alpha)
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
    drag_mode: str,
    hud_requested_visible: bool,
    visualization_mode: str,
    show_clouds: bool,
    show_lone_pairs: bool,
    show_hybridization: bool,
    show_angles: bool,
    show_live_angles: bool,
    fps: int = 0,
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
    draw_overlay_rect(16, 24, 112, 360, (0.12, 0.12, 0.16))
    draw_overlay_text("PALETTE", 26, 34)
    for item in palette:
        size = item.radius * 2
        draw_overlay_rect(item.center[0] - item.radius, item.center[1] - item.radius, size, size, item.tint)
        draw_overlay_text(PALETTE_LABELS.get(item.name, item.name.upper()), 26, item.center[1] + 22)

    # VSEPR toggle buttons (lower-left)
    lx, ly, lw, lh = clouds_toggle_rect(height)
    clouds_color = (0.22, 0.32, 0.26) if show_clouds else (0.18, 0.18, 0.22)
    draw_overlay_rect(lx, ly, lw, lh, clouds_color)
    draw_overlay_text("ELECTRON CLOUDS", lx + 8, ly + 6)

    px, py, pw, ph = pairs_toggle_rect(height)
    pairs_color = (0.22, 0.32, 0.26) if show_lone_pairs else (0.18, 0.18, 0.22)
    draw_overlay_rect(px, py, pw, ph, pairs_color)
    draw_overlay_text("PAIRS", px + 12, py + 6)

    hx, hy, hw, hh = hybrid_toggle_rect(height)
    hybrid_color = (0.22, 0.32, 0.26) if show_hybridization else (0.18, 0.18, 0.22)
    draw_overlay_rect(hx, hy, hw, hh, hybrid_color)
    draw_overlay_text("HYBRID", hx + 8, hy + 6)

    ax, ay, aw, ah = angles_toggle_rect(height)
    angles_color = (0.22, 0.32, 0.26) if show_angles else (0.18, 0.18, 0.22)
    draw_overlay_rect(ax, ay, aw, ah, angles_color)
    draw_overlay_text("ANGLES", ax + 12, ay + 6)

    # LIVE ANGLES - moved lower for clickability
    lx, ly, lw, lh = live_angles_toggle_rect(height)
    live_color = (0.22, 0.32, 0.26) if show_live_angles else (0.18, 0.18, 0.22)
    draw_overlay_rect(lx, ly - 12, lw, lh, live_color)  # Lower Y by 12px
    draw_overlay_text("LIVE ANGLES", lx + 4, ly + 6 - 12)


    hx, hy, hw, hh = hud_toggle_button_rect(width)
    hud_color = (0.22, 0.32, 0.26) if hud_requested_visible else (0.18, 0.18, 0.22)
    draw_overlay_rect(hx, hy, hw, hh, hud_color)
    draw_overlay_text("HIDE 2D VIEW" if hud_requested_visible else "SHOW 2D VIEW", hx + 10, hy + 6)

    vx, vy, vw, vh = view_mode_button_rect(width)
    view_label = "SPACE-FILLING" if visualization_mode == "space-filling" else "BALL-AND-STICK"
    view_color = (0.24, 0.22, 0.30) if visualization_mode == "space-filling" else (0.20, 0.28, 0.22)
    draw_overlay_rect(vx, vy, vw, vh, view_color)
    draw_overlay_text(f"VIEW: {view_label}", vx + 10, vy + 6)

    bx, by, bw, bh = rearrange_button_rect(width)
    draw_overlay_rect(bx, by, bw, bh, (0.28, 0.24, 0.14))
    draw_overlay_text("RE-ARRANGE STRUCTURE", bx + 10, by + 7)

    dx, dy, dw, dh = drag_mode_button_rect(width)
    drag_active = drag_mode == "group"
    drag_color = (0.18, 0.30, 0.22) if drag_active else (0.22, 0.20, 0.16)
    draw_overlay_rect(dx, dy, dw, dh, drag_color)
    draw_overlay_text(f"DRAG MODE: {drag_mode.upper()}", dx + 10, dy + 6)

    for preset, (px, py, pw, ph) in preset_button_rects(width).items():
        draw_overlay_rect(px, py, pw, ph, (0.17, 0.22, 0.18))
        draw_overlay_text(f"SPAWN {preset.upper()}", px + 10, py + 6)

    cx, cy, cw, ch = clear_button_rect(width)
    draw_overlay_rect(cx, cy, cw, ch, (0.30, 0.12, 0.12))
    draw_overlay_text("CLEAR ALL", cx + 10, cy + 6)

    # Optional drag preview square.
    if dragging_preview is not None:
        px, py, _tint = dragging_preview
        # Keep drag preview neutral to avoid bright color flashing while dragging.
        draw_overlay_rect(px - 10, py - 10, 20, 20, (0.78, 0.78, 0.82))

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
        AtomState([oxygen_pos[0], oxygen_pos[1], oxygen_pos[2]], oxygen_scale, oxygen_scale, (1.0, 0.25, 0.25), "red"),
        AtomState([h1[0], h1[1], h1[2]], hydrogen_scale, hydrogen_scale, (0.9, 0.95, 1.0), "white"),
        AtomState([h2[0], h2[1], h2[2]], hydrogen_scale, hydrogen_scale, (0.9, 0.95, 1.0), "white"),
    ]

    ow_kind = canonical_bond_kind(atoms[0].kind, atoms[1].kind)
    bonds = [
        BondLink(atom_a=0, atom_b=1, thickness=bond_thickness, default_thickness=bond_thickness, tint=(0.8, 0.85, 0.95), kind=ow_kind),
        BondLink(atom_a=0, atom_b=2, thickness=bond_thickness, default_thickness=bond_thickness, tint=(0.8, 0.85, 0.95), kind=ow_kind),
    ]

    return atoms, bonds


def atom_kind_to_element(kind: str) -> str:
    value = str(kind).strip().lower()
    mapping = {
        "red": "O",
        "blue": "N",
        "white": "H",
        "black": "C",
        "gray": "C",
        "grey": "C",
        "carbon": "C",
        "nitrogen": "N",
        "oxygen": "O",
        "hydrogen": "H",
        "c": "C",
        "n": "N",
        "o": "O",
        "h": "H",
    }
    if value in mapping:
        return mapping[value]
    if value and value[0] in ("c", "n", "o", "h"):
        return value[0].upper()
    return "C"


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

    fov = math.radians(45.0)
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


def pick_atom_excluding(
    atoms: list[AtomState],
    mouse_pos: Tuple[int, int],
    width: int,
    height: int,
    rotate_x: float,
    rotate_y: float,
    zoom: float,
    camera_pan: Tuple[float, float],
    exclude_idx: int,
    max_pick_radius_px: float = 44.0,
) -> Optional[int]:
    mx, my = mouse_pos
    best_idx: Optional[int] = None
    best_dist = 1e9
    for idx, atom in enumerate(atoms):
        if idx == exclude_idx:
            continue
        screen = world_to_screen((atom.position[0], atom.position[1], atom.position[2]), width, height, rotate_x, rotate_y, zoom, camera_pan)
        if screen is None:
            continue
        dx = screen[0] - mx
        dy = screen[1] - my
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= max_pick_radius_px and dist < best_dist:
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
        offset = bond_visual_offset(atoms, bonds, idx)
        sa = world_to_screen(
            (a.position[0] + offset[0], a.position[1] + offset[1], a.position[2] + offset[2]),
            width,
            height,
            rotate_x,
            rotate_y,
            zoom,
            camera_pan,
        )
        sb = world_to_screen(
            (b.position[0] + offset[0], b.position[1] + offset[1], b.position[2] + offset[2]),
            width,
            height,
            rotate_x,
            rotate_y,
            zoom,
            camera_pan,
        )
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
        kept.append(BondLink(a, b, bond.thickness, bond.default_thickness, bond.tint, bond.kind))
    bonds[:] = kept


def pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def bond_count_between(bonds: list[BondLink], a: int, b: int) -> int:
    key = pair_key(a, b)
    return sum(1 for bond in bonds if pair_key(bond.atom_a, bond.atom_b) == key)


def connected_group_indices(atom_count: int, bonds: list[BondLink], seed: int) -> list[int]:
    if seed < 0 or seed >= atom_count:
        return []

    adjacency: list[list[int]] = [[] for _ in range(atom_count)]
    for bond in bonds:
        a = int(bond.atom_a)
        b = int(bond.atom_b)
        if 0 <= a < atom_count and 0 <= b < atom_count and a != b:
            adjacency[a].append(b)
            adjacency[b].append(a)

    seen = {seed}
    stack = [seed]
    component: list[int] = []
    while stack:
        cur = stack.pop()
        component.append(cur)
        for nxt in adjacency[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    component.sort()
    return component


def bond_slot_offsets(count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    if count == 2:
        return [-0.5, 0.5]
    return [-1.0, 0.0, 1.0]


def safe_normalize(v: Vec3) -> Vec3:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < 1e-8:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def cross_vec(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def bond_visual_offset(
    atoms: list[AtomState],
    bonds: list[BondLink],
    bond_idx: int,
) -> Vec3:
    bond = bonds[bond_idx]
    key = pair_key(bond.atom_a, bond.atom_b)
    same_pair_indices = [i for i, b in enumerate(bonds) if pair_key(b.atom_a, b.atom_b) == key]
    count = min(len(same_pair_indices), 3)
    if count <= 1:
        return (0.0, 0.0, 0.0)

    # Identify which parallel slot this bond occupies.
    same_pair_indices.sort()
    slot_index = min(same_pair_indices.index(bond_idx), count - 1)
    # Use canonical pair direction so offsets are stable even if bond endpoints
    # were created in mixed A->B / B->A order.
    a_idx, b_idx = key
    a = atoms[a_idx].position
    b = atoms[b_idx].position
    direction = safe_normalize((b[0] - a[0], b[1] - a[1], b[2] - a[2]))

    # Build a view-independent orthonormal basis around bond direction.
    reference = (0.0, 1.0, 0.0)
    if abs(direction[0] * reference[0] + direction[1] * reference[1] + direction[2] * reference[2]) > 0.95:
        reference = (1.0, 0.0, 0.0)

    basis_u = safe_normalize(cross_vec(direction, reference))
    if basis_u == (0.0, 0.0, 0.0):
        basis_u = (1.0, 0.0, 0.0)
    basis_v = safe_normalize(cross_vec(direction, basis_u))

    separation = 0.34

    if count == 2:
        slot = bond_slot_offsets(count)[slot_index]
        return (
            basis_u[0] * slot * separation,
            basis_u[1] * slot * separation,
            basis_u[2] * slot * separation,
        )

    if count >= 3:

        # In perspective mode, keep triangle arrangement around bond axis.
        triangle_points = [
            (0.0, 1.0),
            (-0.8660254, -0.5),
            (0.8660254, -0.5),
        ]
        px, py = triangle_points[min(slot_index, 2)]
        radius = separation
        return (
            basis_u[0] * px * radius + basis_v[0] * py * radius,
            basis_u[1] * px * radius + basis_v[1] * py * radius,
            basis_u[2] * px * radius + basis_v[2] * py * radius,
        )

    return (0.0, 0.0, 0.0)


def absolute_from_project(relative_path: str) -> str:
    import os.path
    here = os.path.abspath(__file__ if __file__ != '<string>' else 'main.py')
    project_dir = os.path.dirname(here)
    return os.path.join(project_dir, relative_path)


def load_runtime_settings() -> dict:
    return {
        "atom_model_path": swap_config.ATOM_MODEL_PATH,
        "bond_model_path": swap_config.BOND_MODEL_PATH,
        "material_library_root": getattr(swap_config, "MATERIAL_LIBRARY_ROOT", os.path.join("assets", "materials")),
        "use_atom_kind_overrides": bool(getattr(swap_config, "USE_ATOM_KIND_OVERRIDES", False)),
        "use_atom_mtl_overrides": bool(getattr(swap_config, "USE_ATOM_MTL_OVERRIDES", True)),
        "cpk_base_radius": float(getattr(swap_config, "CPK_BASE_RADIUS", 1.0)),
        "bond_thickness": float(swap_config.BOND_THICKNESS),
    }


def canonical_bond_kind(kind_a: str, kind_b: str) -> str:
    # Keep a stable domain-specific ordering so key names match bond model map.
    order = {"red": 0, "blue": 1, "white": 2}
    a_raw = kind_a.strip().lower()
    b_raw = kind_b.strip().lower()
    a, b = sorted([a_raw, b_raw], key=lambda k: order.get(k, 99))
    return f"{a}-{b}"


def load_optional_obj(path: str) -> Optional[ObjModel]:
    if not os.path.exists(path):
        return None
    try:
        model = load_obj(path)
        model.upload_textures()
        return model
    except Exception as exc:
        print(f"Optional custom OBJ ignored (using fallback): {path}")
        print(f"  Reason: {exc}")
        return None


def load_obj_with_mtl_override(obj_path: str, mtl_path: str) -> Optional[ObjModel]:
    """Load geometry from obj_path and force-apply the first material from mtl_path."""
    if not os.path.exists(obj_path) or not os.path.exists(mtl_path):
        return None

    try:
        model = load_obj(obj_path)
        materials = load_mtl(mtl_path)
        if not materials:
            return None

        forced_name = next(iter(materials.keys()))
        model.materials = materials
        model.triangles = [
            MeshTriangle(material_name=forced_name, vertices=tri.vertices)
            for tri in model.triangles
        ]
        model.upload_textures()
        return model
    except Exception as exc:
        print(f"Optional atom MTL override ignored: {mtl_path}")
        print(f"  Reason: {exc}")
        return None


def load_material_library(
    settings: dict,
    default_atom_model: ObjModel,
    default_bond_model: ObjModel,
) -> tuple[dict[str, ObjModel], dict[str, ObjModel], dict[str, BondMeshProfile]]:
    
    root = absolute_from_project(settings["material_library_root"])
    default_atom_obj_path = absolute_from_project(settings["atom_model_path"])

    atom_models: dict[str, ObjModel] = {
        "red": default_atom_model,
        "blue": default_atom_model,
        "white": default_atom_model,
        "black": default_atom_model,
    }

    # Optional: per-color atom geometry overrides (red.obj/blue.obj/white.obj).
    if settings.get("use_atom_kind_overrides", False):
        for atom_kind in ("red", "blue", "white", "black"):
            atom_path = os.path.join(root, "atoms", f"{atom_kind}.obj")
            atom_models[atom_kind] = load_optional_obj(atom_path) or default_atom_model
    elif settings.get("use_atom_mtl_overrides", True):
        # Reuse the single default atom OBJ geometry, but override MTL per atom kind.
        kind_mtl = {
            "red": os.path.join("assets", "atom", "oxygen.mtl"),
            "blue": os.path.join("assets", "atom", "nitrogen.mtl"),
            "white": os.path.join("assets", "atom", "hydrogen.mtl"),
            "black": os.path.join("assets", "atom", "carbon.mtl"),
        }
        for atom_kind in ("red", "blue", "white", "black"):
            atom_models[atom_kind] = (
                load_obj_with_mtl_override(default_atom_obj_path, kind_mtl[atom_kind])
                or default_atom_model
            )

    bond_models: dict[str, ObjModel] = {}
    bond_profiles: dict[str, BondMeshProfile] = {}
    bond_kinds = (
        "red-red",
        "red-blue",
        "red-white",
        "red-black",
        "blue-blue",
        "blue-white",
        "blue-black",
        "white-white",
        "white-black",
        "black-black",
    )

    # Use one shared custom bond model/texture for all bond kinds.
    shared_bond_model: Optional[ObjModel] = None
    for filename in ("bond.obj", "bond1.obj"):
        bond_path = os.path.join(root, "bonds", filename)
        shared_bond_model = load_optional_obj(bond_path)
        if shared_bond_model is not None:
            break

    for bond_kind in bond_kinds:
        bond_model = shared_bond_model or default_bond_model
        bond_models[bond_kind] = bond_model
        bond_profiles[bond_kind] = get_bond_mesh_profile(bond_model)

    return atom_models, bond_models, bond_profiles


def load_bond_kind_models_from_atom_mtls(settings: dict, default_bond_model: ObjModel) -> dict[str, ObjModel]:
    root = absolute_from_project(settings["material_library_root"])
    default_bond_obj_path = absolute_from_project(settings["bond_model_path"])

    kind_mtl = {
        "red": os.path.join(root, "oxygen.mtl"),
        "blue": os.path.join(root, "nitrogen.mtl"),
        "white": os.path.join(root, "hydrogen.mtl"),
        "black": os.path.join(root, "carbon.mtl"),
    }

    fallback_atom_mtl = kind_mtl  # Use same mappings

    models: dict[str, ObjModel] = {}
    for atom_kind in ("red", "blue", "white", "black"):
        model = load_obj_with_mtl_override(default_bond_obj_path, kind_mtl[atom_kind])
        if model is None:
            model = load_obj_with_mtl_override(default_bond_obj_path, fallback_atom_mtl[atom_kind])
        models[atom_kind] = model or default_bond_model

    return models


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
    base = settings["cpk_base_radius"]
    palette[0].scale = cpk_radius("O", base)
    palette[1].scale = cpk_radius("N", base)
    palette[2].scale = cpk_radius("H", base)
    if len(palette) > 3:
        palette[3].scale = cpk_radius("C", base)

    # Keep existing atoms in place, but update what "default" means.
    for atom in atoms:
        atom.default_scale = cpk_radius(atom_kind_to_element(atom.kind), base)

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
    from OpenGL.GL import glViewport
    glViewport(0, 0, width, height)
    setup_opengl(width, height)

    atom_model = load_obj(atom_obj)
    bond_model = load_obj(bond_obj)
    bond_profile = get_bond_mesh_profile(bond_model)
    atom_model.upload_textures()
    bond_model.upload_textures()
    atom_models, bond_models, bond_profiles = load_material_library(settings, atom_model, bond_model)
    bond_kind_models = load_bond_kind_models_from_atom_mtls(settings, bond_model)

    # Load HDRI skybox
    hdri_path = absolute_from_project(os.path.join("assets", "monochrome_studio_04_2k.exr"))
    skybox = HDRISkybox(hdri_path, sphere_subdivisions=8)

    print(f"Atom OBJ: {atom_obj}")
    print(f"Bond OBJ: {bond_obj}")
    print("Left-drag atom to move it. Bonds stretch automatically with distance.")
    print("Right-drag to orbit camera. Mouse wheel to zoom.")

    # Empty startup (no initial water molecule)
    atoms = []
    bonds = []

    engine = MolecularEngine(iterations=4, stiffness=0.55)
    engine.set_view_mode("space_filling")
    engine.set_toggle('clouds', False)
    engine.set_toggle('lone_pairs', False)
    engine.set_toggle('hybridization', False)

    hud_2d = MoleculeHUD2D()
    invalid_notif = BottomNotification()

# Toggle states (lower-left buttons)
    show_clouds = False

    show_lone_pairs = False
    show_hybridization = False
    show_angles = False
    show_live_angles = False


    def trigger_invalid_notif(message: str) -> None:
        invalid_notif.trigger(message, pygame.time.get_ticks())

    def element_visuals(element: str) -> Tuple[str, Vec3, float]:
        symbol = str(element).upper()
        if symbol == "O":
            return ("red", cpk_color("O"), cpk_radius("O", settings["cpk_base_radius"]))
        if symbol == "N":
            return ("blue", cpk_color("N"), cpk_radius("N", settings["cpk_base_radius"]))
        if symbol == "H":
            return ("white", cpk_color("H"), cpk_radius("H", settings["cpk_base_radius"]))
        return ("black", cpk_color("C"), cpk_radius("C", settings["cpk_base_radius"]))

    def apply_current_mode_spacing() -> None:
        if not atoms:
            return
        mode_key = "space_filling" if visualization_mode == "space-filling" else "ball_and_stick"
        engine.set_view_mode(mode_key)
        engine.set_structure(
            [atom_kind_to_element(atom.kind) for atom in atoms],
            [(int(b.atom_a), int(b.atom_b)) for b in bonds],
            [atom.position for atom in atoms],
        )
        stabilized = engine.rearrange_structure(mode=mode_key)
        sync_scene_from_engine()

    def spawn_preset(name: str) -> None:
        nonlocal topology_dirty
        focal = screen_to_world_camera_plane(
            (width // 2, height // 2),
            width,
            height,
            zoom,
            rotate_x,
            rotate_y,
            (camera_pan[0], camera_pan[1]),
        )
        elements, template_bonds, template_positions = MoleculePresets.spawn(name, focal_point=focal, scale=1.25)

        start_idx = len(atoms)
        for elem, pos in zip(elements, template_positions):
            kind, tint, scale = element_visuals(elem)
            atoms.append(AtomState([float(pos[0]), float(pos[1]), float(pos[2])], scale, scale, tint, kind))

        for a, b in template_bonds:
            ia = start_idx + int(a)
            ib = start_idx + int(b)
            bonds.append(
                BondLink(
                    ia,
                    ib,
                    settings["bond_thickness"],
                    settings["bond_thickness"],
                    (0.8, 0.85, 0.95),
                    canonical_bond_kind(atoms[ia].kind, atoms[ib].kind),
                )
            )

        topology_dirty = True
        apply_current_mode_spacing()

    def rebuild_engine_structure() -> None:
        elements = [atom_kind_to_element(atom.kind) for atom in atoms]
        bond_pairs = [(int(bond.atom_a), int(bond.atom_b)) for bond in bonds]
        engine.set_structure(elements, bond_pairs, [atom.position for atom in atoms])

    def sync_scene_from_engine() -> None:
        elements, bond_pairs, stabilized = engine.get_structure()

        old_atoms = atoms[:]
        atoms.clear()
        for i, (elem, pos) in enumerate(zip(elements, stabilized)):
            kind, tint, default_scale = element_visuals(elem)
            current_scale = default_scale
            if i < len(old_atoms):
                current_scale = float(old_atoms[i].scale)

            atoms.append(
                AtomState(
                    [float(pos[0]), float(pos[1]), float(pos[2])],
                    current_scale,
                    default_scale,
                    tint,
                    kind,
                )
            )

        bonds.clear()
        for a, b in bond_pairs:
            ia = int(a)
            ib = int(b)
            if ia < 0 or ib < 0 or ia >= len(atoms) or ib >= len(atoms) or ia == ib:
                continue
            bonds.append(
                BondLink(
                    ia,
                    ib,
                    settings["bond_thickness"],
                    settings["bond_thickness"],
                    (0.8, 0.85, 0.95),
                    canonical_bond_kind(atoms[ia].kind, atoms[ib].kind),
                )
            )

    def attempt_add_bond(atom_a: int, atom_b: int) -> bool:
        nonlocal topology_dirty
        ia = int(atom_a)
        ib = int(atom_b)
        if ia == ib or ia < 0 or ib < 0 or ia >= len(atoms) or ib >= len(atoms):
            return False

        # Keep engine and scene topology in sync before hard valency validation.
        # Manual connect must only create bond topology; no hydration/rearrange here.
        engine.set_structure(
            [atom_kind_to_element(atom.kind) for atom in atoms],
            [(int(b.atom_a), int(b.atom_b)) for b in bonds],
            [atom.position for atom in atoms],
        )

        if not engine.add_bond(ia, ib):
            return False

        # Immediate visual dual-color bond, while atom positions remain unchanged.
        bonds.append(
            BondLink(
                ia,
                ib,
                settings["bond_thickness"],
                settings["bond_thickness"],
                (0.8, 0.85, 0.95),
                canonical_bond_kind(atoms[ia].kind, atoms[ib].kind),
            )
        )

        # Rebuild-only flag; do not call apply_current_mode_spacing/rearrange here.
        topology_dirty = True
        return True

    rebuild_engine_structure()
    topology_dirty = False

    palette = [
        PaletteItem("red", cpk_color("O"), cpk_radius("O", settings["cpk_base_radius"]), (72, 84), 18),
        PaletteItem("blue", cpk_color("N"), cpk_radius("N", settings["cpk_base_radius"]), (72, 144), 18),
        PaletteItem("white", cpk_color("H"), cpk_radius("H", settings["cpk_base_radius"]), (72, 204), 18),
        PaletteItem("black", cpk_color("C"), cpk_radius("C", settings["cpk_base_radius"]), (72, 264), 18),
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
    dragging_group_indices: list[int] = []
    drag_mode = "precision"
    hud_requested_visible = True
    visualization_mode = "space-filling"
    dragging_new_atom: Optional[PaletteItem] = None
    drag_start_positions: dict[int, list[float]] = {}
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
                    hx, hy, hw, hh = hud_toggle_button_rect(width)
                    if hx <= event.pos[0] <= hx + hw and hy <= event.pos[1] <= hy + hh:
                        hud_requested_visible = not hud_requested_visible
                        continue

                    vx, vy, vw, vh = view_mode_button_rect(width)
                    if vx <= event.pos[0] <= vx + vw and vy <= event.pos[1] <= vy + vh:
                        old_mode = visualization_mode
                        visualization_mode = "ball-stick" if visualization_mode == "space-filling" else "space-filling"
                        mode_key = "space_filling" if visualization_mode == "space-filling" else "ball_and_stick"
                        engine.set_view_mode(mode_key)
                        apply_current_mode_spacing()
                        continue

                    dx, dy, dw, dh = drag_mode_button_rect(width)
                    if dx <= event.pos[0] <= dx + dw and dy <= event.pos[1] <= dy + dh:
                        drag_mode = "group" if drag_mode == "precision" else "precision"
                        continue

                    cx, cy, cw, ch = clear_button_rect(width)
                    if cx <= event.pos[0] <= cx + cw and cy <= event.pos[1] <= cy + ch:
                        atoms.clear()
                        bonds.clear()
                        connect_source_idx = None
                        context_menu = None
                        dragging_atom_idx = None
                        dragging_group_indices = []
                        dragging_new_atom = None
                        overlay_drag_preview = None
                        topology_dirty = True
                        continue

                    bx, by, bw, bh = rearrange_button_rect(width)
                    if bx <= event.pos[0] <= bx + bw and by <= event.pos[1] <= by + bh:
                        mode_key = "space_filling" if visualization_mode == "space-filling" else "ball_and_stick"
                        engine.set_view_mode(mode_key)
                        engine.update_raw_positions([atom.position for atom in atoms])
                        engine.rearrange_structure(mode=mode_key)
                        sync_scene_from_engine()
                        hud_2d.apply_vsepr_layout()
                        continue

    # Toggle buttons (lower-left) - using new stacked rects
                    # LIVE ANGLES
                    live_rect = live_angles_toggle_rect(height)
                    if live_rect[0] <= event.pos[0] <= live_rect[0] + live_rect[2] and live_rect[1] <= event.pos[1] <= live_rect[1] + live_rect[3]:
                        show_live_angles = not show_live_angles
                        continue

                    # ANGLES
                    angles_rect = angles_toggle_rect(height)
                    if angles_rect[0] <= event.pos[0] <= angles_rect[0] + angles_rect[2] and angles_rect[1] <= event.pos[1] <= angles_rect[1] + angles_rect[3]:
                        show_angles = not show_angles
                        continue

                    # HYBRID = LIVE ANGLES
                    hybrid_rect = hybrid_toggle_rect(height)
                    if hybrid_rect[0] <= event.pos[0] <= hybrid_rect[0] + hybrid_rect[2] and hybrid_rect[1] <= event.pos[1] <= hybrid_rect[1] + hybrid_rect[3]:
                        show_live_angles = not show_live_angles
                        print("Live Angles toggled:", show_live_angles)
                        continue

                    # PAIRS
                    pairs_rect = pairs_toggle_rect(height)
                    if pairs_rect[0] <= event.pos[0] <= pairs_rect[0] + pairs_rect[2] and pairs_rect[1] <= event.pos[1] <= pairs_rect[1] + pairs_rect[3]:
                        engine.set_toggle('lone_pairs', not show_lone_pairs)
                        continue

                    # CLOUDS
                    clouds_rect = clouds_toggle_rect(height)
                    if clouds_rect[0] <= event.pos[0] <= clouds_rect[0] + clouds_rect[2] and clouds_rect[1] <= event.pos[1] <= clouds_rect[1] + clouds_rect[3]:
                        engine.set_toggle('clouds', not show_clouds)
                        continue

                    clicked_preset = False
                    for preset_name, (px, py, pw, ph) in preset_button_rects(width).items():
                        if px <= event.pos[0] <= px + pw and py <= event.pos[1] <= py + ph:
                            spawn_preset(preset_name)
                            clicked_preset = True
                            break
                    if clicked_preset:
                        continue


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
                                        topology_dirty = True
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
                                        topology_dirty = True
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
                        if target_idx is not None and target_idx != connect_source_idx:
                            if not attempt_add_bond(connect_source_idx, target_idx):
                                trigger_invalid_notif("Invalid Bond: Atom Valency Exceeded")
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
                    if dragging_atom_idx is not None:
                        if drag_mode == "group":
                            dragging_group_indices = connected_group_indices(len(atoms), bonds, dragging_atom_idx)
                        else:
                            dragging_group_indices = [dragging_atom_idx]
                        drag_start_positions = {idx: atoms[idx].position.copy() for idx in dragging_group_indices}
                    else:
                        dragging_group_indices = []
                        drag_start_positions = {}
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

                    bond_idx = pick_bond(
                        atoms,
                        bonds,
                        event.pos,
                        width,
                        height,
                        rotate_x,
                        rotate_y,
                        zoom,
                        (camera_pan[0], camera_pan[1]),
                    )
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
                released_drag_idx = dragging_atom_idx
                released_group_indices = dragging_group_indices.copy()
                dragging_atom_idx = None
                dragging_group_indices = []
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
                            dragging_new_atom.name,
                        )
                    )
                    topology_dirty = True
                    dragging_new_atom = None
                    overlay_drag_preview = None

                if released_drag_idx is not None and len(atoms) > 1:
                    target_idx = pick_atom_excluding(
                        atoms,
                        event.pos,
                        width,
                        height,
                        rotate_x,
                        rotate_y,
                        zoom,
                        (camera_pan[0], camera_pan[1]),
                        exclude_idx=released_drag_idx,
                    )
                    if target_idx is not None:
                        if not attempt_add_bond(released_drag_idx, target_idx):
                            for idx in released_group_indices:
                                if idx in drag_start_positions:
                                    atoms[idx].position[0] = float(drag_start_positions[idx][0])
                                    atoms[idx].position[1] = float(drag_start_positions[idx][1])
                                    atoms[idx].position[2] = float(drag_start_positions[idx][2])
                                    engine.update_raw_pos(idx, atoms[idx].position)
                            trigger_invalid_notif("Invalid Bond: Atom Valency Exceeded")
                drag_start_positions = {}

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                rotating_camera = False
                panning_camera = False

            elif event.type == pygame.MOUSEMOTION and dragging_atom_idx is not None:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]

                move_scale = world_units_per_pixel(zoom, height) * 0.85  # Drag sensitivity adjustment
                d_world = camera_to_world_vector((dx * move_scale, -dy * move_scale, 0.0), rotate_x, rotate_y)

                move_indices = dragging_group_indices if dragging_group_indices else [dragging_atom_idx]
                for idx in move_indices:
                    atoms[idx].position[0] += d_world[0]
                    atoms[idx].position[1] += d_world[1]
                    atoms[idx].position[2] += d_world[2]
                    engine.update_raw_pos(idx, atoms[idx].position)

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
                elif event.key == pygame.K_g:
                    drag_mode = "group" if drag_mode == "precision" else "precision"
                elif event.key == pygame.K_h:
                    hud_requested_visible = not hud_requested_visible
                elif event.key == pygame.K_v:
                    visualization_mode = "ball-stick" if visualization_mode == "space-filling" else "space-filling"
                    mode_key = "space_filling" if visualization_mode == "space-filling" else "ball_and_stick"
                    engine.set_view_mode(mode_key)
                    apply_current_mode_spacing()
                elif event.key == pygame.K_c and context_menu is not None and context_menu["type"] == "atom":
                    connect_source_idx = context_menu["target"]
                    context_menu = None
                    pygame.display.set_caption("Connect mode: left-click another atom to create bond")
                elif event.key == pygame.K_DELETE and context_menu is not None:
                    if context_menu["type"] == "atom":
                        atom_idx = context_menu["target"]
                        if 0 <= atom_idx < len(atoms):
                            remove_atom(atoms, bonds, atom_idx)
                            topology_dirty = True
                            connect_source_idx = None
                    elif context_menu["type"] == "bond":
                        bond_idx = context_menu["target"]
                        if 0 <= bond_idx < len(bonds):
                            del bonds[bond_idx]
                            topology_dirty = True
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

# Sync toggle states from engine every frame
        toggles = engine.get_toggles()
        show_clouds = toggles['show_clouds']

        show_lone_pairs = toggles['show_lone_pairs']
        show_hybridization = toggles['show_hybridization']
        show_angles = False
        show_live_angles = show_live_angles  # Preserve manual toggle state

        if topology_dirty:
            rebuild_engine_structure()
            topology_dirty = False

        dragging_active = any(
            [
                dragging_atom_idx is not None,
                dragging_new_atom is not None,
                rotating_camera,
                panning_camera,
            ]
        )
        effective_hud_visible = hud_requested_visible and not dragging_active
        hud_2d.set_visible(effective_hud_visible)

        hud_2d.sync_structure(
            [atom_kind_to_element(atom.kind) for atom in atoms],
            [(int(b.atom_a), int(b.atom_b)) for b in bonds],
            [atom.position for atom in atoms],
        )

        stabilized = engine.get_stabilized_positions()
        for atom, pos in zip(atoms, stabilized):
            atom.position[0] = float(pos[0])
            atom.position[1] = float(pos[1])
            atom.position[2] = float(pos[2])

        set_projection(width, height, zoom)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Render HDRI skybox (rotates with camera, unaffected by pan/zoom)
        skybox.render(rotate_x, rotate_y, scale=50.0)

        # Apply camera transforms for molecules (pan, zoom, rotate)
        glTranslatef(camera_pan[0], camera_pan[1], 0.0)
        glTranslatef(0.0, 0.0, zoom)
        glRotatef(rotate_x, 1.0, 0.0, 0.0)
        glRotatef(rotate_y, 0.0, 1.0, 0.0)

        # Keep key light in stable world-space (not camera-locked).
        glLightfv(GL_LIGHT0, GL_POSITION, (6.0, 8.0, 10.0, 1.0))

        for bond_idx, bond in enumerate(bonds):
            visual_offset = bond_visual_offset(atoms, bonds, bond_idx)
            active_bond_model = bond_models.get(bond.kind, bond_model)
            active_profile = bond_profiles.get(bond.kind, bond_profile)
            bond_model_a = bond_kind_models.get(atoms[bond.atom_a].kind, active_bond_model)
            bond_model_b = bond_kind_models.get(atoms[bond.atom_b].kind, active_bond_model)
            draw_bond_between(
                active_profile,
                atoms[bond.atom_a],
                atoms[bond.atom_b],
                bond.thickness,
                bond_model_a,
                bond_model_b,
                visual_offset,
            )

        # Render atoms
        for atom in atoms:
            active_atom_model = atom_models.get(atom.kind, atom_model)
            draw_atom(
                active_atom_model,
                atom,
                scale_multiplier=1.0 if visualization_mode == "space-filling" else 0.4,
            )

# Render Electron Clouds (transparent) AFTER opaque atoms/bonds
        from OpenGL.GL import glDepthMask
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(False)  # Allow transparent writes
        
        if show_clouds or show_lone_pairs:
            for atom_idx in range(len(atoms)):
                if show_clouds:
                    # Proportional electron cloud: radius * 1.6, light blue, proper depth state
                    cloud_radius = atoms[atom_idx].scale * 1.6
                    glColor4f(0.6, 0.7, 1.0, 0.08)
                    glPushMatrix()
                    glTranslatef(atoms[atom_idx].position[0], atoms[atom_idx].position[1], atoms[atom_idx].position[2])
                    glScalef(cloud_radius, cloud_radius, cloud_radius)
                    glDisable(GL_LIGHTING)
                    atom_model.draw()
                    glEnable(GL_LIGHTING)
                    glPopMatrix()
                
                if show_lone_pairs:
                    # Lone pair spheres (higher alpha)
                    ghost_count = engine._ghost_count(atom_idx, len(engine._neighbors[atom_idx]))
                    if ghost_count > 0:
                        atom_center = np.array(atoms[atom_idx].position)
                        atom_scale = atoms[atom_idx].scale * 0.6
                        # Use engine's tetrahedral ghost directions
                        bond_dirs = []
                        for nb_idx in engine._neighbors[atom_idx]:
                            if nb_idx < len(atoms):
                                delta = np.array(atoms[nb_idx].position) - atom_center
                                length = np.linalg.norm(delta)
                                if length > 1e-8:
                                    bond_dirs.append(delta / length)
                        
                        ghost_dirs = engine._init_ghost_directions(np.array(bond_dirs), ghost_count)
                        
                        for ghost_dir in ghost_dirs:
                            ghost_pos = atom_center + ghost_dir * atom_scale * 1.4
                            glPushMatrix()
                            glTranslatef(float(ghost_pos[0]), float(ghost_pos[1]), float(ghost_pos[2]))
                            glScalef(atom_scale*0.7, atom_scale*0.7, atom_scale*0.7)
                            glColor4f(0.65, 0.35, 0.85, 0.4)
                            glDisable(GL_LIGHTING)
                            atom_model.draw()
                            glEnable(GL_LIGHTING)
                            glPopMatrix()
        
        glDepthMask(True)  # Restore opaque depth writes
        glDisable(GL_BLEND)

        hud_2d.draw(width, height)
        draw_2d_overlay(
            width,
            height,
            palette,
            context_menu,
            overlay_drag_preview,
            drag_mode,
            hud_requested_visible,
            visualization_mode,
            show_clouds,
            show_lone_pairs,
            show_hybridization,
            show_angles,
            show_live_angles,
            0  # fps
        )

        # Render Live Bond Angles when toggled ON (top of 3D viewport)
        if show_live_angles:
            try:
                angle_labels = get_visible_angles(atoms, bonds)
                
                # Setup 2D ortho projection exactly per feedback spec
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
                
                # Bright yellow, no lighting
                glColor3f(1.0, 1.0, 0.0)
                
                for angle_text, (sx, sy) in angle_labels:
                    draw_overlay_text(angle_text, float(sx), float(sy), pixel=2)
                
                # Cleanup: restore 3D matrices
                glPopMatrix()
                glMatrixMode(GL_PROJECTION)
                glPopMatrix()
                glMatrixMode(GL_MODELVIEW)
                
                glEnable(GL_LIGHTING)
                glEnable(GL_DEPTH_TEST)
                
            except Exception as e:
                print(f"Live angles render error: {e}")
        
        invalid_notif.draw(width, height, pygame.time.get_ticks())

        pygame.display.flip()


    # Cleanup
    skybox.cleanup()
    pygame.quit()


if __name__ == "__main__":
    main()
