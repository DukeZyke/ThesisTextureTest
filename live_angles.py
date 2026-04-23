import numpy as np
import OpenGL.GL as gl
from OpenGL.GLU import gluProject
from typing import List, Tuple

def get_visible_angles(atoms: List, bonds: List) -> List[Tuple[str, Tuple[float, float]]]:
    if not atoms or not bonds:
        return []
    
    # 1. Group neighbors to ignore double bonds
    temp_adj = [set() for _ in range(len(atoms))]
    for bond in bonds:
        a, b = int(bond.atom_a), int(bond.atom_b)
        if 0 <= a < len(atoms) and 0 <= b < len(atoms):
            temp_adj[a].add(b)
            temp_adj[b].add(a)
    
    adjacency = [list(s) for s in temp_adj]
    
    modelview = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
    projection = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
    viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
    height = viewport[3]
    labels = []

    for center_idx, neighbors in enumerate(adjacency):
        if len(neighbors) < 2: continue
        c_pos = np.array(atoms[center_idx].position, dtype=np.float64)

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                p_a = np.array(atoms[neighbors[i]].position, dtype=np.float64)
                p_b = np.array(atoms[neighbors[j]].position, dtype=np.float64)
                
                v_a, v_b = p_a - c_pos, p_b - c_pos
                l_a, l_b = np.linalg.norm(v_a), np.linalg.norm(v_b)
                if l_a < 0.1 or l_b < 0.1: continue

                dir_a, dir_b = v_a/l_a, v_b/l_b
                dot = np.clip(np.dot(dir_a, dir_b), -1.0, 1.0)
                angle_rad = np.arccos(dot)

                # --- DRAW THE 3D ARCH ---
                gl.glDisable(gl.GL_LIGHTING)
                gl.glLineWidth(2.0)
                gl.glColor4f(0.0, 1.0, 0.4, 0.6)
                gl.glBegin(gl.GL_LINE_STRIP)
                steps, arc_radius = 20, min(l_a, l_b) * 0.6 
                if angle_rad > 0.001:
                    for t in np.linspace(0, 1, steps):
                        interp_v = (np.sin((1-t)*angle_rad)*dir_a + np.sin(t*angle_rad)*dir_b) / np.sin(angle_rad)
                        gl.glVertex3dv(c_pos + interp_v * arc_radius)
                gl.glEnd()
                gl.glEnable(gl.GL_LIGHTING)

                # --- 2D LABEL ---
                combined_v = dir_a + dir_b
                c_len = np.linalg.norm(combined_v)
                label_dir = combined_v / c_len if c_len > 1e-8 else np.array([0,1,0])
                label_world = c_pos + label_dir * (arc_radius + 0.3)
                
                try:
                    wx, wy, wz = gluProject(label_world[0], label_world[1], label_world[2], modelview, projection, viewport)
                    if 0 < wz < 1:
                        # Standard symbols now supported by your BITMAP_FONT
                        label_text = f"{np.degrees(angle_rad):.1f}°"
                        labels.append((label_text, (wx, height - wy)))
                except: continue
                    
    return labels