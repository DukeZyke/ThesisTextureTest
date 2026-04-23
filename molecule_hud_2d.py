

import math
from collections import Counter, deque
from typing import Sequence

import numpy as np
from OpenGL.GL import (
    GL_DEPTH_TEST,
    GL_LIGHTING,
    GL_LINES,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_TEXTURE_2D,
    GL_TRIANGLE_FAN,
    glBegin,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glPopMatrix,
    glPushMatrix,
    glVertex2f,
)

from molecular_engine import cpk_color, cpk_radius, MolecularEngine


class MoleculeHUD2D:
    """Lower-right 2D molecule HUD with planar, readable layout."""

    def __init__(self, panel_size: tuple[int, int] = (320, 280), margin: int = 12) -> None:
        self.panel_w, self.panel_h = panel_size
        self.margin = margin
        self.visible = True

        self.elements: list[str] = []
        self.bonds: list[tuple[int, int]] = []
        self._bond_counts: Counter[tuple[int, int]] = Counter()
        self._positions_3d = np.empty((0, 3), dtype=np.float64)
        self._layout = np.empty((0, 2), dtype=np.float64)
        self._topology_signature: tuple[tuple[str, ...], tuple[tuple[int, int], ...]] | None = None
        self.engine = MolecularEngine()

    def toggle(self) -> None:
        self.visible = not self.visible

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)

    def sync_structure(self, elements: Sequence[str], bonds: Sequence[tuple[int, int]], positions_3d: Sequence[Sequence[float]]) -> None:
        norm_elements = [str(e).upper() for e in elements]
        norm_bonds = []
        for a, b in bonds:
            ia = int(a)
            ib = int(b)
            if ia == ib:
                continue
            if ia < 0 or ib < 0 or ia >= len(norm_elements) or ib >= len(norm_elements):
                continue
            norm_bonds.append((min(ia, ib), max(ia, ib)))
        norm_bonds.sort()

        raw_pos = np.asarray(positions_3d, dtype=np.float64)
        if raw_pos.size == 0:
            raw_pos = np.empty((0, 3), dtype=np.float64)
        if raw_pos.ndim != 2 or raw_pos.shape[1] != 3 or raw_pos.shape[0] != len(norm_elements):
            raw_pos = np.zeros((len(norm_elements), 3), dtype=np.float64)

        signature = (tuple(norm_elements), tuple(norm_bonds))
        topology_changed = signature != self._topology_signature
        self._topology_signature = signature

        self.elements = norm_elements
        self.bonds = norm_bonds
        self._bond_counts = Counter(norm_bonds)
        self._positions_3d = raw_pos

        # Sync engine for hybridization
        self.engine.set_structure(self.elements, self.bonds, self._positions_3d.tolist())

        if topology_changed or self._layout.shape[0] != len(self.elements):
            # Reset orientation anchor when atom/bond topology changes.
            self._layout = np.empty((0, 2), dtype=np.float64)
            self._layout = self._project_topdown_from_3d(self._positions_3d)

    def apply_vsepr_layout(self) -> None:
        self._layout = self._build_skeletal_layout()

    def draw(self, screen_width: int, screen_height: int) -> None:
        if not self.visible:
            return

        x0 = screen_width - self.margin - self.panel_w
        y0 = screen_height - self.margin - self.panel_h

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, screen_width, screen_height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        self._draw_rect(x0, y0, self.panel_w, self.panel_h, (0.08, 0.09, 0.12))
        self._draw_rect(x0 + 2, y0 + 2, self.panel_w - 4, 22, (0.14, 0.18, 0.23))

        if self._layout.shape[0] > 0:
            projected = self._project_layout_into_panel(self._layout, x0, y0)

            for (a, b), count in sorted(self._bond_counts.items()):
                self._draw_multi_bond(projected[a], projected[b], count)

            for i, element in enumerate(self.elements):
                px, py = projected[i]
                radius_px = 7.0 * cpk_radius(element, base_radius=1.0)
                self._draw_circle(float(px), float(py), float(radius_px), cpk_color(element))
                
                # Hybridization label (step 8)
                hybrid = self.engine.hybridization(i)
                if hybrid != 'unknown':
                    label_x = float(px) + 12
                    label_y = float(py) - 8
                    glColor3f(0.95, 0.95, 0.95)
                    self._draw_text_overlay(hybrid, label_x, label_y)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

    def _draw_rect(self, x: float, y: float, w: float, h: float, color: tuple[float, float, float]) -> None:
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def _draw_circle(self, cx: float, cy: float, r: float, color: tuple[float, float, float]) -> None:
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        segments = 28
        for i in range(segments + 1):
            t = (2.0 * math.pi * i) / segments
            glVertex2f(cx + math.cos(t) * r, cy + math.sin(t) * r)
        glEnd()

    def _draw_multi_bond(self, a: np.ndarray, b: np.ndarray, count: int) -> None:
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return

        tangent = np.array([-direction[1], direction[0]], dtype=np.float64) / length
        offsets = self._bond_line_offsets(count)
        for offset in offsets:
            shift = tangent * offset
            glColor3f(0.78, 0.80, 0.84)
            glBegin(GL_LINES)
            glVertex2f(float(a[0] + shift[0]), float(a[1] + shift[1]))
            glVertex2f(float(b[0] + shift[0]), float(b[1] + shift[1]))
            glEnd()

    def _bond_line_offsets(self, count: int) -> list[float]:
        if count <= 1:
            return [0.0]
        if count == 2:
            return [-3.0, 3.0]
        if count == 3:
            return [-4.5, 0.0, 4.5]
        return [float(i - (count - 1) * 0.5) * 3.0 for i in range(count)]

    def _draw_text_overlay(self, text: str, x: float, y: float) -> None:
        pixel_size = 2.5
        cursor_x = x
        color = (0.95, 0.95, 0.95)
        for ch in text.upper():
            if ch == 'S':
                # sp
                self._draw_rect(cursor_x, y, 3 * pixel_size, pixel_size, color)
                self._draw_rect(cursor_x + pixel_size, y + pixel_size, pixel_size, pixel_size, color)
                cursor_x += 6 * pixel_size
            elif ch == 'P':
                # sp2/sp3
                self._draw_rect(cursor_x, y, pixel_size, 2 * pixel_size, color)
                cursor_x += 3 * pixel_size
            elif ch == '2':
                self._draw_rect(cursor_x, y, 2 * pixel_size, pixel_size, color)
                self._draw_rect(cursor_x, y + pixel_size, pixel_size, pixel_size, color)
                cursor_x += 4 * pixel_size
            elif ch == '3':
                self._draw_rect(cursor_x, y, pixel_size, 2 * pixel_size, color)
                self._draw_rect(cursor_x + pixel_size, y + pixel_size, pixel_size, pixel_size, color)
                cursor_x += 4 * pixel_size
            else:
                cursor_x += 4 * pixel_size

    def _project_layout_into_panel(self, layout: np.ndarray, x0: int, y0: int) -> np.ndarray:
        pad = 18.0
        min_xy = np.min(layout, axis=0)
        max_xy = np.max(layout, axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)

        usable_w = self.panel_w - 2.0 * pad
        usable_h = self.panel_h - 2.0 * pad - 18.0
        scale = min(usable_w / span[0], usable_h / span[1])

        centered = layout - (min_xy + max_xy) * 0.5
        projected = centered * scale
        projected[:, 0] += x0 + self.panel_w * 0.5
        projected[:, 1] += y0 + self.panel_h * 0.48
        return projected

    def _project_topdown_from_3d(self, positions: np.ndarray) -> np.ndarray:
        # Keep method name for compatibility with existing call sites. HUD layout
        # is intentionally 2D skeletal and does not depend on camera/3D rotation.
        _ = positions
        return self._build_skeletal_layout()

    def _build_skeletal_layout(self) -> np.ndarray:
        n = len(self.elements)
        if n == 0:
            return np.empty((0, 2), dtype=np.float64)
        if n == 1:
            return np.zeros((1, 2), dtype=np.float64)

        adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
        for a, b in self.bonds:
            adjacency[a].add(b)
            adjacency[b].add(a)

        heavy = [i for i, e in enumerate(self.elements) if e != "H"]
        if not heavy:
            heavy = list(range(n))

        backbone = self._select_backbone_path(heavy, adjacency)
        positions = np.full((n, 2), np.nan, dtype=np.float64)
        direction_map: dict[tuple[int, int], np.ndarray] = {}

        bond_step = 2.1
        for i, atom_idx in enumerate(backbone):
            positions[atom_idx] = np.array([i * bond_step, 0.0], dtype=np.float64)

        for i in range(len(backbone) - 1):
            a = backbone[i]
            b = backbone[i + 1]
            direction_map[(a, b)] = np.array([1.0, 0.0], dtype=np.float64)
            direction_map[(b, a)] = np.array([-1.0, 0.0], dtype=np.float64)

        queue: deque[int] = deque(backbone)
        visited = set(backbone)
        while queue:
            parent = queue.popleft()
            parent_pos = positions[parent]
            parent_dir = None
            for candidate in adjacency[parent]:
                if (candidate, parent) in direction_map:
                    parent_dir = direction_map[(candidate, parent)]
                    break

            for child in sorted(adjacency[parent]):
                if child in visited:
                    continue
                chosen = self._choose_skeletal_direction(parent, child, positions, adjacency, parent_dir)
                child_pos = parent_pos + chosen * bond_step
                positions[child] = child_pos
                direction_map[(parent, child)] = chosen
                direction_map[(child, parent)] = -chosen
                visited.add(child)
                queue.append(child)

        for idx in range(n):
            if not np.isfinite(positions[idx, 0]):
                positions[idx] = np.array([0.0, 0.0], dtype=np.float64)

        self._fan_hydrogens_on_nitrogen(positions, adjacency)
        self._repel_overlaps(positions, adjacency, anchor=set(backbone))

        positions -= np.mean(positions, axis=0, keepdims=True)
        return positions

    def _select_backbone_path(self, heavy: Sequence[int], adjacency: dict[int, set[int]]) -> list[int]:
        if not heavy:
            return []

        heavy_set = set(heavy)

        def bfs_farthest(start: int) -> tuple[int, dict[int, int]]:
            q = deque([start])
            parent = {start: start}
            order = [start]
            while q:
                cur = q.popleft()
                for nxt in sorted(adjacency[cur]):
                    if nxt not in heavy_set or nxt in parent:
                        continue
                    parent[nxt] = cur
                    q.append(nxt)
                    order.append(nxt)
            return order[-1], parent

        start = max(heavy, key=lambda i: (len(adjacency[i]), -i))
        far_a, _ = bfs_farthest(start)
        far_b, parent = bfs_farthest(far_a)

        path = [far_b]
        while path[-1] != far_a:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def _choose_skeletal_direction(
        self,
        parent: int,
        child: int,
        positions: np.ndarray,
        adjacency: dict[int, set[int]],
        parent_dir: np.ndarray | None,
    ) -> np.ndarray:
        del child
        base_angles = np.deg2rad([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
        candidates = [np.array([math.cos(a), math.sin(a)], dtype=np.float64) for a in base_angles]

        used_dirs: list[np.ndarray] = []
        parent_pos = positions[parent]
        for nb in adjacency[parent]:
            if nb == parent or not np.isfinite(positions[nb, 0]):
                continue
            d = positions[nb] - parent_pos
            length = float(np.linalg.norm(d))
            if length > 1e-6:
                used_dirs.append(d / length)

        best = candidates[0]
        best_score = -1e9
        for cand in candidates:
            score = 0.0
            for used in used_dirs:
                # Prefer 120-degree separation for branched/skeletal look.
                dot = float(np.clip(np.dot(cand, used), -1.0, 1.0))
                score += -abs(dot + 0.5)
            if parent_dir is not None:
                dot_parent = float(np.clip(np.dot(cand, parent_dir), -1.0, 1.0))
                score += -0.5 * abs(dot_parent + 0.5)

            test_pos = parent_pos + cand * 1.75
            if np.any(np.isfinite(positions[:, 0])):
                dists = np.linalg.norm(positions[np.isfinite(positions[:, 0])] - test_pos, axis=1)
                if dists.size > 0:
                    score += float(np.min(dists))

            if score > best_score:
                best_score = score
                best = cand
        return best

    def _fan_hydrogens_on_nitrogen(self, positions: np.ndarray, adjacency: dict[int, set[int]]) -> None:
        for center, element in enumerate(self.elements):
            if element != "N":
                continue

            h_neighbors = [i for i in sorted(adjacency[center]) if self.elements[i] == "H"]
            if len(h_neighbors) <= 1:
                continue

            center_pos = positions[center]
            heavy_neighbors = [i for i in adjacency[center] if self.elements[i] != "H"]
            anchor_angle = 0.0
            if heavy_neighbors:
                ref = positions[heavy_neighbors[0]] - center_pos
                anchor_angle = math.atan2(ref[1], ref[0]) + math.pi

            spread = np.deg2rad(120.0)
            start = anchor_angle - spread * 0.5
            step = spread / max(len(h_neighbors) - 1, 1)
            for i, h_idx in enumerate(h_neighbors):
                angle = start + i * step
                positions[h_idx] = center_pos + np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * 1.55

    def _repel_overlaps(self, positions: np.ndarray, adjacency: dict[int, set[int]], anchor: set[int]) -> None:
        n = positions.shape[0]
        for _ in range(22):
            forces = np.zeros_like(positions)
            for i in range(n):
                for j in range(i + 1, n):
                    if j in adjacency[i]:
                        continue
                    delta = positions[j] - positions[i]
                    dist = float(np.linalg.norm(delta))
                    if dist < 1e-6:
                        delta = np.array([0.01, 0.0], dtype=np.float64)
                        dist = 0.01

                    min_dist = 0.65 * (cpk_radius(self.elements[i], 1.0) + cpk_radius(self.elements[j], 1.0))
                    if dist >= min_dist:
                        continue
                    push = (min_dist - dist) * (delta / dist)
                    forces[i] -= 0.12 * push
                    forces[j] += 0.12 * push

            for i in range(n):
                if i in anchor:
                    positions[i] += forces[i] * 0.15
                else:
                    positions[i] += forces[i] * 0.85

    def _connected_components(self, adjacency: dict[int, set[int]]) -> list[list[int]]:
        seen: set[int] = set()
        components: list[list[int]] = []
        for start in range(len(self.elements)):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            component: list[int] = []
            while stack:
                cur = stack.pop()
                component.append(cur)
                for nxt in adjacency[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            component.sort()
            components.append(component)
        components.sort(key=lambda comp: (len(comp) == 0, comp[0] if comp else 0))
        return components

    def _unique_bond_pairs_in_component(self, component: Sequence[int]) -> list[tuple[int, int]]:
        comp_set = set(component)
        seen: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int]] = []
        for a, b in self.bonds:
            ia = int(a)
            ib = int(b)
            key = (min(ia, ib), max(ia, ib))
            if ia in comp_set and ib in comp_set and key not in seen:
                seen.add(key)
                pairs.append(key)
        pairs.sort()
        return pairs
