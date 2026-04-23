from dataclasses import dataclass
from collections import deque
import itertools
from typing import Iterable, Sequence

import numpy as np


CPK_SPEC: dict[str, dict[str, float | tuple[float, float, float]]] = {
    "H": {"color": (0.95, 0.95, 0.95), "radius": 0.8},
    "O": {"color": (1.00, 0.25, 0.25), "radius": 1.4},
    "N": {"color": (0.30, 0.50, 1.00), "radius": 1.5},
    "C": {"color": (0.20, 0.20, 0.24), "radius": 1.6},
}

# Slightly increase target bond lengths so visual atom meshes do not appear overly merged.
BOND_LENGTH_SCALE = 1.12


def cpk_color(element: str) -> tuple[float, float, float]:
    symbol = str(element).upper()
    return tuple(CPK_SPEC.get(symbol, CPK_SPEC["C"])["color"])  # type: ignore[return-value]


def cpk_radius(element: str, base_radius: float = 1.0) -> float:
    symbol = str(element).upper()
    return float(CPK_SPEC.get(symbol, CPK_SPEC["C"])["radius"]) * float(base_radius)


# ============================================================================
# Interaction Weights for VSEPR Force-Directed Solver
# ============================================================================
# These weights control how strongly different interaction types repel each
# other during geometry resolution. Increasing ghost repulsion weights tilts
# the equilibrium to produce tighter bond angles in oxygen/nitrogen compounds.
#
# Tuning Guide:
#   Water (H₂O):   Currently ~109.47°, target 104.5° → increase ghost weights
#   Ammonia (NH₃): Currently ~107°+, target 107° → fine-tune for stability
#
BOND_BOND_WEIGHT = 1.0   # Base weight: bond-to-bond repulsion
GHOST_BOND_WEIGHT = 3.0  # Ghost-to-bond repulsion (makes ghosts push bonds together)
GHOST_GHOST_WEIGHT = 4.5  # Ghost-to-ghost repulsion (intermediate strength)
#
# ============================================================================
# WEIGHTED TORQUE IMPLEMENTATION FOR FINE-TUNED MOLECULAR ANGLES
# ============================================================================
#
# These weights control interaction strength during force-directed geometry
# relaxation. Different interaction types have different physical meanings:
#
#   BOND_BOND_WEIGHT (1.0):
#       Controls repulsion between bonds. Setting to 1.0 (baseline) means
#       bonds repel each other with standard strength.
#
#   GHOST_BOND_WEIGHT (3.0):
#       Controls how strongly lone-pair ghosts repel bonds. Increasing this
#       value makes lone pairs "push" bonded atoms closer together, reducing
#       bond angles from tetrahedral (109.47°) toward experimental values
#       like water (104.5°) or ammonia (107°).
#
#   GHOST_GHOST_WEIGHT (4.5):
#       Controls lone-pair-to-lone-pair repulsion. Proportionally higher
#       than GHOST_BOND_WEIGHT to maintain proper lone-pair geometry.
#
# ============================================================================
# ELEMENT-SPECIFIC TARGET ANGLES
# ============================================================================
#
# In addition to weights, the algorithm uses element-specific target cosine
# values when ghost lone pairs are present. This directly shifts the
# equilibrium point of the force-directed solver.
#
# Empirically-determined target angles (cosines) for common molecules:
#   - Water (O with 2 bonds, 2 ghosts):  cos = -0.40  → H-O-H ≈ 104.18°
#   - Ammonia (N with 3 bonds, 1 ghost): cos = -0.40  → H-N-H ≈ 107.97°
#   - Tetrahedral (no ghosts):           cos = -0.333 → all angles ≈ 109.47°
#
# The negative cosine convention: MORE NEGATIVE values produce SMALLER angles.
# For tuning:
#   - Water too wide? Make cos MORE negative (e.g., -0.42 → 102°)
#   - Water too narrow? Make cos LESS negative (e.g., -0.38 → 106°)
#
# ============================================================================
# STABILITY AND RE-CENTERING
# ============================================================================
#
# Component-aware re-centering (lines 345-350) ensures multi-molecule systems
# remain stable:
#   1. Each bonded component tracked independently
#   2. After geometry solving, each component centered on its centroid
#   3. This prevents molecules from drifting apart during rearrange()
#
# Bond-length correction pass (lines 468-475) maintains structural integrity:
#   1. After all force-directed steps, bonds are "snapped" to target lengths
#   2. Spring-like correction prevents bond stretching
#   3. Ensures rings stay closed and multi-atom chains stay coherent
#
# Result: Stable, physically realistic molecular geometry with accurate angles
#
# ============================================================================
# TUNING GUIDE
# ============================================================================
#
# If molecules look distorted or jitter:
#   1. Decrease GHOST_BOND_WEIGHT and GHOST_GHOST_WEIGHT in 0.2-0.5 steps
#   2. Example: GHOST_BOND_WEIGHT = 2.5, GHOST_GHOST_WEIGHT = 4.0
#
# If water angle too wide (> 105°):
#   1. Increase GHOST_BOND_WEIGHT (e.g., 3.5)
#   2. OR make target_cos slightly MORE negative (e.g., -0.41 → 103°)
#
# If water angle too narrow (< 103°):
#   1. Decrease GHOST_BOND_WEIGHT (e.g., 2.5)
#   2. OR make target_cos slightly LESS negative (e.g., -0.39 → 105°)
#

# If molecules look squashed or distorted, increase target angles (make more positive)
# e.g., OXYGEN_BOND_TARGET = -0.30 makes water wider, -0.22 makes it tighter


@dataclass(frozen=True)
class MoleculePreset:
    elements: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]
    positions: np.ndarray


class MoleculePresets:
    """Preset molecule templates that can be instantiated into a scene."""

    CPK_COLORS: dict[str, tuple[float, float, float]] = {
        "H": cpk_color("H"),
        "C": cpk_color("C"),
        "N": cpk_color("N"),
        "O": cpk_color("O"),
    }

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(v))
        if length < 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return v / length

    @staticmethod
    def _tetrahedral_vectors() -> np.ndarray:
        vectors = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-8, None)

    @staticmethod
    def _planar_pair(bond_length: float, bond_angle_degrees: float) -> np.ndarray:
        half_angle = np.radians(bond_angle_degrees * 0.5)
        return np.array(
            [
                [np.sin(half_angle) * bond_length, np.cos(half_angle) * bond_length, 0.0],
                [-np.sin(half_angle) * bond_length, np.cos(half_angle) * bond_length, 0.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _trigonal_pyramidal_triplet(bond_length: float) -> np.ndarray:
        return np.array(
            [
                [0.94280904 * bond_length, 0.0, -0.33333333 * bond_length],
                [-0.47140452 * bond_length, 0.81649658 * bond_length, -0.33333333 * bond_length],
                [-0.47140452 * bond_length, -0.81649658 * bond_length, -0.33333333 * bond_length],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _linear_pair(bond_length: float) -> np.ndarray:
        half = 0.5 * bond_length
        return np.array([[-half, 0.0, 0.0], [half, 0.0, 0.0]], dtype=np.float64)

    @staticmethod
    def _build_presets() -> dict[str, MoleculePreset]:
        water = MoleculePreset(
            elements=("O", "H", "H"),
            bonds=((0, 1), (0, 2)),
            positions=np.array(
                [[0.00, 0.00, 0.00], [0.28, 0.10, -0.08], [-0.22, -0.07, 0.12]],
                dtype=np.float64,
            ),
        )

        methane_dirs = MoleculePresets._tetrahedral_vectors()
        methane = MoleculePreset(
            elements=("C", "H", "H", "H", "H"),
            bonds=((0, 1), (0, 2), (0, 3), (0, 4)),
            positions=np.vstack(
                [
                    np.array([[0.00, 0.00, 0.00]], dtype=np.float64),
                    methane_dirs * 1.10,
                ]
            ),
        )

        ammonia = MoleculePreset(
            elements=("N", "H", "H", "H"),
            bonds=((0, 1), (0, 2), (0, 3)),
            positions=np.array(
                [[0.00, 0.00, 0.00], [0.30, 0.08, 0.02], [-0.25, 0.18, -0.11], [-0.05, -0.28, 0.10]],
                dtype=np.float64,
            ),
        )

        hydrogen = MoleculePreset(
            elements=("H", "H"),
            bonds=((0, 1),),
            positions=np.array([[0.00, 0.00, 0.00], [0.22, 0.06, -0.03]], dtype=np.float64),
        )

        return {
            "water": water,
            "methane": methane,
            "ammonia": ammonia,
            "hydrogen": hydrogen,
        }

    PRESETS: dict[str, MoleculePreset] = {}

    @classmethod
    def instantiate(
        cls,
        name: str,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
    ) -> tuple[list[str], list[tuple[int, int]], np.ndarray]:
        preset = cls.PRESETS[name.lower()]
        origin_vec = np.asarray(origin, dtype=np.float64)
        positions = preset.positions * float(scale) + origin_vec
        return list(preset.elements), list(preset.bonds), positions.copy()

    @classmethod
    def spawn(
        cls,
        name: str,
        focal_point: Sequence[float] = (0.0, 0.0, 0.0),
        scale: float = 1.0,
    ) -> tuple[list[str], list[tuple[int, int]], np.ndarray]:
        return cls.instantiate(name, origin=focal_point, scale=scale)

    @staticmethod
    def methane() -> tuple[list[str], list[tuple[int, int]], np.ndarray]:
        return MoleculePresets.instantiate("methane")

    @staticmethod
    def ammonia() -> tuple[list[str], list[tuple[int, int]], np.ndarray]:
        return MoleculePresets.instantiate("ammonia")

    @staticmethod
    def water() -> tuple[list[str], list[tuple[int, int]], np.ndarray]:
        return MoleculePresets.instantiate("water")

    @staticmethod
    def hydrogen() -> tuple[list[str], list[tuple[int, int]], np.ndarray]:
        return MoleculePresets.instantiate("hydrogen")


MoleculePresets.PRESETS = MoleculePresets._build_presets()
MoleculeTemplates = MoleculePresets


class MolecularEngine:
    """Headless molecular constraint solver for VSEPR-like stabilization."""

    _TETRA_DIRECTIONS = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )

    _BOND_ANGLE_DEGREES = {
        "H": 180.0,
        "C": 109.5,
        "N": 107.0,
        "O": 104.5,
    }

    _MAX_BONDS = {
        "H": 1,
        "C": 4,
        "N": 3,
        "O": 2,
    }

    # Approximate single-bond targets in scene units for stable rearrangement.
    _BOND_LENGTHS = {
        ("H", "H"): 0.74,
        ("C", "H"): 1.15,
        ("N", "H"): 1.01,
        ("O", "H"): 0.96,
        ("C", "C"): 1.54,
        ("C", "N"): 1.47,
        ("C", "O"): 1.43,
        ("N", "N"): 1.45,
        ("N", "O"): 1.40,
        ("O", "O"): 1.48,
    }

    _LONE_PAIR_COUNTS = {
        "C": 0,
        "N": 1,
        "O": 2,
    }

    def __init__(self, iterations: int = 8, stiffness: float = 0.55) -> None:
        self.iterations = max(int(iterations), 1)
        self.stiffness = float(np.clip(stiffness, 0.05, 0.95))

        # Persistent view-mode state used by rearrangement physics.
        # Supported: "ball_and_stick", "space_filling"
        self.view_mode = "space_filling"
        self.target_distance = 2.4  # updated by set_view_mode()

        # Toggle states for visualization features (step 7)
        self.show_clouds = False
        self.show_lone_pairs = False
        self.show_hybridization = False
        self.show_electron_clouds = False

        self.elements = np.empty((0,), dtype="<U2")
        self.bonds = np.empty((0, 2), dtype=np.int32)
        self._neighbors: list[np.ndarray] = []

        self.raw_positions = np.empty((0, 3), dtype=np.float64)
        self._stabilized_positions = np.empty((0, 3), dtype=np.float64)
        self._stabilized_valid = False

        # Initialize view mode derived target distance.
        self.set_view_mode(self.view_mode)

    def set_structure(
        self,
        elements: Sequence[str],
        bonds: Iterable[tuple[int, int]],
        positions: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.elements = np.array([str(e).upper() for e in elements], dtype="<U2")
        count = len(self.elements)

        pair_list: list[tuple[int, int]] = []
        for a, b in bonds:
            ia = int(a)
            ib = int(b)
            if ia == ib:
                continue
            if ia < 0 or ib < 0 or ia >= count or ib >= count:
                continue
            pair_list.append((ia, ib))

        if pair_list:
            self.bonds = np.array(pair_list, dtype=np.int32)
        else:
            self.bonds = np.empty((0, 2), dtype=np.int32)

        neighbors: list[list[int]] = [[] for _ in range(count)]
        for a, b in self.bonds:
            neighbors[a].append(int(b))
            neighbors[b].append(int(a))
        self._neighbors = [np.array(n, dtype=np.int32) for n in neighbors]

        if positions is None:
            self.raw_positions = np.zeros((count, 3), dtype=np.float64)
        else:
            arr = np.asarray(positions, dtype=np.float64)
            if arr.size == 0 and count == 0:
                arr = np.empty((0, 3), dtype=np.float64)
            if arr.ndim != 2 or arr.shape != (count, 3):
                raise ValueError("positions must be shaped (N, 3) and match element count")
            self.raw_positions = arr.copy()
        self._stabilized_positions = self.raw_positions.copy()
        self._stabilized_valid = False

    def invalidate(self) -> None:
        self._stabilized_valid = False

    def get_structure(self) -> tuple[list[str], list[tuple[int, int]], np.ndarray]:
        return (
            [str(e) for e in self.elements.tolist()],
            [(int(a), int(b)) for a, b in self.bonds.tolist()],
            self.get_stabilized_positions(),
        )

    def update_raw_positions(self, positions: Sequence[Sequence[float]]) -> None:
        arr = np.asarray(positions, dtype=np.float64)
        if arr.size == 0:
            arr = np.empty((0, 3), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("positions must be shaped (N, 3)")
        if arr.shape[0] != self.raw_positions.shape[0]:
            raise ValueError("positions length must match engine atom count")
        self.raw_positions = arr.copy()
        self.invalidate()

    def update_raw_pos(self, atom_index: int, position: Sequence[float]) -> None:
        idx = int(atom_index)
        if idx < 0 or idx >= self.raw_positions.shape[0]:
            return
        self.raw_positions[idx] = np.asarray(position, dtype=np.float64)
        self.invalidate()

    def _bond_order_map(self) -> dict[tuple[int, int], int]:
        order_map: dict[tuple[int, int], int] = {}
        for a, b in self.bonds:
            ia = int(a)
            ib = int(b)
            if ia == ib:
                continue
            key = (min(ia, ib), max(ia, ib))
            order_map[key] = order_map.get(key, 0) + 1
        return order_map

    def _atom_bond_values(self) -> list[int]:
        totals = [0 for _ in range(len(self.elements))]
        for (a, b), order in self._bond_order_map().items():
            clamped = max(1, min(int(order), 3))
            totals[a] += clamped
            totals[b] += clamped
        return totals

    def add_bond(self, atom_a: int, atom_b: int) -> bool:
        ia = int(atom_a)
        ib = int(atom_b)
        n = len(self.elements)
        if ia < 0 or ib < 0 or ia >= n or ib >= n or ia == ib:
            return False

        max_a = int(self._MAX_BONDS.get(str(self.elements[ia]).upper(), 0))
        max_b = int(self._MAX_BONDS.get(str(self.elements[ib]).upper(), 0))
        if max_a <= 0 or max_b <= 0:
            return False

        pair = (min(ia, ib), max(ia, ib))
        order_map = self._bond_order_map()
        existing_order = order_map.get(pair, 0)
        if existing_order >= 3:
            return False

        bond_totals = self._atom_bond_values()
        new_bond_value = 1
        potential_a = bond_totals[ia] + new_bond_value
        potential_b = bond_totals[ib] + new_bond_value

        # Hard valency blocker: illegal bond attempts are rejected.
        if potential_a > max_a or potential_b > max_b:
            return False

        if self.bonds.size == 0:
            self.bonds = np.array([[ia, ib]], dtype=np.int32)
        else:
            self.bonds = np.vstack([self.bonds, np.array([[ia, ib]], dtype=np.int32)])
        self._rebuild_neighbors()
        self.invalidate()
        return True

    def set_view_mode(self, mode: str) -> None:
        """
        Update visualization physics mode only.
        This function intentionally does NOT mutate topology or hydrate atoms.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        if normalized in ("ball_stick", "ball_and_stick"):
            self.view_mode = "ball_and_stick"
            self.target_distance = 2.4
        elif normalized in ("space_filling",):
            self.view_mode = "space_filling"
            # Per-bond distance is dynamic in space-filling; target_distance stores baseline fallback.
            self.target_distance = 2.4
        else:
            # Fallback for legacy calls
            self.view_mode = "ball_and_stick"
            self.target_distance = 2.4

    def rearrange_structure(self, atoms: Sequence[object] | np.ndarray | None = None, mode: str = 'vsepr') -> np.ndarray:
        """
        Master refinement pass:
        1) synchronize raw positions (if supplied)
        2) run auto hydration once for valency completion
        3) solve geometry using distances derived from current view mode
        """
        if atoms is not None:
            if isinstance(atoms, np.ndarray):
                self.update_raw_positions(atoms)
            else:
                self.update_raw_positions([getattr(a, "position") for a in atoms])

        # Keep backward compatibility: explicit mode argument may update current engine mode.
        normalized_mode = str(mode).strip().lower().replace("-", "_")
        if normalized_mode in ("ball_stick", "ball_and_stick", "space_filling"):
            self.set_view_mode(normalized_mode)

        n = self.raw_positions.shape[0]
        if n == 0:
            self._stabilized_positions = self.raw_positions.copy()
            self._stabilized_valid = True
            return self._stabilized_positions

        # Hydration must happen once per refinement pass.
        self._apply_auto_saturation()

        n = self.raw_positions.shape[0]
        if n == 0:
            self._stabilized_positions = self.raw_positions.copy()
            self._stabilized_valid = True
            return self._stabilized_positions

        pos = self.raw_positions.copy()
        components = self._connected_components(n)
        original_centroids = {
            tuple(comp): np.mean(pos[np.array(comp, dtype=np.int32)], axis=0)
            for comp in components
        }

        if self.bonds.shape[0] == 0:
            self._stabilized_positions = self.raw_positions.copy()
            self._stabilized_valid = True
            return self._stabilized_positions

        # Dynamic rest-length selection from current view mode.
        if self.view_mode == "ball_and_stick":
            bond_lengths = np.full(self.bonds.shape[0], 2.4, dtype=np.float64)
        elif self.view_mode == "space_filling":
            # ((RadiusA + RadiusB) * 0.66) + 0.6
            bond_lengths = self._cpk_overlap_bond_lengths()
        else:
            bond_lengths = np.full(self.bonds.shape[0], self.target_distance, dtype=np.float64)

        pos = self.resolve_geometry(pos, bond_lengths)

        # Re-center each disconnected molecule independently to prevent cluster drift.
        for comp in components:
            comp_idx = np.array(comp, dtype=np.int32)
            key = tuple(comp)
            target_centroid = original_centroids[key]
            solved_centroid = np.mean(pos[comp_idx], axis=0)
            pos[comp_idx] += (target_centroid - solved_centroid)

        self._stabilized_positions = pos
        self._stabilized_valid = True
        return self._stabilized_positions

    def resolve_geometry(self, positions: np.ndarray, bond_lengths: np.ndarray) -> np.ndarray:
        """Force-directed 3D VSEPR relaxation using bonded vectors plus ghost lone-pair vectors."""
        if positions.shape[0] == 0 or self.bonds.shape[0] == 0:
            return positions

        pos = positions.copy()
        pair_length: dict[tuple[int, int], float] = {}
        for i, (a, b) in enumerate(self.bonds):
            ia = int(a)
            ib = int(b)
            l = float(bond_lengths[i]) if i < len(bond_lengths) else 1.2
            pair_length[(ia, ib)] = l
            pair_length[(ib, ia)] = l

        target_cos = -1.0 / 3.0  # tetrahedral ~109.47° (default)
        local_steps = max(6, self.iterations)
        outer_steps = max(4, self.iterations * 2)

        for _ in range(outer_steps):
            for center in range(pos.shape[0]):
                neigh = [int(n) for n in self._neighbors[center]]
                if not neigh:
                    continue

                rel = pos[np.array(neigh, dtype=np.int32)] - pos[center]
                lengths = np.linalg.norm(rel, axis=1)
                lengths = np.clip(lengths, 1e-8, None)
                bond_dirs = rel / lengths[:, None]

                ghost_count = self._ghost_count(center, len(neigh))
                ghost_dirs = self._init_ghost_directions(bond_dirs, ghost_count)

                if ghost_dirs.shape[0] > 0:
                    dirs = np.vstack([bond_dirs, ghost_dirs])
                else:
                    dirs = bond_dirs.copy()

                if dirs.shape[0] >= 2:
                    bond_count = len(neigh)  # First N directions are bonds
                    
                    # Adjust target angle based on central atom and ghost count
                    # This creates tighter bond angles when lone pairs are present
                    center_elem = self.elements[center].upper() if center < len(self.elements) else "C"
                    if ghost_count > 0:
                        if center_elem == "O" and bond_count == 2:
                            # Water: target ~104.5° for O-H bonds
                            # Fine-tuned empirically: -0.40 gives 104.18°
                            active_target_cos = -0.40
                        elif center_elem == "N" and bond_count == 3:
                            # Ammonia: target ~107° for N-H bonds
                            # Fine-tuning: -0.38 gives 108.61°, need slightly more negative
                            active_target_cos = -0.40  # Increasing to -0.40
                        else:
                            active_target_cos = target_cos
                    else:
                        active_target_cos = target_cos  # Tetrahedral for no ghosts
                    
                    for _inner in range(local_steps):
                        forces = np.zeros_like(dirs)
                        for i in range(dirs.shape[0]):
                            for j in range(i + 1, dirs.shape[0]):
                                di = dirs[i]
                                dj = dirs[j]
                                cos_ij = float(np.clip(np.dot(di, dj), -1.0, 1.0))
                                err = cos_ij - active_target_cos
                                
                                # Determine interaction type and apply weight
                                is_ghost_i = i >= bond_count
                                is_ghost_j = j >= bond_count
                                if is_ghost_i and is_ghost_j:
                                    weight = GHOST_GHOST_WEIGHT
                                elif is_ghost_i or is_ghost_j:
                                    weight = GHOST_BOND_WEIGHT
                                else:
                                    weight = BOND_BOND_WEIGHT
                                
                                grad_i = dj - cos_ij * di
                                grad_j = di - cos_ij * dj
                                forces[i] -= weight * err * grad_i
                                forces[j] -= weight * err * grad_j

                        tangent = forces - np.sum(forces * dirs, axis=1, keepdims=True) * dirs
                        dirs = self._normalize_rows(dirs + tangent * (0.14 * self.stiffness))

                desired_bond_dirs = dirs[: len(neigh)]
                target_pos = []
                for idx, nb in enumerate(neigh):
                    bl = pair_length.get((center, nb), 1.2)
                    target_pos.append(pos[center] + desired_bond_dirs[idx] * bl)

                target_arr = np.array(target_pos, dtype=np.float64)
                move_alpha = 0.28
                pos[np.array(neigh, dtype=np.int32)] = (
                    (1.0 - move_alpha) * pos[np.array(neigh, dtype=np.int32)]
                    + move_alpha * target_arr
                )

            # Bond-length correction pass.
            a = self.bonds[:, 0]
            b = self.bonds[:, 1]
            delta = pos[b] - pos[a]
            dist = np.linalg.norm(delta, axis=1)
            safe_dist = np.clip(dist, 1e-8, None)
            mismatch = (safe_dist - bond_lengths) / safe_dist
            correction = 0.5 * mismatch[:, None] * delta
            pos[a] += correction
            pos[b] -= correction

        return pos

    def auto_arrange(self, atoms: Sequence[object] | np.ndarray | None = None, mode: str = 'vsepr') -> np.ndarray:
        return self.rearrange_structure(atoms, mode)

    def update_bond_distances(self, mode: str = "ball_and_stick") -> np.ndarray:
        """
        Dynamic bond distance update for visualization mode.

        Supported modes:
        - ball_and_stick / ball-stick: fixed bond length = 2.4
        - space_filling / space-filling: ((RadiusA + RadiusB) * 0.66) + 0.6
        """
        if self.raw_positions.shape[0] == 0:
            return self.raw_positions.copy()

        self.set_view_mode(mode)

        if self.view_mode == "ball_and_stick":
            bond_lengths = np.full(self.bonds.shape[0], 2.8, dtype=np.float64)  # Increased gap
        elif self.view_mode == "space_filling":
            bond_lengths = self._cpk_overlap_bond_lengths()
        else:
            bond_lengths = np.full(self.bonds.shape[0], self.target_distance, dtype=np.float64)

        stabilized = self.resolve_geometry(self.raw_positions.copy(), bond_lengths)
        self._stabilized_positions = stabilized
        self._stabilized_valid = True
        return stabilized

    def recalculate_bond_distances(self, mode: str = "vsepr"):
        """Backward-compatible wrapper around update_bond_distances()."""
        if str(mode).strip().lower().replace("-", "_") in ("ball_stick", "ball_and_stick"):
            return self.update_bond_distances("ball_and_stick")
        if str(mode).strip().lower().replace("-", "_") in ("space_filling",):
            return self.update_bond_distances("space_filling")
        return self.update_bond_distances(mode)

    def solve_constraints(self, atoms: Sequence[object] | np.ndarray | None = None, mode: str = 'vsepr') -> np.ndarray:
        return self.auto_arrange(atoms, mode)

    def get_stabilized_positions(self) -> np.ndarray:
        if self._stabilized_valid:
            return self._stabilized_positions.copy()
        return self.raw_positions.copy()

    def _target_bond_directions(self, center_index: int, bond_count: int) -> np.ndarray | None:
        element = self.elements[center_index] if center_index < len(self.elements) else "C"
        element = element.upper()

        if element not in self._BOND_ANGLE_DEGREES:
            return None

        bond_count = min(int(bond_count), self._MAX_BONDS.get(element, bond_count))
        if bond_count <= 0:
            return None

        domains = min(max(bond_count + self._LONE_PAIR_COUNTS.get(element, 0), bond_count), 4)
        dirs = self._TETRA_DIRECTIONS[:domains].copy()
        dirs = self._normalize_rows(dirs)

        if domains > bond_count:
            lone_pair_dirs = dirs[bond_count:domains]
            ghost_axis = self._normalize_rows(np.sum(lone_pair_dirs, axis=0, keepdims=True))[0]
            desired_angle = self._BOND_ANGLE_DEGREES[element]
            base_angle = self._BOND_ANGLE_DEGREES["C"]
            squeeze = np.clip((base_angle - desired_angle) / 15.0, 0.0, 0.45)
            if squeeze > 0.0:
                pull = -ghost_axis
                bond_dirs = []
                for d in dirs[:bond_count]:
                    blended = self._normalize_rows(((1.0 - squeeze) * d + squeeze * pull)[None, :])[0]
                    bond_dirs.append(blended)
                return np.asarray(bond_dirs, dtype=np.float64)

        return dirs[:bond_count]

    def _ghost_count(self, center_index: int, bond_count: int) -> int:
        element = self.elements[center_index] if center_index < len(self.elements) else "C"
        symbol = str(element).upper()
        if symbol == "N" and bond_count == 3:
            return 1
        if symbol == "O" and bond_count == 2:
            return 2
        return 0

    def _init_ghost_directions(self, bond_dirs: np.ndarray, ghost_count: int) -> np.ndarray:
        if ghost_count <= 0:
            return np.empty((0, 3), dtype=np.float64)

        candidates = [d.astype(np.float64) for d in self._TETRA_DIRECTIONS]
        candidates.extend((-d).astype(np.float64) for d in self._TETRA_DIRECTIONS)
        selected: list[np.ndarray] = []

        used = []
        for d in bond_dirs:
            l = float(np.linalg.norm(d))
            if l > 1e-8:
                used.append(d / l)

        for _ in range(ghost_count):
            best = None
            best_score = -1e9
            for cand in candidates:
                cand_n = cand / max(float(np.linalg.norm(cand)), 1e-8)
                all_used = used + selected
                if not all_used:
                    score = 1.0
                else:
                    score = min(float(1.0 - np.dot(cand_n, u)) for u in all_used)
                if score > best_score:
                    best_score = score
                    best = cand_n
            if best is None:
                best = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            selected.append(best)

        return np.array(selected, dtype=np.float64)

    def _connected_components(self, count: int) -> list[list[int]]:
        seen: set[int] = set()
        components: list[list[int]] = []

        for start in range(count):
            if start in seen:
                continue
            q = deque([start])
            seen.add(start)
            comp: list[int] = []
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nxt in self._neighbors[cur]:
                    idx = int(nxt)
                    if idx not in seen:
                        seen.add(idx)
                        q.append(idx)
            comp.sort()
            components.append(comp)

        return components

    def _rebuild_neighbors(self) -> None:
        count = len(self.elements)
        neighbors: list[list[int]] = [[] for _ in range(count)]
        for a, b in self.bonds:
            ia = int(a)
            ib = int(b)
            if ia < 0 or ib < 0 or ia >= count or ib >= count or ia == ib:
                continue
            neighbors[ia].append(ib)
            neighbors[ib].append(ia)
        self._neighbors = [np.array(n, dtype=np.int32) for n in neighbors]

    def _apply_auto_saturation(self) -> None:
        if len(self.elements) == 0:
            return

        elements = [str(e).upper() for e in self.elements.tolist()]
        positions: list[np.ndarray] = [self.raw_positions[i].copy() for i in range(self.raw_positions.shape[0])]
        bonds: list[tuple[int, int]] = [(int(a), int(b)) for a, b in self.bonds.tolist()]

        neighbors: list[set[int]] = [set() for _ in range(len(elements))]
        bond_value: list[int] = [0 for _ in range(len(elements))]
        pair_order_counter: dict[tuple[int, int], int] = {}
        for a, b in bonds:
            if 0 <= a < len(elements) and 0 <= b < len(elements) and a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)

                key = (min(a, b), max(a, b))
                pair_order_counter[key] = pair_order_counter.get(key, 0) + 1

        # Count bond value (single=1, double=2, triple=3) per atom.
        for (a, b), order in pair_order_counter.items():
            clamped = max(1, min(int(order), 3))
            bond_value[a] += clamped
            bond_value[b] += clamped

        # Saturate only existing non-hydrogen atoms in this pass.
        original_count = len(elements)
        for center in range(original_count):
            element = elements[center]
            if element == "H":
                continue

            max_bonds = int(self._MAX_BONDS.get(element, 0))
            if max_bonds <= 0:
                continue

            # Strict valency gate: stop adding hydrogens when bond order already
            # satisfies the element valency (e.g. O= with valency 2).
            missing = max_bonds - bond_value[center]
            if missing <= 0:
                continue

            used_dirs = []
            center_pos = positions[center]
            for nb in neighbors[center]:
                delta = positions[nb] - center_pos
                length = float(np.linalg.norm(delta))
                if length > 1e-8:
                    used_dirs.append(delta / length)

            for _ in range(missing):
                new_dir = self._best_open_direction(used_dirs)
                bond_len = self._BOND_LENGTHS.get(tuple(sorted((element, "H"))), 1.0)
                new_pos = center_pos + new_dir * bond_len

                new_idx = len(elements)
                elements.append("H")
                positions.append(new_pos.astype(np.float64))
                neighbors.append({center})
                neighbors[center].add(new_idx)
                bonds.append((center, new_idx))
                bond_value.append(1)
                bond_value[center] += 1
                used_dirs.append(new_dir)

        self.elements = np.array(elements, dtype="<U2")
        self.raw_positions = np.array(positions, dtype=np.float64)
        self.bonds = np.array(bonds, dtype=np.int32) if bonds else np.empty((0, 2), dtype=np.int32)
        self._rebuild_neighbors()

    def _best_open_direction(self, used_dirs: list[np.ndarray]) -> np.ndarray:
        candidates = [d.astype(np.float64) for d in self._TETRA_DIRECTIONS]
        candidates.extend((-d).astype(np.float64) for d in self._TETRA_DIRECTIONS)

        used = []
        for d in used_dirs:
            length = float(np.linalg.norm(d))
            if length > 1e-8:
                used.append(d / length)

        if not used:
            base = self._TETRA_DIRECTIONS[0]
            return base / np.linalg.norm(base)

        best = None
        best_score = -1e9
        for cand in candidates:
            cand_n = cand / max(float(np.linalg.norm(cand)), 1e-8)
            score = min(float(1.0 - np.dot(cand_n, u)) for u in used)
            if score > best_score:
                best_score = score
                best = cand_n

        if best is None:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return best

    def _cpk_overlap_bond_lengths(self) -> np.ndarray:
        if self.bonds.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        
        lengths = np.empty((self.bonds.shape[0],), dtype=np.float64)
        for i, (a, b) in enumerate(self.bonds):
            ra = cpk_radius(self.elements[int(a)])
            rb = cpk_radius(self.elements[int(b)])
            lengths[i] = ((ra + rb) * 0.66) + 0.6  # 1/3 overlap + readability spacing
        return lengths

    def _target_bond_lengths(self) -> np.ndarray:
        if self.bonds.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)

        lengths = np.empty((self.bonds.shape[0],), dtype=np.float64)
        for i, (a, b) in enumerate(self.bonds):
            ea = self.elements[int(a)] if int(a) < len(self.elements) else "C"
            eb = self.elements[int(b)] if int(b) < len(self.elements) else "C"
            key = tuple(sorted((str(ea).upper(), str(eb).upper())))
            base_len = self._BOND_LENGTHS.get(key, 1.20)
            lengths[i] = float(base_len) * BOND_LENGTH_SCALE
        return lengths

    def _best_direction_permutation(self, current: np.ndarray, target: np.ndarray) -> tuple[int, ...]:
        count = current.shape[0]
        best_perm = tuple(range(count))
        best_score = -1e18
        for perm in itertools.permutations(range(count)):
            score = float(np.sum(current * target[list(perm)], axis=None))
            if score > best_score:
                best_score = score
                best_perm = perm
        return best_perm

    def _rotate_towards(self, current: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
        cur = self._normalize_rows(current)
        tar = self._normalize_rows(target)

        axis = np.cross(cur, tar)
        axis_norm = np.linalg.norm(axis, axis=1)

        dot = np.sum(cur * tar, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot) * fraction

        out = np.empty_like(cur)
        strong = axis_norm > 1e-8

        if np.any(strong):
            axis_s = axis[strong] / axis_norm[strong][:, None]
            v = cur[strong]
            ct = np.cos(theta[strong])[:, None]
            st = np.sin(theta[strong])[:, None]
            out[strong] = (
                v * ct
                + np.cross(axis_s, v) * st
                + axis_s * np.sum(axis_s * v, axis=1)[:, None] * (1.0 - ct)
            )

        if np.any(~strong):
            out[~strong] = self._normalize_rows((1.0 - fraction) * cur[~strong] + fraction * tar[~strong])

        return self._normalize_rows(out)

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return vectors / norms

    def auto_hydrate(self) -> dict[int, int]:
        original_count = len(self.elements)
        self._apply_auto_saturation()
        bond_totals = self._atom_bond_values()
        gaps = {}
        for i in range(original_count):
            elem = str(self.elements[i]).upper()
            if elem in self._MAX_BONDS:
                max_b = self._MAX_BONDS[elem]
                gaps[i] = max(0, max_b - bond_totals[i])
        return gaps

    def hybridization(self, atom_idx: int) -> str:
        idx = int(atom_idx)
        if idx < 0 or idx >= len(self.elements):
            return 'unknown'
        bond_count = len(self._neighbors[idx])
        ghost_count = self._ghost_count(idx, bond_count)
        total_domains = bond_count + ghost_count
        if total_domains == 2:
            return 'sp'
        elif total_domains == 3:
            return 'sp2'
        elif total_domains == 4:
            return 'sp3'
        else:
            return 'unknown'

    def set_toggle(self, name: str, value: bool) -> None:
        name = name.lower()
        if name == 'clouds':
            self.show_clouds = value
        elif name == 'lone_pairs':
            self.show_lone_pairs = value
        elif name == 'hybridization':
            self.show_hybridization = value
        elif name == 'electron_clouds':
            self.show_electron_clouds = value

    def get_toggles(self) -> dict[str, bool]:
        return {
            'show_clouds': self.show_clouds,
            'show_lone_pairs': self.show_lone_pairs,
            'show_hybridization': self.show_hybridization,
            'show_electron_clouds': self.show_electron_clouds,
        }
