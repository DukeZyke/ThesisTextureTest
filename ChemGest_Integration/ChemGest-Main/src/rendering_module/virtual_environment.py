import math
from OpenGL.GL import *
from OpenGL.GLU import *

from src.rendering_module.objects.object_renderer import render_atoms, render_bonds, render_atom
from src.rendering_module.objects.molecule_manager import MoleculeManager


class VirtualEnvironment:
    def __init__(self, atom_config_path=None):
        self.global_zoom = -80.0

        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0

        self.molecule_manager = MoleculeManager(atom_config_path)

        self.quadric = None


    @property
    def ATOM_DEFS(self):
        return self.molecule_manager.ATOM_DEFS

    @property
    def atoms(self):
        return self.molecule_manager.atoms

    @atoms.setter
    def atoms(self, value):
        self.molecule_manager.atoms = value

    @property
    def bonds(self):
        return self.molecule_manager.bonds

    @bonds.setter
    def bonds(self, value):
        self.molecule_manager.bonds = value

    @property
    def palette_elements(self):
        return self.molecule_manager.palette_elements

    @property
    def selected_element(self):
        return self.molecule_manager.selected_element

    def select_element(self, element_name):
        self.molecule_manager.select_element(element_name)

    def create_atom(self, element_name, x, y, z):
        return self.molecule_manager.create_atom(element_name, x, y, z)

    def add_atom(self, atom):
        self.molecule_manager.add_atom(atom)
    
    def remove_atom(self, atom):
        self.molecule_manager.remove_atom(atom)

    def get_element_symbol(self, element_name):
        return self.molecule_manager.get_element_symbol(element_name)

    def add_bond(self, atom1, atom2):
        self.molecule_manager.add_bond(atom1, atom2)

    def remove_bond(self, atom1, atom2):
        self.molecule_manager.remove_bond(atom1, atom2)

    def project_normalized_to_3d(self, norm_x, norm_y, target_z=0.0, override_zoom=None):
        aspect = 16.0 / 9.0
        fov_y = 45.0
        current_zoom = override_zoom if override_zoom is not None else self.global_zoom
        cam_dist = abs(current_zoom) - target_z
        h_world = 2.0 * cam_dist * math.tan(math.radians(fov_y / 2.0))
        w_world = h_world * aspect

        world_x = (norm_x - 0.5) * w_world
        world_y = -(norm_y - 0.5) * h_world
        return world_x, world_y

    def rotate_environment(self, rot_x_vel, rot_y_vel):
        self.rotation_x += rot_x_vel * 0.05
        self.rotation_y += rot_y_vel * 0.05

        if self.rotation_x > math.pi * 2:
            self.rotation_x -= math.pi * 2
        if self.rotation_x < -math.pi * 2:
            self.rotation_x += math.pi * 2
        if self.rotation_y > math.pi * 2:
            self.rotation_y -= math.pi * 2
        if self.rotation_y < -math.pi * 2:
            self.rotation_y += math.pi * 2

    def render(self):
        if self.quadric is None:
            self.quadric = gluNewQuadric()
            gluQuadricNormals(self.quadric, GLU_SMOOTH)

        glPushMatrix()

        glRotatef(math.degrees(self.rotation_x), 1.0, 0.0, 0.0)
        glRotatef(math.degrees(self.rotation_y), 0.0, 1.0, 0.0)
        glRotatef(math.degrees(self.rotation_z), 0.0, 0.0, 1.0)

        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.6, 0.6, 0.6, 1.0])

        # Render molecules (bonds + atoms) via object renderer
        render_bonds(self.quadric, self.bonds)
        render_atoms(self.quadric, self.atoms)

        glPopMatrix()

    def render_atom(self, atom):
        if self.quadric is None:
            self.quadric = gluNewQuadric()
            gluQuadricNormals(self.quadric, GLU_SMOOTH)

        render_atom(self.quadric, atom)



    # global toolbar button actions 

    def home(self):
        """Reset view to default orientation and zoom"""
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.global_zoom = -80.0

    def delete_selected(self):
        """Delete the currently selected atom or bond"""
        pass

    def set_mode(self, mode):
        """Set the interaction mode (e.g., 'mark', 'draw', etc.)"""
        pass

    def reset(self):
        """Reset the environment to initial state"""
        pass

    def center_views(self):
        """Center the view on the molecule"""
        pass

    def zoom(self, direction):
        """Zoom in (direction > 0) or out (direction < 0)"""
        if direction > 0:
            self.global_zoom += 5.0
        else:
            self.global_zoom -= 5.0

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        pass

    def undo(self):
        """Undo the last action"""
        pass

    def redo(self):
        """Redo the last undone action"""
        pass

    def save(self):
        """Save the current molecule structure"""
        pass

    def tutorial(self):
        """Open the tutorial or help documentation"""
        pass

