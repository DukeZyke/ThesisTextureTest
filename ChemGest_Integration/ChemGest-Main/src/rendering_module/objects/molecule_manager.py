from src.rendering_module.objects import VirtualAtom, VirtualBond
from src.utils.atom_config import load_atom_config, get_default_element


class MoleculeManager:
    def __init__(self, atom_config_path=None):
        self.atom_defs = load_atom_config(atom_config_path)
        self.selected_element = get_default_element(self.atom_defs)
        if self.selected_element is None:
            raise RuntimeError("No atom definitions found in atom config")

        self.palette_elements = sorted([e for e in self.atom_defs.keys() if e != self.selected_element])
        self.atoms = []
        self.bonds = []
        self.next_obj_id = 1

    @property
    def ATOM_DEFS(self):
        return self.atom_defs

    def select_element(self, element_name):
        if element_name not in self.atom_defs:
            return
        if element_name == self.selected_element:
            return

        self.palette_elements = sorted([e for e in self.atom_defs.keys() if e != element_name])
        self.selected_element = element_name

    def create_atom(self, element_name, x, y, z):
        info = self.atom_defs.get(element_name) or self.atom_defs.get('Carbon')

        atom = VirtualAtom(
            obj_id=self.next_obj_id,
            element=element_name,
            x=x,
            y=y,
            z=z,
            color=tuple(info.get('color', (1.0, 1.0, 1.0))),
            radius=info.get('radius', 3.8),
        )
        self.next_obj_id += 1
        return atom

    def add_atom(self, atom):
        self.atoms.append(atom)

    def remove_atom(self, atom):
        if atom in self.atoms:
            self.atoms.remove(atom)
        # Remove any bonds involving the deleted atom
        self.bonds = [b for b in self.bonds if b.atom1 != atom and b.atom2 != atom]

    def get_element_symbol(self, element_name):
        return self.atom_defs.get(element_name, {}).get('symbol', '?')

    def add_bond(self, atom1, atom2):
        for b in self.bonds:
            if (b.atom1 == atom1 and b.atom2 == atom2) or (b.atom1 == atom2 and b.atom2 == atom1):
                return
        self.bonds.append(VirtualBond(atom1, atom2))

    def remove_bond(self, atom1, atom2):
        self.bonds = [
            b
            for b in self.bonds
            if not ((b.atom1 == atom1 and b.atom2 == atom2) or (b.atom1 == atom2 and b.atom2 == atom1))
        ]
