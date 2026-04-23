import math
from OpenGL.GL import *
from OpenGL.GLU import *


def _render_toon_outline(quadric, radius, color, scale):
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glDisable(GL_LIGHTING)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_FRONT)
    glColor3f(color[0], color[1], color[2])
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    glPushMatrix()
    glScalef(scale, scale, scale)
    gluSphere(quadric, radius, 24, 24)
    glPopMatrix()

    glPopAttrib()


def render_atom(quadric, atom):
    if quadric is None:
        return

    glPushMatrix()
    glTranslatef(atom.x, atom.y, atom.z)

    grab_scale = 1.3 if atom.is_grabbed else 1.0
    outline_scale = grab_scale * 1.12

    if atom.is_flick_selected or atom.is_flick_delete or atom.is_grabbed:
        if atom.is_flick_delete:
            outline_color = (1.0, 0.0, 0.0)
        elif atom.is_flick_selected:
            outline_color = (1.0, 1.0, 1.0)
        else:
            outline_color = (1.0, 1.0, 1.0)

        _render_toon_outline(quadric, atom.base_radius, outline_color, outline_scale)

    glScalef(grab_scale, grab_scale, grab_scale)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [atom.color[0], atom.color[1], atom.color[2], 1.0])
    glColor3f(atom.color[0], atom.color[1], atom.color[2])
    gluSphere(quadric, atom.base_radius, 24, 24)

    glPopMatrix()


def render_atoms(quadric, atoms):
    if quadric is None:
        return
    for atom in atoms:
        render_atom(quadric, atom)


def render_bonds(quadric, bonds):
    if quadric is None:
        return
    for bond in bonds:
        a1, a2 = bond.atom1, bond.atom2
        dx = a2.x - a1.x
        dy = a2.y - a1.y
        dz = a2.z - a1.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 0.0001:
            continue

        glPushMatrix()
        glTranslatef(a1.x, a1.y, a1.z)

        yaw = math.degrees(math.atan2(dx, dz))
        pitch = math.degrees(math.atan2(-dy, math.sqrt(dx * dx + dz * dz)))

        glRotatef(yaw, 0.0, 1.0, 0.0)
        glRotatef(pitch, 1.0, 0.0, 0.0)

        glColor3f(0.8, 0.8, 0.8)
        gluCylinder(quadric, 1.0, 1.0, dist, 12, 1)

        glPopMatrix()
