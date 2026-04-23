from OpenGL.GL import *

from src.rendering_module.interface.ar_background import draw_ar_background


def render_scene(venv, spawn_candidate=None):
    glClear(GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glPushMatrix()
    glTranslatef(0.0, 0.0, venv.global_zoom)
    venv.render()
    glPopMatrix()

    if spawn_candidate is not None:
        glPushMatrix()
        glTranslatef(0.0, 0.0, -80.0)
        venv.render_atom(spawn_candidate)
        glPopMatrix()


def render_frame(frame, bg_tex_id, bg_tex_size, venv, spawn_candidate=None):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    bg_tex_size = draw_ar_background(frame, bg_tex_id, bg_tex_size)
    render_scene(venv, spawn_candidate)
    return bg_tex_size
