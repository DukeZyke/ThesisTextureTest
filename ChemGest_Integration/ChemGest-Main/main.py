import os
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('GLOG_logtostderr', '1')

from site import venv
import sys
import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import glDeleteTextures
from OpenGL.GLU import *

import chemgest_runtime  # noqa: F401
from benchmarking.profiler import benchmark_module

from src.rendering_module.virtual_environment import VirtualEnvironment
from src.temporal_interaction_module.interaction_manager import GestureInteractionManager
from src.rendering_module.opengl import setup_display, create_background_texture, setup_lighting, render_frame
from src.rendering_module.interface import draw_hud, draw_atom_palette
from src.rendering_module.interface.tabs.global_icons_tab import build_global_icons_buttons, draw_global_icons_tab, build_tutorial_button, draw_tutorial_button, handle_global_buttons_click, handle_tutorial_button_click
from src.rendering_module.interface.tabs.refine_tab import RefineTab
from src.rendering_module.interface.tabs.bond_tab import BondTab
from src.rendering_module.interface.tabs.geometry_tab import GeometryTab
from src.rendering_module.interface.tabs.visibility_tab import VisibilityTab
from src.rendering_module.interface.icon_manager import IconManager
from src.rendering_module.interface.output import OutputBox

print("Python executable:", sys.executable)

TAB_TOP_MARGIN = 70  # Distance from top of frame to top of first tab
TAB_GAP = 8

def reposition_tabs(tabs: list) -> None:
    """Set each tab's anchor_y dynamically so open panels push siblings down."""
    cursor_y = TAB_TOP_MARGIN
    for tab in tabs:
        tab.anchor_y = cursor_y
        
        # When closed, use the full panel_h_closed (not just handle height)
        # This is the actual screen space the closed tab occupies
        closed_height = getattr(tab, 'panel_h_closed', 64)
        cursor_y += closed_height + TAB_GAP
        
        if getattr(tab, 'is_open', False):
            # If open, add the additional open panel height on top
            open_height = getattr(tab, 'panel_h_open', closed_height)
            additional_height = open_height - closed_height
            cursor_y += additional_height + TAB_GAP
 
 
def open_tab_exclusively(tapped_tab, all_tabs: list) -> None:
    """Toggle tapped_tab; close all siblings first."""
    if getattr(tapped_tab, 'is_open', False):
        tapped_tab.is_open = False
    else:
        for tab in all_tabs:
            if tab is not tapped_tab:
                tab.is_open = False
        tapped_tab.is_open = True

@benchmark_module
def app_initialization():
    print("Initializing ChemGest: Augmented Reality System...")

@benchmark_module
def render_controller_loop():
    pygame.init()
    max_w, max_h = setup_display()
    setup_lighting()

    # Cached texture for the OpenCV background
    bg_tex_id, bg_tex_size = create_background_texture()

    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, 'models', 'chemgest_temporal_optimized.pth')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failure.")
        return

    venv = VirtualEnvironment()
    gesture_manager = GestureInteractionManager(model_path)
    icons = IconManager()
    print("Loaded icons:", icons.icons.keys())


    # for buttons
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global_buttons = build_global_icons_buttons(frame_w, frame_h, icons, venv)
    tutorial_button = build_tutorial_button(frame_w, frame_h, icons)
    output_box = OutputBox(frame_w=frame_w, frame_h=frame_h)

    
    # Position tabs vertically without overlap
    refine_tab   = RefineTab(frame_w=frame_w,   frame_h=frame_h, icon_manager=icons, anchor_y=TAB_TOP_MARGIN)
    bond_tab     = BondTab(frame_w=frame_w,     frame_h=frame_h, icon_manager=icons, anchor_y=TAB_TOP_MARGIN)
    geometry_tab = GeometryTab(frame_w=frame_w, frame_h=frame_h, icon_manager=icons, anchor_y=TAB_TOP_MARGIN)

    visibility_tab = VisibilityTab(frame_w=frame_w, frame_h=frame_h, icon_manager=icons, anchor_y=TAB_TOP_MARGIN)
    column_tabs = [refine_tab, bond_tab, geometry_tab]

    spawn_candidate = None
    last_gesture_id = -1

    print("Running Render Controller Loop... (Press Q to stop)")
    running = True
    show_hud = True # Toggle HUD visibility with 'H' key (handled in keybinds)

    try:
        while running and cap.isOpened():

            #Checking for keyboard input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:  # Toggle HUD with H
                        show_hud = not show_hud
                    if event.key == pygame.K_q:  # Quit with Q
                        running = False
                if event.type == pygame.MOUSEBUTTONDOWN: 
                    mx, my = pygame.mouse.get_pos()
                    win_w, win_h = pygame.display.get_surface().get_size()

                    scale_x = frame_w / win_w
                    scale_y = frame_h / win_h

                    fx = int(mx * scale_x)
                    fy = int(my * scale_y)

                    print(f"Mouse: ({mx},{my}) → Frame: ({fx},{fy})")

                    tab_handled = False
                    for tab in column_tabs:
                        if tab._hit_handle(fx, fy):
                            open_tab_exclusively(tab, column_tabs)
                            tab_handled = True
                            break
                        elif tab.is_open and tab._hit_panel(fx, fy):
                            tab.handle_click(fx, fy)
                            tab_handled = True
                            break
 
                    if not tab_handled:
                        if visibility_tab._hit_handle(fx, fy) or (visibility_tab.is_open and visibility_tab._hit_panel(fx, fy)):
                            visibility_tab.handle_click(fx, fy)
                            tab_handled = True

                    if not tab_handled:
                        handle_global_buttons_click(mx, my, global_buttons)
                        handle_tutorial_button_click(mx, my, tutorial_button)

            key_result = gesture_manager.process_events(venv)
            if not key_result.get('running', True):
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            interaction_result = gesture_manager.process_frame(frame, venv, spawn_candidate, last_gesture_id)
            spawn_candidate = interaction_result['spawn_candidate']
            last_gesture_id = interaction_result.get('current_gesture_id', last_gesture_id)

            if show_hud:
                draw_hud(
                    interaction_result['frame'],
                    gesture_manager.model_type,
                    interaction_result['active_state'],
                    interaction_result['conf_pct'],
                    len(gesture_manager.buffer),
                    interaction_result['hand_mode'],
                    interaction_result['top_probs'],
                    venv,
                    gesture_manager.keybind_manager.get_binding_strings(),
                )
            draw_atom_palette(
                interaction_result['frame'],
                venv,
                interaction_result['palette_buttons'],
                interaction_result['highlight_element'],
            )

            # buttons on global icons tab
            fingertip_pos = interaction_result.get('fingertip_coords', None) 
            draw_global_icons_tab(
                interaction_result['frame'], 
                global_buttons, 
                fingertip=fingertip_pos
            )

            draw_tutorial_button(frame, tutorial_button, fingertip_pos)

            # Global buttons now use mouse click callbacks instead of gesture triggers
            # (see handle_global_buttons_click in mouse event handler above)

            reposition_tabs(column_tabs)
            refine_tab.draw(interaction_result['frame'])
            bond_tab.draw(interaction_result['frame'])
            geometry_tab.draw(interaction_result['frame'])
            visibility_tab.draw(interaction_result['frame'])
            output_box.draw(interaction_result['frame'], column_tabs=column_tabs)

            output_box.set_data(
                name="Water",
                molecular_geometry="Bent",
                iupac_name="Oxidane",
            )
                      
            # Render 3D Scene + AR background in rendering module
            bg_tex_size = render_frame(
                interaction_result['frame'],
                bg_tex_id,
                bg_tex_size,
                venv,
                spawn_candidate,
            )

            pygame.display.flip()
            
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        try:
            glDeleteTextures([bg_tex_id])
        except Exception:
            pass
        pygame.quit()

if __name__ == "__main__":
    app_initialization()
    render_controller_loop()
