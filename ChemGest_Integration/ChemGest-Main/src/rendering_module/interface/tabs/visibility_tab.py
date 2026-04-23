import numpy as np
import cv2
from PIL import Image, ImageDraw
from src.rendering_module.interface.slide_tab import SlideTab
from src.rendering_module.interface.icon_manager import IconManager
from src.rendering_module.interface.ui_button import INTER_FONT


class ToggleRow:
    """A single labelled toggle switch row."""

    TOGGLE_W   = 40
    TOGGLE_H   = 24
    KNOB_R     = 10
    COLOR_ON   = (180, 180, 180)   # BGR light grey when ON
    COLOR_OFF  = (100, 100, 100)   # BGR dark grey when OFF
    KNOB_COLOR = (230, 230, 230)   # BGR white-ish knob

    def __init__(self, id: str, label: str, default: bool = False, on_change=None):
        self.id          = id
        self.label       = label
        self.value       = default
        self.on_change   = on_change
        self.toggle_rect = (0, 0, 0, 0)   # set each frame by _draw_toggles

    def toggle(self):
        self.value = not self.value
        if self.on_change:
            self.on_change(self.id, self.value)

    def contains_toggle(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self.toggle_rect
        return x0 <= x <= x1 and y0 <= y <= y1


class VisibilityTab(SlideTab):

    # ── Layout ────────────────────────────────────────────────────────────
    ROW_H          = 32     # height of each toggle row
    PADDING_TOP    = 40     # space from top of panel to first row (label lives here)
    PADDING_SIDE   = 12     # horizontal inset inside panel
    ROW_GAP        = 4      # gap between rows
    PADDING_BOT    = 0   # space below last row
    PANEL_W        = 180    # panel content width

    # ── Expandable height (mirrors left-tab pattern) ───────────────────────
    PANEL_H_CLOSED = 64     # height when tab is closed

    # ── Right-side corner hiding (mirrors SlideTab.PANEL_OFFSET_MAX) ──────
    # Positive value pushes the panel RIGHT (off-screen), hiding the right corners.
    PANEL_OFFSET_MAX = 16

    # ── Label / content left-shift ─────────────────────────────────────────
    CONTENT_SHIFT  = 4     # shift label and toggles this many px to the left
    LABEL_Y_OFFSET = 24

    def __init__(self, frame_w: int, frame_h: int,
                 icon_manager: IconManager = None,
                 anchor_y: int = None):
        if icon_manager is None:
            icon_manager = IconManager()

        self._rows: list[ToggleRow] = self._build_rows()

        # Expandable heights
        self.panel_h_open   = self.PANEL_H_CLOSED + self._calc_content_h(len(self._rows))
        self.panel_h_closed = self.PANEL_H_CLOSED
        self.frame_h        = frame_h

        if anchor_y is None:
            anchor_y = (frame_h - self.panel_h_open) // 2

        super().__init__(
            id="visibility_tab",
            label="View",
            anchor_y=anchor_y,
            frame_w=frame_w,
            frame_h=frame_h,
            panel_w=self.PANEL_W,
            panel_h=self.PANEL_H_CLOSED,   # start closed
            icon=icon_manager.get_icon("view"),
            icon_manager=icon_manager,
        )

        self.panel_arrow = icon_manager.get_icon("view_panel_arrow")

    # ── Layout-engine contract (read by reposition_tabs if ever added) ────

    @property
    def handle_height(self) -> int:
        return self.PANEL_H_CLOSED

    @property
    def panel_height(self) -> int:
        """Extra height below the handle when open (0 when closed)."""
        if not self.is_open:
            return 0
        return self.panel_h_open - self.panel_h_closed

    # ── Public API ────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return {id: bool} for all toggles."""
        return {r.id: r.value for r in self._rows}

    def handle_click(self, x: int, y: int):
        if self._hit_handle(x, y):
            self._toggle()
            return
        if self.is_open:
            for row in self._rows:
                if row.contains_toggle(x, y):
                    row.toggle()
                    break

    def handle_point(self, x: int, y: int, is_pinched: bool):
        super().handle_point(x, y, is_pinched)
        if not self.is_open:
            return
        if is_pinched and not self.was_pinched:
            for row in self._rows:
                if row.contains_toggle(x, y):
                    row.toggle()
                    break

    def draw(self, frame: np.ndarray):
        """Right-edge slide with expandable height."""
        self._animate()

        # ── Expand/contract panel height with slide progress ──────────────
        progress = self._slide_x / self.panel_w if self.panel_w > 0 else 0
        progress = max(0, min(1, progress))
        self.panel_h = int(
            self.panel_h_closed
            + (self.panel_h_open - self.panel_h_closed) * progress
        )

        # ── Right-corner hiding: push panel right as it opens ─────────────
        # Mirrors SlideTab's left-side dynamic_offset but in the opposite direction.
        dynamic_offset = int(self.PANEL_OFFSET_MAX * progress)

        # panel_x1 is the right edge of the panel content area.
        # When closed  : panel_x1 = frame_w + panel_w  (fully off-screen right)
        # When open    : panel_x1 = frame_w + dynamic_offset  (right corners hidden)
        panel_x1 = self.frame_w + self.panel_w - self._slide_x + dynamic_offset

        self._draw_panel(frame, panel_x1)

        if self._slide_x > self.panel_w * 0.3:
            alpha = min(1.0, (self._slide_x - self.panel_w * 0.3) /
                        (self.panel_w * 0.4))
            self._draw_toggles(frame, alpha)

    # ── SlideTab overrides — flip hit-testing to right edge ───────────────

    def _hit_handle(self, x: int, y: int) -> bool:
        hx1 = self.frame_w - self._slide_x
        hx0 = hx1 - self.HANDLE_W
        hy0 = self.anchor_y + (self.panel_h - self.HANDLE_H) // 2
        hy1 = hy0 + self.HANDLE_H
        return hx0 <= x <= hx1 and hy0 <= y <= hy1

    def _hit_panel(self, x: int, y: int) -> bool:
        px0 = self.frame_w - self._slide_x
        px1 = min(self.frame_w, px0 + self.panel_w)
        return px0 <= x <= px1 and self.anchor_y <= y <= self.anchor_y + self.panel_h

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw_panel(self, frame: np.ndarray, x1: int):
        """Right-side panel: right corners pushed off-screen."""
        x0_panel = x1 - self.panel_w
        x0_full  = x0_panel - self.HANDLE_W    
        y0 = self.anchor_y
        y1 = y0 + self.panel_h

        h, w = frame.shape[:2]
        fx0 = max(0, x0_full);  fy0 = max(0, y0)
        fx1 = min(w, x1);       fy1 = min(h, y1)

        if fx0 >= fx1 or fy0 >= fy1:
            return

        roi_w, roi_h = fx1 - fx0, fy1 - fy0
        overlay_roi  = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)

        local_x0 = x0_full - fx0
        local_y0 = y0      - fy0
        local_x1 = x1      - fx0
        local_y1 = y1      - fy0

        # Fill — all four corners rounded (right ones go off-screen naturally)
        self._filled_rounded_rect(overlay_roi,
                                  (local_x0, local_y0), (local_x1, local_y1),
                                  self.PANEL_CORNER, self.PANEL_COLOR,
                                  corners=(True, True, True, True))

        # Mask — only left corners rounded visibly; right side clipped by frame edge
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        self._filled_rounded_rect(mask,
                                  (local_x0, local_y0), (local_x1, local_y1),
                                  self.PANEL_CORNER, 255,
                                  corners=(True, False, True, False))

        mask_3    = cv2.merge([mask, mask, mask]) / 255.0
        roi_frame = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        overlay   = overlay_roi.astype(np.float32)

        frame[fy0:fy1, fx0:fx1] = (
            overlay   * mask_3 * self.PANEL_BG_ALPHA +
            roi_frame * (1 - mask_3 * self.PANEL_BG_ALPHA)
        ).astype(np.uint8)

        # Border — left corners rounded, right side runs off-screen
        self._rounded_rect_stroke(frame,
                                  (x0_full, y0), (x1, y1),
                                  self.PANEL_CORNER, self.BORDER_COLOR,
                                  self.BORDER_WIDTH,
                                  corners=(True, True, True, True))

        # Arrow icon on left edge of handle
        if self.panel_arrow is not None:
            self._blit_icon(frame, self.panel_arrow,
                            cx=x0_full + 6, cy=y0 + self.panel_h // 2, size=24)

        # Tab icon — visible when closed
        if self.icon is not None and self._slide_x < self.panel_w * 0.5:
            icon_x = (self.frame_w - self._slide_x) - self.HANDLE_W // 2 + 7
            self._blit_icon(frame, self.icon,
                            cx=icon_x, cy=y0 + self.panel_h // 2, size=18)

        # Header label + divider fade in as panel opens
        if self._slide_x > self.panel_w * 0.4:
            alpha = min(1.0, (self._slide_x - self.panel_w * 0.4) /
                        (self.panel_w * 0.3))
            self._draw_header(frame, x0_panel, x1, y0, alpha)

    def _draw_header(self, frame, x0_panel, x1, y0, alpha):
        """'View' label centred in the panel header, shifted left by CONTENT_SHIFT."""
        label_y = y0 + self.LABEL_Y_OFFSET

        roi_y0 = max(0, y0)
        roi_y1 = min(frame.shape[0], y0 + self.PADDING_TOP)
        # Shift the ROI leftward so the label renders left of centre
        roi_x0 = max(0, x0_panel - self.CONTENT_SHIFT)
        roi_x1 = min(frame.shape[1], x1 - self.CONTENT_SHIFT)

        if roi_x0 >= roi_x1 or roi_y0 >= roi_y1:
            return

        roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]
        img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        local_cx = (roi_x1 - roi_x0) // 2
        local_cy = label_y - roi_y0
        draw.text((local_cx, local_cy), self.label,
                  font=INTER_FONT, fill=(255, 255, 255), anchor="mm")
        blended = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        frame[roi_y0:roi_y1, roi_x0:roi_x1] = cv2.addWeighted(
            blended, alpha, frame[roi_y0:roi_y1, roi_x0:roi_x1], 1 - alpha, 0)

    def _draw_toggles(self, frame: np.ndarray, alpha: float):
        """Draw label + toggle switch for each row, shifted left by CONTENT_SHIFT."""
        # Panel left edge in screen space, shifted left
        panel_x0 = (self.frame_w - self._slide_x) + self.PADDING_SIDE - self.CONTENT_SHIFT
        panel_x1 = (self.frame_w - self._slide_x) + self.panel_w - self.PADDING_SIDE - self.CONTENT_SHIFT

        tw = ToggleRow.TOGGLE_W
        th = ToggleRow.TOGGLE_H

        for i, row in enumerate(self._rows):
            ry_center = (self.anchor_y + self.PADDING_TOP
                         + i * (self.ROW_H + self.ROW_GAP)
                         + self.ROW_H // 2)

            # Toggle switch — right-aligned within the shifted panel area
            tx1 = int(panel_x1)
            tx0 = tx1 - tw
            ty0 = ry_center - th // 2
            ty1 = ty0 + th

            row.toggle_rect = (tx0, ty0, tx1, ty1)

            # Track
            track_color = ToggleRow.COLOR_ON if row.value else ToggleRow.COLOR_OFF
            track_color_faded = tuple(int(c * alpha) for c in track_color)

            ftx0 = max(0, tx0); fty0 = max(0, ty0)
            ftx1 = min(frame.shape[1], tx1); fty1 = min(frame.shape[0], ty1)
            if ftx0 < ftx1 and fty0 < fty1:
                r = th // 2
                self._filled_rounded_rect(
                    frame, (ftx0, fty0), (ftx1, fty1),
                    r, track_color_faded,
                    corners=(True, True, True, True)
                )

            # Knob
            kr = ToggleRow.KNOB_R
            knob_cx = (tx1 - kr - 2) if row.value else (tx0 + kr + 2)
            knob_cy = ry_center
            knob_color = tuple(int(c * alpha) for c in ToggleRow.KNOB_COLOR)
            if 0 <= knob_cx < frame.shape[1] and 0 <= knob_cy < frame.shape[0]:
                cv2.circle(frame, (int(knob_cx), int(knob_cy)),
                           kr, knob_color, -1, cv2.LINE_AA)

            # Row label — left-aligned, stops before toggle
            label_x = int(panel_x0)
            roi_y0  = max(0, ry_center - self.ROW_H // 2)
            roi_y1  = min(frame.shape[0], ry_center + self.ROW_H // 2)
            roi_x0  = max(0, label_x)
            roi_x1  = min(frame.shape[1], tx0 - 4)

            if roi_x0 < roi_x1 and roi_y0 < roi_y1:
                roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]
                img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)
                local_y = (roi_y1 - roi_y0) // 2
                draw.text((0, local_y), row.label,
                          font=INTER_FONT, fill=(255, 255, 255), anchor="lm")
                blended = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                frame[roi_y0:roi_y1, roi_x0:roi_x1] = cv2.addWeighted(
                    blended, alpha,
                    frame[roi_y0:roi_y1, roi_x0:roi_x1], 1 - alpha, 0)

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_rows(self) -> list[ToggleRow]:
        return [
            ToggleRow("atom_labels",         "Atom Labels",         default=True,  on_change=self._on_toggle),
            ToggleRow("live_degree",         "Live Degree",         default=True,  on_change=self._on_toggle),
            ToggleRow("electron_clouds",     "Electron Clouds",     default=False, on_change=self._on_toggle),
            ToggleRow("lone_pairs",          "Lone Pairs",          default=False, on_change=self._on_toggle),
            ToggleRow("hybridization_state", "Hybridization State", default=False, on_change=self._on_toggle),
        ]

    def _calc_content_h(self, n_rows: int) -> int:
        """Height of the toggle content area below the header."""
        return (n_rows * self.ROW_H
                + (n_rows - 1) * self.ROW_GAP
                + self.PADDING_BOT)

    def _on_toggle(self, toggle_id: str, value: bool):
        print(f"[VisibilityTab] {toggle_id} → {value}")
        # Hook into venv here, e.g.: venv.set_visibility(toggle_id, value)