"""Minimal HDRI skybox compatibility layer.

This project currently imports `HDRISkybox` from `hdri_skybox.py`.
If the full skybox implementation is unavailable, this fallback keeps
application startup working while preserving the same public API.
"""


class HDRISkybox:
    """Fallback skybox that intentionally renders nothing."""

    def __init__(self, hdri_path, sphere_subdivisions=32):
        self.hdri_path = hdri_path
        self.sphere_subdivisions = int(sphere_subdivisions)
        self._enabled = True

    def render(self, rotate_x=0.0, rotate_y=0.0, scale=50.0):
        # No-op fallback: keep scene rendering functional without HDRI assets.
        return None

    def cleanup(self):
        return None
