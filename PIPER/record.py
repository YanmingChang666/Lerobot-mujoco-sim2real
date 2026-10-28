"""Offscreen MuJoCo recording of a writing run to MP4 (works with --headless)."""
from pathlib import Path

import mujoco
import numpy as np

from .env import DT


class WritingRecorder:
    """Capture RGB frames from a free camera and encode an MP4 with imageio."""

    def __init__(self, model, path, *, width=640, height=480, fps=25.,
                 lookat=(0., 0., 0.), distance=1.2, azimuth=90., elevation=-45.):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("--record-fps must be finite and positive")
        self.width = int(width)
        self.height = int(height)
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = lookat
        self.camera.distance = float(distance)
        self.camera.azimuth = float(azimuth)
        self.camera.elevation = float(elevation)
        # Write every N sim steps so wall-clock video ≈ sim time at fps.
        self.stride = max(1, int(round(1. / (self.fps * DT))))
        self._step = 0
        self._frames = 0
        import imageio
        self._writer = imageio.get_writer(
            str(self.path), fps=self.fps, macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"])

    @property
    def scene(self):
        return self.renderer.scene

    def update(self, data):
        self.renderer.update_scene(data, camera=self.camera)

    def write_frame(self):
        frame = self.renderer.render()
        self._writer.append_data(np.asarray(frame))
        self._frames += 1

    def capture(self, data, overlay=None, *, force=False):
        """Render one frame. overlay(scene) may append ink/trails after update_scene."""
        self._step += 1
        if not force and (self._step - 1) % self.stride != 0:
            return
        self.update(data)
        if overlay is not None:
            overlay(self.scene)
        self.write_frame()

    def close(self):
        try:
            self._writer.close()
        finally:
            self.renderer.close()
        print(f"Saved recording ({self._frames} frames @ {self.fps:.1f} fps, "
              f"stride {self.stride}): {self.path}", flush=True)


def resolve_record_path(args, *, default_stem):
    """args.record is None (off), '' (auto path), or an explicit path string/Path."""
    if args.record is None:
        return None
    if args.record == "" or args.record is True:
        return Path(args.output) / f"{default_stem}_record.mp4"
    return Path(args.record)
