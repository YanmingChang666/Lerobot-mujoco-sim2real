"""Shared viewer/offscreen overlay: red reference, green measured tip trail, black ink."""
import mujoco
import numpy as np

REFERENCE_COLOR = np.array([1., .12, .08, 1.])
ACTUAL_COLOR = np.array([.05, .9, .15, 1.])
INK_COLOR = np.array([.05, .05, .05, 1.])
PAPER_COLOR = np.array([.97, .97, .94, 1.])


def _z_rotation_mat(rotation_deg):
    angle = np.radians(rotation_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=float)


def draw_paper_sheet(scene, origin_xy, rotation_deg, half_x, half_y, z, *,
                     thickness=.002, color=None, append=True):
    """Draw a finished paper copy (white box) in the viewer overlay."""
    if not append:
        scene.ngeom = 0
    if scene.ngeom >= scene.maxgeom:
        return
    color = PAPER_COLOR if color is None else np.asarray(color, dtype=float)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_BOX,
        np.array([half_x, half_y, thickness], dtype=float),
        np.array([origin_xy[0], origin_xy[1], z], dtype=float),
        _z_rotation_mat(rotation_deg).ravel(), color)
    scene.ngeom += 1


def draw_trajectories(scene, reference, actual, pen_down, *, append=False):
    """Show the entire path, including lighter pen-up segments.

    Co-located red dots are wider than the green measured line so accurate
    tracking does not hide the reference. No artificial coordinate offsets.
    Bounded sampling prevents the trail filling the viewer geometry buffer.
    """
    if not append:
        scene.ngeom = 0
    budget = scene.maxgeom-scene.ngeom
    if budget < 4:
        return
    # Reserve 1/3 for sparse red reference dots, 2/3 for the green trail.
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))]
    nref = min(len(reference), max(2, budget//3), max(2, int(distance[-1]/.005)+1), 450)
    # Spatial spacing avoids dense dots hiding the green line at slow corners.
    ref_indices = np.unique(np.searchsorted(distance, np.linspace(0, distance[-1], nref)))
    for index in ref_indices:
        color = REFERENCE_COLOR.copy()
        color[3] = 1 if pen_down[index] else .35
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([.0015]*3), reference[index], np.eye(3).ravel(), color)
        scene.ngeom += 1
    actual = np.asarray(actual)
    if len(actual) < 2:
        return
    count = min(len(actual), budget-nref, 1200)
    indices = np.unique(np.linspace(0, len(actual)-1, count).astype(int))
    for left, right in zip(indices[:-1], indices[1:]):
        if np.linalg.norm(actual[right]-actual[left]) < 1e-9:
            continue
        color = ACTUAL_COLOR.copy()
        color[3] = 1 if np.all(pen_down[left+1:right+1]) else .35
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                           np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0008,
                            actual[left], actual[right])
        scene.ngeom += 1


def draw_ink(scene, polylines_world, *, color=None, width=.0012, append=True, max_segments=2000):
    """Draw finished pen-down strokes (black ink on paper). Each polyline is (N, 3)."""
    if not append:
        scene.ngeom = 0
    color = INK_COLOR if color is None else np.asarray(color, dtype=float)
    budget = scene.maxgeom - scene.ngeom
    if budget < 1:
        return
    segments = []
    for line in polylines_world:
        pts = np.asarray(line, dtype=float)
        if len(pts) < 2:
            continue
        for left, right in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(right - left) < 1e-9:
                continue
            segments.append((left, right))
    if not segments:
        return
    if len(segments) > min(budget, max_segments):
        pick = np.unique(np.linspace(0, len(segments) - 1, min(budget, max_segments)).astype(int))
        segments = [segments[i] for i in pick]
    for left, right in segments[:budget]:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                           np.zeros(3), np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, left, right)
        scene.ngeom += 1
