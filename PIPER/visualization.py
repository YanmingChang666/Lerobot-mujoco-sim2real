"""Shared viewer/offscreen overlay: red reference, green measured tip trail."""
import mujoco
import numpy as np

REFERENCE_COLOR = np.array([1., .12, .08, 1.])
ACTUAL_COLOR = np.array([.05, .9, .15, 1.])


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
