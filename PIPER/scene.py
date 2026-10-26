"""Shared look for every writing robot: blue gradient sky and a blue grid floor.

Applied when a model is loaded, so the robot XML files stay unchanged (their
SHA256 is what checkpoints and training references are checked against).
Only visual elements are added; the floor has no collisions, so the simulated
motion is identical with or without it.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

FLOOR = "writing_floor"  # name of the floor added to models that have none


def _styled_xml(xml_path):
    """Robot XML text with the background added, and whether a floor was added."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Absolute asset directories, so the edited text compiles from memory with
    # exactly the same mesh/texture files as the original file.
    compilers = root.findall("compiler") or [ET.SubElement(root, "compiler")]
    base = Path(xml_path).resolve().parent
    for compiler in compilers:
        assetdir = compiler.attrib.pop("assetdir", None)
        for key in ("meshdir", "texturedir"):
            value = compiler.get(key, assetdir if assetdir is not None else "")
            compiler.set(key, str((base / value).resolve()) + "/")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    if visual.find("rgba") is None:
        ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    if visual.find("headlight") is None:
        ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="writing_sky", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, "texture", name="writing_grid", type="2d", builtin="checker", mark="edge",
                  rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
    ET.SubElement(asset, "material", name="writing_grid", texture="writing_grid", texuniform="true",
                  texrepeat="5 5", reflectance="0.2")
    worldbody = root.find("worldbody")
    planes = [g for g in worldbody.iter("geom") if g.get("type") == "plane"]
    for plane in planes:  # restyle an existing floor in place (position unchanged)
        plane.attrib.pop("rgba", None)
        plane.set("material", "writing_grid")
    if not planes:
        ET.SubElement(worldbody, "geom", name=FLOOR, type="plane", size="0 0 0.05", material="writing_grid",
                      contype="0", conaffinity="0", group="0")
    return ET.tostring(root, encoding="unicode"), not planes


def load_model(xml_path):
    """Compile the robot with the shared background -> (model, lowest robot Z or None).

    Compiled from memory; nothing is written to disk. The second value is set
    only when a floor was added (pass it to place_floor when the paper moves).
    """
    text, added_floor = _styled_xml(Path(xml_path))
    model = mujoco.MjModel.from_xml_string(text)
    lowest = _lowest_point(model) if added_floor else None
    if added_floor:
        place_floor(model, lowest)
    return model, lowest


def _lowest_point(model):
    """Lowest world Z of the robot's geometry at the model's initial pose."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lowest = np.inf
    for i in range(model.ngeom):
        name = model.geom(i).name
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE or name in ("paper", "pen"):
            continue
        rotation, position = data.geom_xmat[i].reshape(3, 3), data.geom_xpos[i]
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[i]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[start:start + count]
        else:
            center, half = model.geom_aabb[i, :3], model.geom_aabb[i, 3:]
            points = center + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        lowest = min(lowest, float((points @ rotation.T)[:, 2].min() + position[2]))
    return lowest


def place_floor(model, lowest, paper_height=None):
    """Put an added floor under both the robot and the paper (models with their own floor are left alone)."""
    if lowest is None:
        return
    if paper_height is None:
        paper_height = float(model.geom("paper").pos[2]) + .002
    model.geom(FLOOR).pos[2] = min(paper_height, lowest) - .004
