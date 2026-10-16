"""Copy standard Piper STL assets and compile its URDF to a standalone MJCF."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent


def prepare(source: Path, destination: Path):
    robot = source / "agx_arm_description/agx_arm_urdf/piper"
    urdf = robot / "urdf/piper_with_gripper_description.urdf"
    destination.mkdir(parents=True, exist_ok=True)
    mesh_dir = destination / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    tree = ET.parse(urdf)
    manifest = {}
    # MuJoCo accepts STL, not the source DAE visual meshes. Both source
    # geometries use identical link coordinates; reuse the collision STL.
    for mesh in tree.findall(".//mesh"):
        name = Path(mesh.attrib["filename"]).stem + ".stl"
        src = robot / "meshes" / name
        shutil.copy2(src, mesh_dir / name)
        mesh.set("filename", str(mesh_dir / name))
        manifest[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    root = tree.getroot()
    # Writing holds an 8mm pen. Freeze the two finger slides at +/-4mm,
    # preserving the source joint-frame transforms and mimic convention.
    # The six arm joints remain the only learned/controlled DOFs.
    for joint in root.findall("joint"):
        if joint.attrib["name"] not in ("gripper", "gripper_joint1", "gripper_joint2"):
            continue
        displacement = {"gripper": 0., "gripper_joint1": .004, "gripper_joint2": -.004}[joint.attrib["name"]]
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib["xyz"], sep=" ")
        rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
        axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
        xyz += Rotation.from_euler("xyz", rpy).apply(axis*displacement)
        origin.set("xyz", " ".join(map(str, xyz)))
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "mimic"):
            element = joint.find(tag)
            if element is not None:
                joint.remove(element)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", discardvisual="true", fusestatic="false", balanceinertia="true", strippath="false")
    compiled_urdf = destination / "piper_source.urdf"
    tree.write(compiled_urdf, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(compiled_urdf))
    xml = destination / "piper.xml"
    mujoco.mj_saveLastXML(str(xml), model)
    tree = ET.parse(xml)
    root = tree.getroot()
    root.find("compiler").set("meshdir", "meshes")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", Path(mesh.attrib["file"]).name)
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.2", armature="0.02")
    # Initial writing benchmark: ideal gravity compensation, visual geometry
    # only. These assumptions are not a calibrated hardware/contact model.
    for body in root.findall(".//body"):
        body.set("gravcomp", "1")
    for geom in root.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("rgba", "0.65 0.68 0.73 1")
    gripper = root.find(".//body[@name='gripper_base']")
    # gripper_base is 4.5mm beyond link6. TCP is 180mm from link6,
    # 37.5mm beyond the finger joint origin at gripper_base z=138mm.
    ET.SubElement(gripper, "geom", name="pen", type="capsule", fromto="0 0 0.09 0 0 0.1755",
                  size="0.004", mass="0.02", contype="0", conaffinity="0", rgba="0.15 0.2 0.8 1")
    ET.SubElement(gripper, "site", name="pen_tip", pos="0 0 0.1755", size="0.002", rgba="0.05 0.05 0.05 1")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
    ET.SubElement(world, "geom", name="paper", type="box", pos="0.30 0 -0.002",
                  size="0.13 0.13 0.002", contype="0", conaffinity="0", rgba="0.96 0.96 0.96 1")
    actuator = ET.SubElement(root, "actuator")
    for i in range(1, 7):
        ET.SubElement(actuator, "velocity", name=f"velocity{i}", joint=f"joint{i}",
                      kv="30", ctrllimited="true", ctrlrange="-0.5 0.5",
                      forcelimited="true", forcerange="-20 20")
    ET.indent(tree)
    tree.write(xml, encoding="unicode")
    # Keep the copied URDF relocatable too, after the importer has used it.
    portable = ET.parse(compiled_urdf)
    for mesh in portable.findall(".//mesh"):
        mesh.set("filename", "meshes/" + Path(mesh.attrib["filename"]).name)
    portable.write(compiled_urdf, encoding="unicode")
    license_path = robot.parent / "LICENSE"
    if license_path.exists():
        shutil.copy2(license_path, destination / "SOURCE_LICENSE")
    (destination / "source_manifest.json").write_text(json.dumps({
        "source_urdf": str(urdf), "meshes_sha256": manifest,
        "assumptions": "standard Piper with fixed 8mm gripper opening, pen TCP 0.18m from link6, ideal gravity compensation, no collision/contact",
    }, indent=2))
    checked = mujoco.MjModel.from_xml_path(str(xml))
    assert checked.nq == checked.nv == checked.nu == 6
    print(f"Prepared {xml}: nq={checked.nq}, nv={checked.nv}, nu={checked.nu}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/cym/ROS/Arms/agx_arm_sim"))
    parser.add_argument("--destination", type=Path, default=HERE / "assets")
    args = parser.parse_args()
    prepare(args.source.resolve(), args.destination.resolve())
