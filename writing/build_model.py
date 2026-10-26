"""URDF -> writing-ready MJCF, driven by the `build:` section of robots/<name>.yaml.

    python -m writing.build_model --inspect path/to/robot.urdf   # list joints/links to fill in the YAML
    python -m writing.build_model --robot g1_right               # write the model named by `model:`

What the generated model contains (the conventions every writing robot follows):
  * the joints listed in `joints:` are the only moving joints; every other joint is
    welded at 0 (or at the angle given in build.fixed_joints), base fixed in the world
  * one <velocity> actuator per writing joint, in the same order (ctrl = rad/s)
  * gravity compensation on every body, no collisions (contact is not simulated)
  * a pen capsule and a `pen_tip` site whose +Z points along the pen, plus a `paper` box
"""
import argparse
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from PIPER.env import MAX_SPEED
from .robot import _path, load_robot

MESH_SUFFIXES = {".stl", ".obj", ".msh"}
BUILD_KEYS = {"urdf", "package_dirs", "copy_meshes", "skip_missing_meshes", "base_pos", "fixed_joints", "pen",
              "kv", "default_force", "damping", "armature", "floor_height"}
PEN_KEYS = {"link", "mount", "direction", "length"}


# ---------------------------------------------------------------- URDF helpers
def package_candidates(name, start, package_dirs):
    """Possible roots of ROS package `name`, most likely first."""
    if name in package_dirs:
        yield _path(package_dirs[name])
        return
    for directory in [start, *start.parents]:
        manifest = directory / "package.xml"
        if manifest.exists():
            tag = ET.parse(manifest).getroot().find("name")
            if tag is not None and tag.text and tag.text.strip() == name:
                yield directory
        if directory.name == name:
            yield directory
        if (directory / name).is_dir():
            yield directory / name


def find_package_file(name, relative, start, package_dirs):
    """First candidate package root that actually contains `relative` (a same-named
    folder elsewhere, e.g. urdf/<name>/ with USD files, is skipped)."""
    candidates = list(dict.fromkeys(package_candidates(name, start, package_dirs)))
    for root in candidates:
        path = root / relative
        if path.exists() or any(path.with_suffix(s).exists() for s in (".stl", ".STL", ".obj", ".OBJ")):
            return path
    if not candidates:
        raise FileNotFoundError(f"Cannot locate ROS package '{name}' above {start}; "
                                f"set build.package_dirs: {{{name}: /path/to/{name}}}")
    raise FileNotFoundError(f"Mesh '{relative}' not found in package '{name}' (looked in "
                            f"{', '.join(map(str, candidates))}); set build.package_dirs if it is elsewhere")


def resolve_mesh(filename, urdf_dir, package_dirs):
    """Return a MuJoCo-loadable mesh path, or None if only an unsupported format exists."""
    if filename.startswith("package://"):
        package, _, relative = filename[len("package://"):].partition("/")
        path = find_package_file(package, relative, urdf_dir, package_dirs)
    elif filename.startswith("file://"):
        path = Path(filename[len("file://"):])
    else:
        path = Path(filename)
        path = path if path.is_absolute() else urdf_dir / path
    if path.suffix.lower() not in MESH_SUFFIXES:
        # MuJoCo cannot read DAE/GLB etc.; use an STL/OBJ with the same stem if present.
        for suffix in (".stl", ".STL", ".obj", ".OBJ"):
            if path.with_suffix(suffix).exists():
                return path.with_suffix(suffix)
        return None
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {filename} -> {path}")
    return path


def inspect_urdf(urdf):
    root = ET.parse(urdf).getroot()
    print(f"Robot: {root.get('name')}   ({urdf})")
    print("Movable joints (copy the names you want into `joints:`, base -> tip order):")
    for joint in root.findall("joint"):
        kind = joint.get("type")
        if kind == "fixed":
            continue
        limit = joint.find("limit")
        span = ""
        if limit is not None and limit.get("lower") is not None:
            span = f"[{np.degrees(float(limit.get('lower'))):7.1f}, {np.degrees(float(limit.get('upper'))):7.1f}] deg"
        print(f"  {joint.get('name'):34s} {kind:10s} {joint.find('parent').get('link')} -> "
              f"{joint.find('child').get('link')}  {span}")
    children = {j.find("parent").get("link") for j in root.findall("joint")}
    leaves = [l.get("name") for l in root.findall("link") if l.get("name") not in children]
    print("Leaf links (candidates for build.pen.link):", ", ".join(leaves))


def prepare_urdf(urdf, mesh_dir, package_dirs, copy_meshes, skip_missing=False, simplified_dir=None):
    """Rewrite mesh paths and drop duplicate collision geometry; return the edited tree."""
    tree = ET.parse(urdf)
    root = tree.getroot()
    used = {}
    for link in root.findall("link"):
        if link.findall("visual"):
            for collision in link.findall("collision"):
                link.remove(collision)
        for number, element in enumerate(link.findall("visual") + link.findall("collision")):
            if not element.get("name"):
                # Unnamed elements make MuJoCo print a duplicate-name warning per geom.
                element.set("name", f"{link.get('name')}_{element.tag}{number}")
            mesh = element.find("geometry/mesh")
            if mesh is None:
                continue
            try:
                source = resolve_mesh(mesh.get("filename"), urdf.parent, package_dirs)
            except FileNotFoundError as error:
                if not skip_missing:
                    raise FileNotFoundError(f"{error}\n(or set build.skip_missing_meshes: true "
                                            "to build without this part's appearance)") from None
                print(f"  warning: link '{link.get('name')}' mesh {mesh.get('filename')} not found; "
                      "shown without it")
                link.remove(element)
                continue
            if source is None:
                print(f"  warning: link '{link.get('name')}' mesh {mesh.get('filename')} has no STL/OBJ; skipped")
                link.remove(element)
                continue
            if not copy_meshes:
                # Reference the source mesh in place (made relative after import);
                # only meshes over MuJoCo's face limit get a simplified copy.
                if (binary_stl_faces(source) or 0) > MAX_FACES:
                    simplified_dir.mkdir(parents=True, exist_ok=True)
                    target = simplified_dir / source.name
                    copy_mesh(source, target)
                    source = target
                mesh.set("filename", str(source.resolve()))
                continue
            key = source.resolve()
            if key not in used:
                name = source.name
                if name in used.values():
                    name = f"{len(used)}_{name}"
                used[key] = name
                copy_mesh(source, mesh_dir / name)
            mesh.set("filename", used[key])
    for old in root.findall("mujoco"):
        root.remove(old)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", meshdir=str(mesh_dir), strippath="false", discardvisual="false",
                  fusestatic="false", balanceinertia="true", boundmass="1e-4", boundinertia="1e-7")
    return tree


# ---------------------------------------------------------------- large meshes
MAX_FACES = 200000  # MuJoCo's per-mesh STL limit
_STL_FACE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def binary_stl_faces(path):
    """Triangle count of a binary STL, or None for other formats (ASCII STL, OBJ)."""
    if path.suffix.lower() != ".stl" or path.stat().st_size < 84:
        return None
    with open(path, "rb") as f:
        f.seek(80)
        count = int.from_bytes(f.read(4), "little")
    return count if path.stat().st_size == 84 + 50 * count else None


def simplify_stl(source, destination, max_faces=MAX_FACES):
    """Vertex-clustering decimation for display meshes: merge vertices on a grid that
    grows until the mesh has at most max_faces triangles. Returns (before, after)."""
    triangles = np.fromfile(source, dtype=_STL_FACE, offset=84)["vertices"].astype(np.float64)
    vertices = triangles.reshape(-1, 3)
    cell = np.ptp(vertices, axis=0).max() / 1024
    while True:
        keys, inverse = np.unique(np.floor(vertices / cell).astype(np.int64), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        centers = np.stack([np.bincount(inverse, vertices[:, k]) for k in range(3)], 1)
        centers /= np.bincount(inverse)[:, None]
        faces = inverse.reshape(-1, 3)
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
        faces = faces[np.unique(np.sort(faces, axis=1), axis=0, return_index=True)[1]]
        if len(faces) <= max_faces:
            break
        cell *= 1.3
    out = np.zeros(len(faces), dtype=_STL_FACE)
    out["vertices"] = centers[faces]
    with open(destination, "wb") as f:
        f.write(b"simplified by writing.build_model".ljust(80, b" "))
        f.write(len(faces).to_bytes(4, "little"))
        out.tofile(f)
    return len(triangles), len(faces)


def copy_mesh(source, destination):
    faces = binary_stl_faces(source)
    if faces is not None and faces > MAX_FACES:
        before, after = simplify_stl(source, destination)
        print(f"  {source.name}: {before} triangles > MuJoCo limit {MAX_FACES}; simplified to {after} (display only)")
    else:
        shutil.copy2(source, destination)


# ---------------------------------------------------------------- MJCF edits
def _quat_xyzw(body):
    if any(body.get(k) is not None for k in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError(f"Body '{body.get('name')}' uses an unsupported orientation attribute")
    w, x, y, z = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
    return np.array([x, y, z, w])


def bake_joint(body, joint, value):
    """Weld a hinge/slide joint at `value` (rad or m) by moving the child body frame."""
    axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
    axis = axis / np.linalg.norm(axis)
    anchor = np.fromstring(joint.get("pos", "0 0 0"), sep=" ")
    pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
    rotation = Rotation.from_quat(_quat_xyzw(body))
    if joint.get("type", "hinge") == "slide":
        pos = pos + rotation.apply(axis * value)
    else:
        turn = Rotation.from_rotvec(axis * value)
        pos = pos + rotation.apply(anchor - turn.apply(anchor))
        rotation = rotation * turn
    x, y, z, w = rotation.as_quat()
    body.set("pos", " ".join(f"{v:.9g}" for v in pos))
    body.set("quat", " ".join(f"{v:.9g}" for v in (w, x, y, z)))


def pen_site_quat(direction):
    """Site rotation whose +Z is the pen direction (tip side)."""
    if np.allclose(direction, [0, 0, -1]):
        return "0 0 1 0"  # half turn about Y, the convention of the existing models
    rotation, _ = Rotation.align_vectors([direction], [[0, 0, 1]])
    x, y, z, w = rotation.as_quat()
    return f"{w:.9g} {x:.9g} {y:.9g} {z:.9g}"


def build(spec, log=print):
    cfg = spec.build
    if not cfg:
        raise ValueError(f"{spec.source.name} has no `build:` section")
    unknown = set(cfg) - BUILD_KEYS
    unknown |= set(cfg.get("pen") or {}) - PEN_KEYS
    if unknown:
        raise ValueError(f"{spec.source.name}: unknown build field(s) {sorted(unknown)}")
    if not spec.joints:
        raise ValueError(f"{spec.source.name}: list the writing joints in `joints:` before building")
    pen = cfg.get("pen") or {}
    if "link" not in pen:
        raise ValueError(f"{spec.source.name}: build.pen.link is required (the link that holds the pen)")
    urdf = _path(cfg["urdf"])
    copy_meshes = cfg.get("copy_meshes", True)
    output = spec.model
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = output.parent / "meshes"
    if copy_meshes:
        mesh_dir.mkdir(exist_ok=True)

    source_joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    missing = [name for name in spec.joints if name not in source_joints]
    if missing:
        raise ValueError(f"Writing joint(s) {missing} not in {urdf.name}; "
                         f"run python -m writing.build_model --inspect {urdf}")
    fixed = dict(cfg.get("fixed_joints") or {})
    bad = [name for name in fixed if name not in source_joints or name in spec.joints]
    if bad:
        raise ValueError(f"build.fixed_joints {bad}: unknown, or also a writing joint")

    log(f"Importing {urdf}")
    with tempfile.TemporaryDirectory() as directory:
        work_mesh_dir = mesh_dir if copy_meshes else Path(directory)
        tree = prepare_urdf(urdf, work_mesh_dir, cfg.get("package_dirs") or {}, copy_meshes,
                            cfg.get("skip_missing_meshes", False), simplified_dir=mesh_dir)
        temp_urdf = Path(directory) / "robot.urdf"
        tree.write(temp_urdf, encoding="unicode")
        imported = mujoco.MjModel.from_xml_path(str(temp_urdf))
        temp_xml = Path(directory) / "imported.xml"
        mujoco.mj_saveLastXML(str(temp_xml), imported)
        tree = ET.parse(temp_xml)
    root = tree.getroot()
    root.set("model", f"{spec.name}_writing")
    compiler = root.find("compiler")
    compiler.set("angle", "radian")
    if copy_meshes:
        compiler.set("meshdir", "meshes")
    else:
        compiler.attrib.pop("meshdir", None)
        # Mesh paths relative to the model file, so the repository can be moved or cloned.
        for mesh in root.iter("mesh"):
            if mesh.get("file") and Path(mesh.get("file")).is_absolute():
                mesh.set("file", os.path.relpath(mesh.get("file"), output.parent))
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", ".002")
    option.set("integrator", "implicitfast")
    for tag in ("actuator", "keyframe", "sensor", "tendon"):
        for element in root.findall(tag):
            root.remove(element)

    parents = {child: parent for parent in root.iter() for child in parent}
    writing = {}
    for joint in list(root.iter("joint")):
        name, body = joint.get("name"), parents[joint]
        if name in spec.joints:
            writing[name] = joint
            continue
        kind = joint.get("type", "hinge")
        value = fixed.get(name, 0.)
        if value:
            if kind not in ("hinge", "slide"):
                raise ValueError(f"Cannot hold {kind} joint '{name}' at a nonzero value")
            bake_joint(body, joint, np.radians(value) if kind == "hinge" else value)
        body.remove(joint)
    for freejoint in list(root.iter("freejoint")):
        parents[freejoint].remove(freejoint)
    equality = root.find("equality")
    if equality is not None:
        for constraint in list(equality):
            if {constraint.get("joint1"), constraint.get("joint2")} - set(spec.joints) - {None}:
                equality.remove(constraint)
        if not len(equality):
            root.remove(equality)

    for name in spec.joints:
        joint = writing[name]
        if joint.get("type", "hinge") not in ("hinge", "slide"):
            raise ValueError(f"Writing joint '{name}' must be revolute/continuous/prismatic")
        if joint.get("range") is None or joint.get("limited") == "false":
            joint.set("range", f"{-2*np.pi:.6f} {2*np.pi:.6f}")  # continuous joint
        joint.set("limited", "true")
        joint.set("damping", str(cfg.get("damping", .2)))
        joint.set("armature", str(cfg.get("armature", .02)))
        joint.set("frictionloss", "0")
    for body in root.iter("body"):
        body.set("gravcomp", "1")
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    worldbody = root.find("worldbody")
    base = worldbody.find("body")
    if cfg.get("base_pos") is not None:
        base.set("pos", " ".join(map(str, cfg["base_pos"])))

    holder = next((b for b in root.iter("body") if b.get("name") == pen["link"]), None)
    if holder is None:
        raise ValueError(f"build.pen.link '{pen['link']}' is not a link of {urdf.name}")
    mount = np.asarray(pen.get("mount", [0, 0, 0]), dtype=float)
    direction = np.asarray(pen.get("direction", [0, 0, -1]), dtype=float)
    direction /= np.linalg.norm(direction)
    length = float(pen.get("length", .12))
    tip = mount + length * direction
    fmt = lambda v: " ".join(f"{x:.6g}" for x in v)
    ET.SubElement(holder, "geom", name="pen", type="capsule", fromto=fmt([*(mount - .02*direction), *tip]),
                  size=".003", mass=".01", rgba=".12 .12 .12 1", contype="0", conaffinity="0")
    ET.SubElement(holder, "site", name="pen_tip", pos=fmt(tip), quat=pen_site_quat(direction),
                  size=".003", rgba="0 0 1 1")
    ET.SubElement(worldbody, "light", pos="0 -1 3", dir="0 .3 -1", directional="true", castshadow="false")
    if cfg.get("floor_height") is not None:
        ET.SubElement(worldbody, "geom", name="floor", type="plane", size="3 3 .01",
                      pos=f"0 0 {cfg['floor_height']}", rgba=".8 .82 .85 1", contype="0", conaffinity="0")
    ET.SubElement(worldbody, "geom", name="paper", type="box", pos=fmt([*spec.center, spec.height - .002]),
                  size=".13 .13 .002", rgba=".97 .97 .94 1", contype="0", conaffinity="0")

    actuator = ET.SubElement(root, "actuator")
    for name in spec.joints:
        limit = source_joints[name].find("limit")
        effort = float(limit.get("effort", 0) or 0) if limit is not None else 0.
        effort = effort if effort > 0 else float(cfg.get("default_force", 40))
        ET.SubElement(actuator, "velocity", name=name, joint=name, kv=str(cfg.get("kv", 60)),
                      ctrlrange=f"{-MAX_SPEED} {MAX_SPEED}", forcerange=f"{-effort:g} {effort:g}")
    zeros = " ".join(["0"] * len(spec.joints))
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home", qpos=zeros, ctrl=zeros)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=True)
    return output


def summarize(spec, log=print):
    env = spec.env_class()()
    state = env.reset(np.radians(spec.home))
    m = env.model
    log(f"Wrote {spec.model}")
    log(f"  {len(spec.joints)} writing joints, {m.nbody - 1} bodies, {m.nmesh} meshes, IK mode {spec.resolved_ik_mode}")
    for name, (low, high) in zip(env.joint_names, env.limits):
        log(f"    {name:34s} [{np.degrees(low):7.1f}, {np.degrees(high):7.1f}] deg")
    first = m.body(m.jnt_bodyid[env.model.joint(spec.joints[0]).id])
    shoulder = env.data.xpos[first.id]
    axis = env.data.site_xmat[env.site].reshape(3, 3)[:, 2]
    log(f"  first joint body '{first.name}' at {np.round(shoulder, 3).tolist()} m")
    log(f"  pen tip at home pose {np.round(state[:3], 3).tolist()} m, pen axis {np.round(axis, 3).tolist()}")
    log(f"  paper (write_defaults) center {list(spec.center)} height {spec.height} m; "
        f"horizontal distance from first joint {np.hypot(*(np.array(spec.center) - shoulder[:2])):.3f} m")
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m writing.build_model", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--robot", help="robot name (robots/<name>.yaml with a build: section)")
    group.add_argument("--inspect", type=Path, help="print the joints and links of a URDF")
    args = parser.parse_args(argv)
    if args.inspect:
        inspect_urdf(args.inspect)
        return
    spec = load_robot(args.robot)
    build(spec)
    summarize(spec)


if __name__ == "__main__":
    main()
