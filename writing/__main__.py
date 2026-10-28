"""Unified writing entry point.

    python -m writing --list
    python -m writing --robot x02 --text 永 --hold
    python -m writing --robot g1_right --text HELLO --headless

Every option after --robot is the same as python -m X02.write --help.
"""
import argparse
import os
import sys

# Offscreen recording / headless needs EGL (or OSMesa) before MuJoCo loads GL.
if "MUJOCO_GL" not in os.environ and ("--headless" in sys.argv or "--record" in sys.argv):
    os.environ["MUJOCO_GL"] = "egl"

from .robot import ROOT, all_robots, load_robot


def list_robots():
    rows = []
    for spec in all_robots():
        try:
            dof, mode = str(spec.dof), spec.resolved_ik_mode
        except (FileNotFoundError, ValueError) as error:
            dof, mode = "?", f"(model missing: {error.__class__.__name__})"
        rows.append((spec.name, spec.train_env, dof, mode, spec.description))
    widths = [max(len(r[i]) for r in rows + [("robot", "train --env", "dof", "ik mode", "")]) for i in range(4)]
    print("  ".join(h.ljust(w) for h, w in zip(("robot", "train --env", "dof", "ik mode"), widths)), " description")
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row[:4], widths)), "", row[4])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(prog="python -m writing", add_help=False)
    pre.add_argument("--robot")
    pre.add_argument("--list", action="store_true")
    known, rest = pre.parse_known_args(argv)
    if known.list:
        list_robots()
        return
    if not known.robot:
        print(__doc__.strip())
        print()
        list_robots()
        if not {"-h", "--help"} & set(rest):
            sys.exit("\nerror: --robot is required")
        return
    spec = load_robot(known.robot)
    if not spec.model.exists():
        sys.exit(f"error: {spec.name} model not found: {spec.model}\n"
                 f"Generate it first (once per machine): python -m writing.build_model --robot {spec.name}")
    if spec.ik_mode != "auto" or spec.dof < 6:
        print(f"{spec.name}: {spec.dof} joints, IK mode {spec.resolved_ik_mode}", flush=True)
    from PIPER.write import main as run_writing
    run_writing(env_factory=spec.env_class(), planner=spec.planner(), here=spec.model.parent,
                reference=spec.reference, model_file=spec.model, center=spec.center,
                height=spec.height, home_deg=spec.home, metadata_name=spec.metadata_name,
                camera_distance=spec.camera_distance, camera_height=spec.camera_height,
                output_dir=spec.output, hanzi_data=ROOT.parent / "hanzi-writer-data/data",
                argv=rest, prog=f"python -m writing --robot {spec.name}", robot_name=spec.name,
                rotation_deg=spec.rotation_deg, pen_yaw_deg=spec.pen_yaw_deg,
                approach_seconds=spec.approach_seconds)


if __name__ == "__main__":
    main()
