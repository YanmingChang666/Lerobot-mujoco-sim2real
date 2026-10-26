"""Headless tests for the generic writing layer: python -m unittest writing.test_writing -v

Robots whose generated model is missing (python -m writing.build_model --robot <name>)
are skipped, not failed.
"""
import tempfile
import textwrap
import unittest
from pathlib import Path

import mujoco
import numpy as np

from PIPER.startup import approach_action, approach_trajectory, verify_arrival
from PIPER.trajectory import PEN_ROTATION
from .env import check_model
from .ik import make_solver
from .robot import ROOT, RobotSpec, all_robots, find_by_train_env, load_robot

HANZI = ROOT.parent / "hanzi-writer-data/data"


def built(name):
    spec = load_robot(name)
    if not spec.model.exists():
        raise unittest.SkipTest(f"{name} model not built: python -m writing.build_model --robot {name}")
    return spec


def write_and_track(spec, text, **kwargs):
    """Plan, approach from home for 5 s, then track with the baseline law; return errors and tilts."""
    env = spec.env_class()()
    xyz, q, down = spec.planner()(env, text, .08, spec.center, spec.height, .03, .025, HANZI,
                                  rotation_deg=spec.rotation_deg, **kwargs)
    home = np.radians(spec.home)
    times, approach, _ = approach_trajectory(home, q[0], env.limits)
    state = env.reset(home)
    for i in range(len(times) - 1):
        state = env.step(approach_action(approach[i], approach[i + 1], state[3:]))
    verify_arrival(state, env.data.qvel[env.vadr], q[0])
    errors, tilts = [], []
    for i in range(len(q) - 1):
        state = env.step(approach_action(q[i], q[i + 1], state[3:]))
        errors.append(np.linalg.norm(state[:3] - xyz[i + 1]))
        axis = env.data.site_xmat[env.site].reshape(3, 3)[:, 2]
        tilts.append(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))
    return xyz, down, np.array(errors), np.array(tilts)


class RegistryTest(unittest.TestCase):
    def test_existing_robots_keep_their_classes_and_dimensions(self):
        from PIPER.data_collection import PiperDataGenerator
        from PIPER.env import PiperEnv
        from X02.data_collection import X02WritingDataGenerator
        from x02cTpose.env import TPoseWritingEnv
        self.assertIs(load_robot("piper").env_class(), PiperEnv)
        self.assertIs(load_robot("piper").data_generator(), PiperDataGenerator)
        self.assertIs(find_by_train_env("X02Writing").data_generator(), X02WritingDataGenerator)
        self.assertIs(load_robot("x02c_tpose").env_class(), TPoseWritingEnv)
        dims = {s.train_env: (s.dof + 3, s.dof) for s in all_robots() if s.name in
                ("piper", "wheelarm_r", "x02", "x02c_tpose")}
        self.assertEqual(dims, {"PIPER": (9, 6), "WheelArmRWriting": (10, 7),
                                "X02Writing": (10, 7), "X02CTPoseWriting": (7, 4)})
        self.assertIsNone(find_by_train_env("SOARM101"))

    def test_ik_mode_follows_joint_count(self):
        spec = load_robot("piper")
        for dof, mode in ((7, "full_pose"), (6, "full_pose"), (5, "pen_axis"), (4, "position_soft_tilt")):
            spec.joints = [f"j{i}" for i in range(dof)]
            self.assertEqual(spec.resolved_ik_mode, mode)

    def test_yaml_errors_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("name: bad\ntrain_env: Bad\nmodel: x.xml\ncentre: [0, 0]\n")
            with self.assertRaisesRegex(ValueError, "centre"):
                RobotSpec.from_yaml(path)
            path.write_text("name: bad\nmodel: x.xml\n")
            with self.assertRaisesRegex(ValueError, "train_env"):
                RobotSpec.from_yaml(path)

    def test_model_without_writing_conventions_is_rejected(self):
        model = mujoco.MjModel.from_xml_string(textwrap.dedent("""
            <mujoco><worldbody><body><joint name="a" range="-1 1"/><geom size=".1"/>
            <body pos="0 0 .3"><joint name="b"/><geom size=".1"/></body></body></worldbody>
            <actuator><position joint="a"/></actuator></mujoco>"""))
        with self.assertRaises(ValueError) as caught:
            check_model(model, ["a", "b"])
        message = str(caught.exception)
        for expected in ("'b' needs a finite range", "1 actuators but 2", "pen_tip", "paper"):
            self.assertIn(expected, message)


class GeneratedRobotTest(unittest.TestCase):
    def test_g1_right_arm_writes_with_vertical_pen(self):
        spec = built("g1_right")
        env = spec.env_class()()
        self.assertEqual((env.model.nq, env.model.nu, env.xdim), (7, 7, 10))
        self.assertEqual(env.model.body("pelvis").jntnum[0], 0)  # legs, waist, left arm welded
        xyz, down, errors, tilts = write_and_track(spec, "二")
        np.testing.assert_allclose(xyz[down, 2], spec.height)
        self.assertEqual(np.count_nonzero(np.diff(np.r_[False, down].astype(int)) == 1), 2)
        self.assertLess(errors.max(), .002)
        self.assertLess(tilts.max(), .3)

    def test_so101_five_axes_keep_pen_axis_vertical(self):
        spec = built("so101")
        self.assertEqual(spec.resolved_ik_mode, "pen_axis")
        with self.assertRaisesRegex(ValueError, "pen yaw"):
            spec.planner()(spec.env_class()(), "二", .08, spec.center, spec.height, .03, .025, HANZI,
                           pen_yaw_deg=30)
        _, _, errors, tilts = write_and_track(spec, "HI")
        self.assertLess(errors.max(), .002)
        self.assertLess(tilts.max(), .3)

    def test_full_pose_solver_matches_target(self):
        spec = built("g1_right")
        env = spec.env_class()()
        point = np.array([*spec.center, spec.height + .03])
        q = make_solver("full_pose")(env, point, np.array(spec.seeds[0]))
        np.testing.assert_allclose(env.reset(q)[:3], point, atol=2e-4)
        np.testing.assert_allclose(env.data.site_xmat[env.site].reshape(3, 3), PEN_ROTATION, atol=.005)
        with self.assertRaises(ValueError):
            make_solver("full_pose")(env, np.array([3., 0, 0]), q)


if __name__ == "__main__":
    unittest.main()
