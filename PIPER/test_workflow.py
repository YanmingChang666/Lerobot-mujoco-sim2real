"""Headless integration checks: python -m unittest PIPER.test_workflow -v."""
import shutil
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from .env import PiperEnv, HERE, MODEL, DT
from .trajectory import plan, solve_ik, PEN_ROTATION
from .startup import HOME_DEG, approach_trajectory, approach_action, verify_arrival


class PiperWorkflowTest(unittest.TestCase):
    def test_five_second_startup(self):
        env = PiperEnv()
        target = solve_ik(env, np.array([.3, 0., .03]), np.array([0, 2, -1, 0, 1, 0]))
        home = np.radians(HOME_DEG)
        times, joints, velocities = approach_trajectory(home, target, env.limits)
        self.assertEqual(len(times), 251)
        self.assertEqual(times[-1], 5.)
        np.testing.assert_allclose(joints[0], home)
        np.testing.assert_allclose(joints[-1], target)
        np.testing.assert_allclose(velocities[[0, -1]], 0., atol=1e-12)
        self.assertLess(np.max(np.abs(velocities)), .45)
        self.assertTrue(np.all(joints >= env.limits[:, 0]))
        self.assertTrue(np.all(joints <= env.limits[:, 1]))
        state = env.reset(home)
        for i in range(len(times)-1):
            state = env.step(approach_action(joints[i], joints[i+1], state[3:]))
        error, speed = verify_arrival(state, env.data.qvel[env.vadr], target)
        self.assertAlmostEqual(env.data.time, 5., places=9)
        self.assertLess(error, .1)
        self.assertLess(speed, .02)
        with self.assertRaises(ValueError):
            approach_trajectory(home, target, env.limits, duration=1.)
        with self.assertRaises(ValueError):
            approach_trajectory(np.full(6, np.nan), target, env.limits)
        with self.assertRaises(ValueError):
            approach_trajectory(home, target, env.limits, duration=5.01)
        with self.assertRaises(RuntimeError):
            verify_arrival(np.zeros(9), np.zeros(6), target)

    def test_assets_are_relocatable(self):
        with tempfile.TemporaryDirectory() as directory:
            dst = Path(directory) / "assets"
            shutil.copytree(MODEL.parent, dst)
            model = mujoco.MjModel.from_xml_path(str(dst / "piper.xml"))
            self.assertEqual((model.nq, model.nv, model.nu), (6, 6, 6))
            self.assertEqual(model.nmesh, 11)
            for name in ("gripper_base", "gripper_link1", "gripper_link2"):
                self.assertGreaterEqual(model.body(name).id, 0)
            # The copied URDF is portable as well as the generated MJCF.
            imported = mujoco.MjModel.from_xml_path(str(dst / "piper_source.urdf"))
            self.assertEqual(imported.nq, 6)

    def test_pose_and_unreachable_target(self):
        env = PiperEnv()
        point = np.array([.3, 0., .03])
        q = solve_ik(env, point, np.array([0, 2, -1, 0, 1, 0]))
        np.testing.assert_allclose(env.reset(q)[:3], point, atol=2e-4)
        np.testing.assert_allclose(env.data.site_xmat[env.site].reshape(3, 3), PEN_ROTATION, atol=.005)
        with self.assertRaises(ValueError):
            solve_ik(env, np.array([2., 0., 0.]), q)

    def test_multistroke_dynamic_tracking(self):
        env = PiperEnv()
        xyz, q, down = plan(env, "二", .08, [.3, 0], 0., .03, .025,
                            HERE.parent.parent / "hanzi-writer-data/data")
        # There must be exactly two pen-down intervals, separated by lifts.
        self.assertEqual(np.count_nonzero(np.diff(np.r_[False, down].astype(int)) == 1), 2)
        self.assertTrue(np.allclose(xyz[down, 2], 0))
        self.assertGreater(xyz[~down, 2].max(), .025)
        state = env.reset(q[0])
        errors = []
        for i in range(len(q)-1):
            state = env.step((q[i+1]-q[i])/DT + 8*(q[i]-state[3:]))
            errors.append(np.linalg.norm(state[:3]-xyz[i+1]))
        self.assertLess(max(errors), .002)
        with self.assertRaises(ValueError):
            env.step(np.full(6, np.nan))


if __name__ == "__main__":
    unittest.main()
