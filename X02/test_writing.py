"""Headless tests: python -m unittest X02.test_writing -v."""
import shutil
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
from .env import X02WritingEnv, HERE
from .trajectory import plan, solve_ik, SEED, PEN_ROTATION
from PIPER.startup import approach_trajectory, approach_action, verify_arrival


class X02WritingTest(unittest.TestCase):
    def test_portable_assets_and_fixed_left_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            shutil.copy(HERE / 'assets/x02_right_write.xml', destination)
            shutil.copytree(HERE / 'assets/meshes', destination / 'meshes')
            model = mujoco.MjModel.from_xml_path(str(destination / 'x02_right_write.xml'))
            self.assertEqual((model.nq, model.nv, model.nu), (7, 7, 7))
            for i in range(1, 8):
                self.assertEqual(model.body(f'link{i}_l').jntnum[0], 0)
                self.assertEqual(model.joint(i-1).name, f'joint{i}_r')

    def test_hold_and_unreachable_pose(self):
        env = X02WritingEnv()
        state = env.reset(np.array(SEED))
        for _ in range(250):
            measured = env.step(np.zeros(7))
        np.testing.assert_allclose(measured, state, atol=1e-9)
        with self.assertRaises(ValueError):
            solve_ik(env, [2., 0., 0.], SEED)
        with self.assertRaises(ValueError):
            env.step(np.full(7, np.nan))

    def test_centered_writing_pose(self):
        from scipy.spatial.transform import Rotation
        env = X02WritingEnv()
        xyz, q, down = plan(env, '二', .08, [.2, 0], -.1, .03, .025,
                           HERE.parent.parent / 'hanzi-writer-data/data', pen_yaw_deg=90)
        target_rotation = Rotation.from_euler('z', 90, degrees=True).as_matrix() @ PEN_ROTATION
        self.assertTrue(np.all(np.abs(xyz[:, 1]) < .05))
        np.testing.assert_allclose(xyz[down, 2], -.1)
        approach_trajectory(np.zeros(7), q[0], env.limits, duration=5.)
        state = env.reset(q[0])
        errors = []
        for i in range(len(q)-1):
            state = env.step(approach_action(q[i], q[i+1], state[3:]))
            errors.append(np.linalg.norm(state[:3]-xyz[i+1]))
            np.testing.assert_allclose(env.data.site_xmat[env.site].reshape(3, 3),
                                       target_rotation, atol=.005)
        self.assertLess(max(errors), .002)

    def test_rotated_multistroke_from_home(self):
        env = X02WritingEnv()
        xyz, q, down = plan(env, '二', .08, [.2, -.25], -.23, .03, .025,
                           HERE.parent.parent / 'hanzi-writer-data/data', rotation_deg=-90)
        self.assertEqual(np.count_nonzero(np.diff(np.r_[False, down].astype(int)) == 1), 2)
        np.testing.assert_allclose(xyz[down, 2], -.23)
        self.assertGreater(xyz[~down, 2].max(), -.201)
        times, approach, speed = approach_trajectory(np.zeros(7), q[0], env.limits)
        np.testing.assert_allclose(speed[[0, -1]], 0., atol=1e-12)
        state = env.reset(np.zeros(7))
        for i in range(len(times)-1):
            state = env.step(approach_action(approach[i], approach[i+1], state[3:]))
        verify_arrival(state, env.data.qvel[env.vadr], q[0])
        self.assertAlmostEqual(env.data.time, 5., places=9)
        errors = []
        for i in range(len(q)-1):
            state = env.step(approach_action(q[i], q[i+1], state[3:]))
            errors.append(np.linalg.norm(state[:3]-xyz[i+1]))
            np.testing.assert_allclose(env.data.site_xmat[env.site].reshape(3, 3), PEN_ROTATION, atol=.005)
        self.assertLess(max(errors), .002)


if __name__ == '__main__':
    unittest.main()
