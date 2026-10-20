"""Run: python -m unittest WheelArmR.test_writing -v."""
import unittest
import numpy as np
from .env import WheelArmWritingEnv, HERE, DT
from .trajectory import plan, solve_ik, SEED, PEN_ROTATION
from PIPER.startup import approach_trajectory, approach_action, verify_arrival


class WritingTest(unittest.TestCase):
    def test_hold_and_invalid_commands(self):
        env = WheelArmWritingEnv()
        q = np.array(SEED)
        initial = env.reset(q)
        for _ in range(250):
            state = env.step(np.zeros(7))
        np.testing.assert_allclose(state, initial, atol=1e-9)
        self.assertEqual((env.xdim, env.udim), (10, 7))
        for bad in (np.zeros(6), np.full(7, np.nan)):
            with self.assertRaises(ValueError): env.step(bad)
        with self.assertRaises(ValueError): solve_ik(env, [2., 0., 0.], q)

    def test_startup_and_multistroke(self):
        env = WheelArmWritingEnv()
        xyz, joints, down = plan(env, '二', .08, [.35, -.4], 0., .03, .025,
                                HERE.parent.parent / 'hanzi-writer-data/data')
        self.assertEqual(np.count_nonzero(np.diff(np.r_[False, down].astype(int)) == 1), 2)
        self.assertTrue(np.allclose(xyz[down, 2], 0))
        self.assertGreater(xyz[~down, 2].max(), .025)
        times, approach, velocity = approach_trajectory(np.zeros(7), joints[0], env.limits)
        np.testing.assert_allclose(velocity[[0, -1]], 0, atol=1e-12)
        state = env.reset(np.zeros(7))
        for i in range(len(times)-1):
            state = env.step(approach_action(approach[i], approach[i+1], state[3:]))
        verify_arrival(state, env.data.qvel[env.vadr], joints[0])
        self.assertAlmostEqual(env.data.time, 5., places=9)
        errors, tilts = [], []
        # Continue from actual arrival without resetting the robot.
        for i in range(len(joints)-1):
            state = env.step(approach_action(joints[i], joints[i+1], state[3:]))
            errors.append(np.linalg.norm(state[:3]-xyz[i+1]))
            rotation = env.data.site_xmat[env.site].reshape(3, 3)
            tilts.append(np.max(np.abs(rotation-PEN_ROTATION)))
        self.assertLess(max(errors), .002)
        self.assertLess(max(tilts), .005)


if __name__ == '__main__': unittest.main()
