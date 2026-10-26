"""Four-axis URDF fidelity and continuous-writing integration checks."""
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from .env import TPoseWritingEnv, HERE
from .prepare_assets import JOINT_NAMES, ASSETS
from .trajectory import plan, solve_ik, SEEDS
from PIPER.startup import approach_trajectory, approach_action, verify_arrival


class TPoseWritingTest(unittest.TestCase):
    def test_urdf_limits_and_fixed_body(self):
        env=TPoseWritingEnv()
        self.assertEqual((env.model.nq,env.model.nv,env.model.nu),(4,4,4))
        source=ET.parse(ASSETS/'x2c_TPose.urdf').getroot()
        for i,name in enumerate(JOINT_NAMES):
            joint=source.find(f'joint[@name="{name}"]');limit=joint.find('limit')
            np.testing.assert_allclose(env.limits[i],[float(limit.get('lower')),float(limit.get('upper'))])
        self.assertEqual(env.model.body('right_wrist_roll_link').jntnum[0],0)
        self.assertEqual(env.model.body('pelvis').jntnum[0],0)
        with self.assertRaises(ValueError):
            plan(env,'HI',.08,[.3,-.2],.9,.03,.025,HERE,pen_yaw_deg=90)
        with self.assertRaises(ValueError): solve_ik(env,[10,0,0],SEEDS[0])

    def test_five_second_startup_and_english(self):
        env=TPoseWritingEnv()
        xyz,q,down=plan(env,'HI',.08,[.3,-.2],.9,.03,.025,HERE,rotation_deg=-90)
        self.assertGreater(np.count_nonzero(np.diff(np.r_[False,down].astype(int))==1),2)
        np.testing.assert_allclose(xyz[down,2],.9)
        t,approach,velocity=approach_trajectory(np.zeros(4),q[0],env.limits)
        state=env.reset(np.zeros(4))
        for i in range(len(t)-1):
            state=env.step(approach_action(approach[i],approach[i+1],state[3:]))
        verify_arrival(state,env.data.qvel[env.vadr],q[0])
        self.assertAlmostEqual(env.data.time,5.,places=9)
        errors=[]
        for i in range(len(q)-1):
            state=env.step(approach_action(q[i],q[i+1],state[3:]))
            errors.append(np.linalg.norm(state[:3]-xyz[i+1]))
        self.assertLess(max(errors),.002)
        self.assertTrue(np.isfinite(state).all())


if __name__=='__main__':unittest.main()
