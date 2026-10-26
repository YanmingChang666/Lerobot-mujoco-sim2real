"""Six-joint velocity-controlled Piper; state = [pen xyz, six joint angles]."""
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "robots/piper/assets/piper.xml"
REFERENCE = HERE / "data/reference.npz"
DT = 0.02
MAX_SPEED = 0.5


class PiperEnv:
    def __init__(self, xml_path=MODEL, joint_names=None):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = joint_names or [f"joint{i}" for i in range(1, 7)]
        joints = [self.model.joint(name) for name in self.joint_names]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.vadr = np.array([j.dofadr[0] for j in joints])
        self.limits = np.array([j.range for j in joints])
        self.site = self.model.site("pen_tip").id
        self.udim, self.xdim = len(joints), len(joints)+3
        self.frame_skip = round(DT / self.model.opt.timestep)
        self.dt = self.frame_skip * self.model.opt.timestep

    def state(self):
        mujoco.mj_forward(self.model, self.data)
        return np.r_[self.data.site_xpos[self.site], self.data.qpos[self.qadr]].copy()

    def reset(self, q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = q
        return self.state()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.udim,) or not np.isfinite(action).all():
            raise ValueError(f"Action must contain {self.udim} finite velocities (rad/s)")
        self.data.ctrl[:] = np.clip(action, -MAX_SPEED, MAX_SPEED)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        state = self.state()
        if not np.isfinite(state).all():
            raise RuntimeError("Non-finite simulation state")
        return state

    def close(self):
        pass
