"""Seven-axis writing model; original wheel_arm_r.xml remains a legacy asset."""
from pathlib import Path
from PIPER.env import PiperEnv, DT, MAX_SPEED
HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "robots/wheelarm_r/assets/wheel_arm_r_write.xml"
REFERENCE = HERE / "data/writing_reference.npz"
class WheelArmWritingEnv(PiperEnv):
    def __init__(self, xml_path=MODEL):
        super().__init__(xml_path, [f"Link{i}_R_joint" for i in range(1, 8)])
