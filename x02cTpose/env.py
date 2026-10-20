"""Four active right-arm joints; all other joints are welded at URDF zero."""
from pathlib import Path
from PIPER.env import PiperEnv, DT, MAX_SPEED
from .prepare_assets import JOINT_NAMES
HERE=Path(__file__).resolve().parent
MODEL=HERE/'assets/right_write.xml'
REFERENCE=HERE/'data/writing_reference.npz'

class TPoseWritingEnv(PiperEnv):
    orientation_constraint = "position_with_soft_tilt"
    def __init__(self, xml_path=MODEL):
        super().__init__(xml_path, JOINT_NAMES)
