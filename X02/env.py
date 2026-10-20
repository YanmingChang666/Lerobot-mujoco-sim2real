"""X02 right arm: seven velocity commands and ten measured state components."""
from pathlib import Path
from PIPER.env import PiperEnv, DT, MAX_SPEED
HERE = Path(__file__).resolve().parent
MODEL = HERE / 'assets/x02_right_write.xml'
REFERENCE = HERE / 'data/writing_reference.npz'


class X02WritingEnv(PiperEnv):
    def __init__(self, xml_path=MODEL):
        super().__init__(xml_path, [f'joint{i}_r' for i in range(1, 8)])
