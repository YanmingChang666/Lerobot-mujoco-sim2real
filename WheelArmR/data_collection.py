"""Independent writing data/checkpoints; old WheelArmR caches are not used."""
from PIPER.data_collection import PiperDataGenerator
from .env import WheelArmWritingEnv, REFERENCE

class WheelArmWritingDataGenerator(PiperDataGenerator):
    env_factory = WheelArmWritingEnv
    reference_file = REFERENCE
    metadata_name = "wheelarm_writing_data.json"
