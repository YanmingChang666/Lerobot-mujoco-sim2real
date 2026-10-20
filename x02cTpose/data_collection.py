from PIPER.data_collection import PiperDataGenerator
from .env import TPoseWritingEnv, REFERENCE

class TPoseWritingDataGenerator(PiperDataGenerator):
    env_factory=TPoseWritingEnv
    reference_file=REFERENCE
    metadata_name='tpose_writing_data.json'
