"""X02-specific data and metadata; shared sampling implementation."""
from PIPER.data_collection import PiperDataGenerator
from .env import X02WritingEnv, REFERENCE


class X02WritingDataGenerator(PiperDataGenerator):
    env_factory = X02WritingEnv
    reference_file = REFERENCE
    metadata_name = 'x02_writing_data.json'
