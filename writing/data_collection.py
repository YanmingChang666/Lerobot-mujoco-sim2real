"""Training-data generator for any robot spec; same sampling as the Piper/X02 writers."""
from PIPER.data_collection import PiperDataGenerator


def spec_data_generator(spec):
    return type(f"{spec.name}WritingDataGenerator", (PiperDataGenerator,), {
        "__module__": __name__,
        "env_factory": spec.env_class(),
        "reference_file": spec.reference,
        "metadata_name": spec.metadata_name,
    })
