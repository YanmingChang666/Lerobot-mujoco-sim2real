"""Robot-agnostic writing: python -m writing --robot <name> ...; robots are YAML files in robots/."""
from .robot import RobotSpec, all_robots, load_robot, find_by_train_env  # noqa: F401
