"""Robot registry: one YAML file in robots/ describes one writing robot.

Paths in a YAML file are relative to the repository root (absolute paths also work).
See robots/README_CN.md for every field. Only `name`, `train_env` and `model`
are required; everything else has a default.
"""
from dataclasses import dataclass, field
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROBOTS_DIR = ROOT / "robots"
IK_MODES = ("auto", "full_pose", "pen_axis", "position_soft_tilt")


def _path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _import(dotted):
    """'package.module:attribute' -> object."""
    module, _, attribute = dotted.partition(":")
    if not attribute:
        raise ValueError(f"Expected 'module:attribute', got {dotted!r}")
    return getattr(importlib.import_module(module), attribute)


@dataclass
class RobotSpec:
    name: str
    train_env: str
    model: Path
    source: Path
    description: str = ""
    joints: list = None
    reference: Path = None
    cache_dir: Path = None
    output: Path = None
    metadata_name: str = None
    center: tuple = (.30, 0.)
    height: float = 0.
    home_deg: tuple = None
    camera_distance: float = 1.1
    camera_height: float = None
    rotation_deg: float = 0.
    pen_yaw_deg: float = 0.
    ik_mode: str = "auto"
    seeds: list = field(default_factory=list)
    random_seeds: int = 300
    continuity_weight: float = None
    overrides: dict = field(default_factory=dict)
    build: dict = None

    @classmethod
    def from_yaml(cls, path):
        import yaml
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        known = {"name", "train_env", "model", "description", "joints", "reference", "cache_dir",
                 "output", "metadata_name", "write_defaults", "ik", "overrides", "build"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{path.name}: unknown field(s) {sorted(unknown)}")
        for key in ("name", "train_env", "model"):
            if key not in raw:
                raise ValueError(f"{path.name}: missing required field '{key}'")
        name = str(raw["name"])
        home = ROBOTS_DIR / name
        defaults = raw.get("write_defaults") or {}
        ik = raw.get("ik") or {}
        unknown = set(defaults) - {"center", "height", "home_deg", "camera_distance", "camera_height",
                                 "rotation_deg", "pen_yaw_deg"}
        unknown |= set(ik) - {"mode", "seeds", "random_seeds", "continuity_weight"}
        unknown |= set(raw.get("overrides") or {}) - {"env_class", "planner", "data_generator"}
        if unknown:
            raise ValueError(f"{path.name}: unknown field(s) {sorted(unknown)}")
        mode = ik.get("mode", "auto")
        if mode not in IK_MODES:
            raise ValueError(f"{path.name}: ik.mode must be one of {IK_MODES}")
        spec = cls(
            name=name, train_env=str(raw["train_env"]), model=_path(raw["model"]), source=path,
            description=raw.get("description", ""),
            joints=list(raw["joints"]) if raw.get("joints") else None,
            reference=_path(raw.get("reference", home / "data/writing_reference.npz")),
            cache_dir=_path(raw.get("cache_dir", home / "data")),
            output=_path(raw.get("output", home / "outputs")),
            metadata_name=raw.get("metadata_name", f"{name}_writing_data.json"),
            center=tuple(float(v) for v in defaults.get("center", (.30, 0.))),
            height=float(defaults.get("height", 0.)),
            home_deg=tuple(float(v) for v in defaults["home_deg"]) if "home_deg" in defaults else None,
            camera_distance=float(defaults.get("camera_distance", 1.1)),
            camera_height=defaults.get("camera_height"),
            rotation_deg=float(defaults.get("rotation_deg", 0.)),
            pen_yaw_deg=float(defaults.get("pen_yaw_deg", 0.)),
            ik_mode=mode, seeds=[list(map(float, s)) for s in ik.get("seeds", [])],
            random_seeds=int(ik.get("random_seeds", 300)),
            continuity_weight=float(ik["continuity_weight"]) if "continuity_weight" in ik else None,
            overrides=dict(raw.get("overrides") or {}), build=raw.get("build"),
        )
        if len(spec.center) != 2:
            raise ValueError(f"{path.name}: write_defaults.center needs two numbers (X Y)")
        return spec

    # ---- derived properties -------------------------------------------------
    @property
    def joint_names(self):
        """Explicit `joints`, else the joints driven by the model's actuators, in actuator order."""
        if self.joints:
            return list(self.joints)
        if not hasattr(self, "_auto_joints"):
            import mujoco
            if not self.model.exists():
                raise FileNotFoundError(f"{self.name}: model {self.model} does not exist; build it first "
                                        f"(python -m writing.build_model --robot {self.name})")
            model = mujoco.MjModel.from_xml_path(str(self.model))
            self._auto_joints = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
        return self._auto_joints

    @property
    def dof(self):
        return len(self.joint_names)

    @property
    def home(self):
        home = self.home_deg if self.home_deg is not None else (0.,) * self.dof
        if len(home) != self.dof:
            raise ValueError(f"{self.name}: home_deg has {len(home)} values but the robot has {self.dof} joints")
        return home

    @property
    def resolved_ik_mode(self):
        if self.ik_mode != "auto":
            return self.ik_mode
        # 6+ joints can hold a fully fixed pen pose; 5 can keep the pen vertical
        # with free yaw; fewer joints only track position.
        return "full_pose" if self.dof >= 6 else "pen_axis" if self.dof == 5 else "position_soft_tilt"

    # ---- factories ------------------------------------------------------------
    def env_class(self):
        if "env_class" in self.overrides:
            return _import(self.overrides["env_class"])
        from .env import spec_env_class
        return spec_env_class(self)

    def planner(self):
        if "planner" in self.overrides:
            return _import(self.overrides["planner"])
        from .ik import spec_planner
        return spec_planner(self)

    def data_generator(self):
        if "data_generator" in self.overrides:
            return _import(self.overrides["data_generator"])
        from .data_collection import spec_data_generator
        return spec_data_generator(self)


def robot_files():
    return sorted(ROBOTS_DIR.glob("*.yaml"))


def all_robots():
    specs = [RobotSpec.from_yaml(path) for path in robot_files()]
    for key in ("name", "train_env"):
        seen = {}
        for spec in specs:
            value = getattr(spec, key)
            if value in seen:
                raise ValueError(f"Duplicate {key} {value!r} in {seen[value].name} and {spec.source.name}")
            seen[value] = spec.source
    return specs


def load_robot(name):
    for spec in all_robots():
        if spec.name == name:
            return spec
    available = ", ".join(s.name for s in all_robots())
    raise KeyError(f"Unknown robot {name!r}. Available: {available}")


def find_by_train_env(train_env):
    """train.py --env value -> RobotSpec, or None for the legacy (non-writing) environments."""
    for spec in all_robots():
        if spec.train_env == train_env:
            return spec
    return None
