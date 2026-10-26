"""Generic velocity-controlled writing environment built from a RobotSpec."""
import mujoco
import numpy as np

from PIPER.env import PiperEnv

ORIENTATION_LABELS = {
    "full_pose": "full_pose_tip_down",
    "pen_axis": "pen_axis_down_free_yaw",
    "position_soft_tilt": "position_with_soft_tilt",
}


def check_model(model, joint_names, source="model"):
    """Fail early with a clear message when an MJCF does not follow the writing conventions."""
    problems = []
    for name in joint_names:
        try:
            joint = model.joint(name)
        except KeyError:
            problems.append(f"joint '{name}' not found")
            continue
        if model.jnt_type[joint.id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            problems.append(f"joint '{name}' must be a hinge or slide joint (1 DoF)")
        low, high = model.jnt_range[joint.id]
        if not model.jnt_limited[joint.id] or not high > low:
            problems.append(f"joint '{name}' needs a finite range")
    if model.nu != len(joint_names):
        problems.append(f"{model.nu} actuators but {len(joint_names)} writing joints; "
                        "need exactly one velocity actuator per writing joint")
    else:
        for i, name in enumerate(joint_names):
            if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT or \
                    model.joint(model.actuator_trnid[i, 0]).name != name:
                problems.append(f"actuator {i} must drive joint '{name}' (same order as joints)")
            elif model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE or \
                    not np.isclose(model.actuator_biasprm[i, 2], -model.actuator_gainprm[i, 0]):
                problems.append(f"actuator {i} ('{name}') must be a <velocity> actuator")
    for kind, name in (("site", "pen_tip"), ("geom", "paper")):
        try:
            getattr(model, kind)(name)
        except KeyError:
            problems.append(f"missing {kind} '{name}'")
    if problems:
        raise ValueError(f"{source} does not follow the writing model conventions:\n  - "
                         + "\n  - ".join(problems)
                         + "\nGenerate it with: python -m writing.build_model --robot <name>")


def spec_env_class(spec):
    """A named PiperEnv subclass bound to one robot spec.

    A real class (not a closure) keeps training-data fingerprints stable,
    because the data generator hashes the file that defines the env class.
    """
    joint_names = spec.joint_names

    def __init__(self, xml_path=None):
        PiperEnv.__init__(self, xml_path or spec.model, joint_names)
        check_model(self.model, joint_names, xml_path or spec.model)

    return type(f"{spec.name}WritingEnv", (PiperEnv,), {
        "__init__": __init__, "__module__": __name__, "spec": spec,
        "orientation_constraint": ORIENTATION_LABELS[spec.resolved_ik_mode],
    })
