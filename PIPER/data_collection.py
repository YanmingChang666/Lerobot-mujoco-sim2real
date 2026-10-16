"""Task-local Piper data; reuse the existing training loader interface."""
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from SOARM101.SOARM101_DataCollection import SOARM101DataGenerator, Collater
from .env import PiperEnv, REFERENCE, DT, MAX_SPEED


def initialize_piper_model(model, env, reference):
    """Near-identity initialization for slow 50Hz velocity dynamics.

    This is only an initial guess; MuJoCo sequences still train all operators.
    Initialize B using the local derivative of the lifted kinematic state.
    """
    import torch
    if type(model).__name__ != "KoopmanBlinear" or model.use_stable:
        raise ValueError("Piper writing currently supports standard DBKN (no --use_stable/--u_z)")
    if model.u_z:
        raise ValueError("Piper MPC expects default DBKN with u_z=False")
    q = reference.mean(axis=0)
    device = next(model.parameters()).device
    with torch.no_grad():
        model.lA.weight.copy_(torch.eye(model.Nkoopman, device=device))
        model.H.weight.zero_()
        for j in range(6):
            delta = np.zeros(6)
            delta[j] = 1e-4
            plus = torch.tensor(env.reset(q+delta), dtype=torch.float32, device=device)[None]
            minus = torch.tensor(env.reset(q-delta), dtype=torch.float32, device=device)[None]
            derivative = (model.x_encoder(plus)-model.x_encoder(minus)).squeeze(0)/(2e-4)
            model.lB.weight[:, j].copy_(DT*derivative)


class PiperDataGenerator(SOARM101DataGenerator):
    def __init__(self, args):
        self.args = args
        self.udim, self.xdim = 6, 9
        self.env = PiperEnv(args.xml_path)
        self.collate_fn = Collater(9, 6, args.device)
        if not REFERENCE.exists():
            raise FileNotFoundError("First run: python -m PIPER.write --mode plan")
        with np.load(REFERENCE) as reference:
            model_hash = hashlib.sha256(Path(args.xml_path).read_bytes()).hexdigest()
            if reference["model_sha256"].item() != model_hash:
                raise ValueError("Piper XML changed; regenerate the reference before training")
            self.reference = reference["q"].copy()
        self.rng = np.random.default_rng(args.seed)
        # Hash all environment/model assets, reference and this sampler so a
        # change in geometry, task, timestep or excitation cannot reuse old data.
        digest = hashlib.sha256()
        for path in [Path(__file__), Path(__file__).with_name("env.py"), REFERENCE,
                     *sorted(Path(args.xml_path).parent.rglob("*.xml")),
                     *sorted(Path(args.xml_path).parent.rglob("*.stl"))]:
            digest.update(path.read_bytes())
        digest.update(str(args.seed).encode())
        self.fingerprint = digest.hexdigest()[:16]
        self.cache = Path(args.data_dir_save) / self.fingerprint
        self.cache.mkdir(parents=True, exist_ok=True)

    def generate_physics_based_data(self, traj_num, steps, input_type):
        samples = np.empty((traj_num, steps + 1, 15), dtype=np.float32)
        for i in tqdm(range(traj_num), desc=f"Piper {input_type}"):
            q = self.reference[self.rng.integers(len(self.reference))] + self.rng.uniform(-.12, .12, 6)
            state = self.env.reset(np.clip(q, self.env.limits[:, 0]+.02, self.env.limits[:, 1]-.02))
            phase = self.rng.uniform(0, 2*np.pi, 6)
            frequency = self.rng.uniform(.2, 2., 6)
            amplitude = self.rng.uniform(.05, .3, 6)
            for k in range(steps + 1):
                if input_type == "random":
                    action = self.rng.uniform(-.3, .3, 6)
                else:
                    t = k*DT
                    angle = frequency*t + (.5*t*t if input_type == "chirp" else 0)
                    action = amplitude*np.sin(2*np.pi*angle+phase)
                # Keep sampled trajectories within the task neighborhood and limits.
                action += np.clip(2*(q-state[3:]), -.15, .15)
                action = np.clip(action, -MAX_SPEED, MAX_SPEED)
                samples[i, k] = np.r_[action, state]
                if k < steps:
                    state = self.env.step(action)
        return samples

    def generate_and_save_data(self):
        def get(split, count, steps, kind):
            path = self.cache / f"{split}_{count}_{steps}.npy"
            if not path.exists():
                np.save(path, self.generate_physics_based_data(count, steps, kind))
            return np.load(path)
        a = self.args
        self.train_data = get("train", a.train_samples, a.train_steps, "random")
        self.val_data = get("val", a.test_samples, a.test_steps, "random")
        self.test_data_dict = {kind: get("test_"+kind, a.test_samples, a.test_steps, kind)
                               for kind in ("random", "sin", "chirp")}
        output = Path(a.output_dir)
        if a.mode == "train":
            output.mkdir(parents=True, exist_ok=True)
            (output / "piper_data.json").write_text(json.dumps({
                "fingerprint": self.fingerprint, "cache": str(self.cache),
                "dt": DT, "x_dim": 9, "u_dim": 6,
                "model_sha256": hashlib.sha256(Path(a.xml_path).read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                "q_min": self.train_data[:, :, 9:].min(axis=(0, 1)).tolist(),
                "q_max": self.train_data[:, :, 9:].max(axis=(0, 1)).tolist(),
            }, indent=2))
