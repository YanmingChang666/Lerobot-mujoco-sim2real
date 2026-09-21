# SO-ARM101 Koopman Sim2Real

A Koopman-operator-based sim-to-real project for the SO-ARM101 6-DoF robotic arm: data-driven dynamics modeling, prediction, and MPC control.

English | [简体中文](README_CN.md)

## ✨ Features

- 🤖 Multiple Koopman network architectures (DKUC, DBKN, IKN, IBKN, Koopformer, KANKoopman)
- 🎮 High-fidelity MuJoCo physics simulation
- 🎯 MPC controller implemented with CasADi
- 🔄 Real-time sim2real communication over ZMQ
- 📊 Full training, evaluation, and visualization pipeline

## 📁 Project Structure

```
├── SOARM101/                 # MuJoCo simulation environment
│   ├── SOARM101_Env.py       # Gymnasium-style env wrapper
│   ├── SOARM101_DataCollection.py  # Data collection script
│   └── SO101/                # Robot URDF/XML models
├── models/                   # Koopman network implementations
│   ├── base_model.py         # KoopmanNet base class
│   ├── KoopmanBase.py        # DKUC/DBKN linear/bilinear models
│   ├── InvertKoopman.py      # IKN/IBKN invertible-network models
│   └── losses.py             # Loss functions
├── control/                  # MPC control module
│   ├── MPC_Controler.py      # MPC controller implementation
│   ├── TrajectoryGenerator.py # Trajectory generator
│   └── config.py             # Experiment configuration management
├── lerobot_sim2real/         # Sim2real communication module
│   ├── so101_mujoco.py       # MuJoCo-side ZMQ publisher
│   └── so101_real.py         # Real-robot-side ZMQ subscriber
├── args.py                   # CLI argument configuration
└── train.py                  # Training/testing entry point
```

## 🛠️ Environment Setup

Use a dedicated conda environment. The versions pinned in `requirements.txt` no longer match what's actually available/compatible on PyPI (see "Known issues" below), so install in this order instead:

```bash
conda create -n koopman-sim2real python=3.10 -y
conda activate koopman-sim2real
cd Lerobot-mujoco-sim2real

# 1. torch needs the official CUDA index (many mirrors don't carry the +cu121 variant)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 2. lerobot pulls in a fairly modern dependency set (numpy 2.x / gymnasium 1.x, etc.) — install it on its own
pip install lerobot==0.4.4

# 3. Simulation / control packages
pip install casadi==3.7.2 dm_control==1.0.23 matplotlib mujoco==3.2.3 pyzmq tqdm tensorboard

# 4. Pinocchio (the correct PyPI package name is `pin`, not `pinocchio`)
pip install pin
```

> As long as your GPU driver isn't too old (verified on an RTX 4090 / driver 580.x), the cu121-built torch wheel works fine — CUDA runtimes bundled with pip wheels are backward compatible with newer drivers.

### Known issues in `requirements.txt`

A plain `pip install -r requirements.txt` currently fails. Root causes, for reference / future fixing:

| Issue | Symptom | Fix |
|---|---|---|
| `torch==2.4.1+cu121` | This CUDA build isn't hosted on most PyPI mirrors | Install from `--index-url https://download.pytorch.org/whl/cu121` instead |
| `pinocchio==0.4.3` | **Wrong package name.** On PyPI, `pinocchio` is an unrelated 12KB package, not the robotics library | The code does `import pinocchio as pin`; the correct package is `pin` |
| `rerun==1.0.30` | **Same mistake.** On PyPI, `rerun` is an unrelated 19KB package | The code does `import rerun as rr`; the correct package is `rerun-sdk` (already pulled in transitively by `lerobot`) |
| `gymnasium==1.0.0` / `numpy==1.24.4` | Conflicts with what `lerobot==0.4.4` actually requires (`gymnasium>=1.1.1`; `opencv-python-headless` transitively needs `numpy>=2`) | Don't pin these — let pip resolve them (ends up with `gymnasium>=1.2` and numpy 2.x; the codebase doesn't use any numpy-2.x-removed APIs, so this is safe) |
| Missing `tensorboard` | `train.py` does `from torch.utils.tensorboard import SummaryWriter`, which fails without it | `pip install tensorboard` |
| Unconditional `from .casadi_ik import Kinematics` at the top of `control/TrajectoryGenerator.py` | The pip `pin` wheel doesn't ship the `pinocchio.casadi` bindings, so this import crashes at module load time — even though the default code path (`use_pinocchio=False`) never actually uses it | Fixed in this repo: the import was moved into `CartesianTrajectoryGenerator_pinocchio._initialize_ik_solver`, where it's only needed |

## 🚀 Quick Start

### Train a model

```bash
# Train DBKN
python train.py --model DBKN --mode train

# Train the invertible Koopman network
python train.py --model IBKN --mode train

# Quick smoke test to verify the pipeline (finishes in a few minutes)
python train.py --model DBKN --mode train --suffix smoke_test \
  --train_samples 2000 --test_samples 200 --num_epochs 5 --eval_interval 1
```

### Visualize a trained model in MuJoCo

`IBKN_UKF.py` / `DBKN_KESO.py` load `results/SOARM101/<suffix>/<model>/best_model.pt` and run closed-loop MPC trajectory tracking in a MuJoCo viewer window. **Run these in an interactive terminal on a machine with a graphical desktop session** — the script waits for you to press Enter once the viewer window appears:

```bash
conda activate koopman-sim2real
cd Lerobot-mujoco-sim2real

# DBKN + KESO: Helix trajectory, noise/payload/KF/ESO all off by default (baseline tracking)
python DBKN_KESO.py --model DBKN --suffix 12_22

# IBKN + UKF: FigStar trajectory, noise + UKF state estimation on by default (disturbance rejection demo)
python IBKN_UKF.py --model IBKN --suffix 12_22
```

- `--model` / `--suffix` select which checkpoint to load (`results/SOARM101/<suffix>/<model>/best_model.pt`)
- Trajectory shape (Helix/FigStar/…) and whether to enable noise/payload/UKF/ESO are hardcoded in each script's `config_dict` inside `if __name__ == "__main__":` — **not** controlled by the `--traj_name`/`--noise`/`--UKF` CLI flags (those exist in `args.py` but aren't read by these two scripts)
- Once the window appears, press **Enter** at the prompt to start the simulation. When the trajectory finishes, the arm returns to its Home pose and results are saved to `control/ControlResults/<suffix>/<traj_name>/<method flags>.npz`
- Close the MuJoCo window to end the simulation

## 🧪 Main Experiments

This project validates two disturbance-rejection control architectures:

### 1. IBKN + UKF (UKF state estimation on an invertible Koopman network)

Combines the **Invertible Bilinear Koopman Network (IBKN)** with an **Unscented Kalman Filter (UKF)** to address model uncertainty and observation noise.

- **Run**:

```bash
  # Manual A/B comparison required
  python IBKN_UKF.py
```

- **Results**: ![control results](control/FigResults/12_11/noise_robustness.png)

### 2. DBKN + KESO (a Koopman Extended State Observer on a deep bilinear Koopman network)

Uses a **Deep Bilinear Koopman Network (DBKN)** with a **Koopman Extended State Observer (KESO)** to estimate and compensate for unmodeled disturbances in real time.

- **Run**:

```bash
  # Manual A/B comparison required
  python DBKN_KESO.py
```

- **Noise-robustness results**: ![control results](control/FigResults/12_22/noise_robustness_3rows.png)
- **Payload-robustness results**: ![control results](control/FigResults/12_22/payload_robustness_3rows.png)
- **Noise + payload robustness results**: ![control results](control/FigResults/12_22/both_robustness_3rows.png)

## 🧪 Sim2Real Deployment

![demo 1](control/media/Video1.gif)
![demo 2](control/media/Video2.gif)
