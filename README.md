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
# General form: --model is one of DKUC / DBKN / IKN / IBKN / all
python train.py --model <DKUC|DBKN|IKN|IBKN|all> --mode train --suffix <your-tag>

# Train DBKN
python train.py --model DBKN --mode train

# Train the invertible Koopman network
python train.py --model IBKN --mode train

# Train all 4 architectures in one go
python train.py --model all --mode train --suffix <your-tag>

# Quick smoke test to verify the pipeline (finishes in a few minutes)
python train.py --model DBKN --mode train --suffix smoke_test \
  --train_samples 2000 --test_samples 200 --num_epochs 5 --eval_interval 1
```

Common hyperparameters (see `args.py`): `--train_samples` (default 50000), `--num_epochs` (default 500), `--lr` (default 1e-3), `--batch_size` (default 256). Training data is cached as `train_data_<train_samples>_<train_steps>.npy` under `SOARM101/data/` and reused automatically for matching sample counts. If you change `perturbation_limits` in `SOARM101_Env.py` and want it to actually take effect, either pick a different `--train_samples`/`--train_steps` (new filename) or delete the stale cache in `SOARM101/data/` first.

Evaluate a trained model offline (prediction error + plots, no MuJoCo window):

```bash
python train.py --model IBKN --mode test --suffix <the-suffix-you-trained-with>
# results land in results/SOARM101/<suffix>/IBKN/test_<random|sin|chirp>/
```

### 📈 Monitoring training progress

This is supervised learning, not RL — there's no "reward" to watch, only a **prediction loss** that should **go down**. Two ways to watch it live:

```bash
# Option 1: TensorBoard (recommended, gives you curves), can be started anytime during training
tensorboard --logdir results/SOARM101/<suffix>
# open http://localhost:6006 in a browser

# Option 2: read the JSON directly (only rewritten when a new best eval score is found, every --eval_interval epochs)
cat results/SOARM101/<suffix>/<model>/best_scores.json
```

`train_losses.json` (the full loss history) is only written once training **finishes entirely** — it won't exist yet mid-run, so use TensorBoard for live curves.

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

To change trajectory shape / plane orientation / disturbance toggles, edit these spots directly in the script:

| Want to change | Where | Values |
|---|---|---|
| Trajectory shape | `"traj_name"` in `config_dict` (`IBKN_UKF.py` / `DBKN_KESO.py`) | `Fig8` `FigStar` `Heart` `Rectangle` `Lissajous` (all planar), `Helix` (the only genuinely 3D one) |
| Plane orientation | `idx=` argument to `CartesianTrajectoryGenerator(...)` | `0` = X-Y plane (horizontal, table-parallel), `1` = Y-Z plane (vertical) |
| Noise on/off | `"use_nosie"` in `config_dict` | `True`/`False` |
| UKF on/off (`IBKN_UKF.py` only) | `"use_KEM"` in `config_dict` | `True`/`False` |
| KESO on/off (`DBKN_KESO.py` only) | `"use_eso"` in `config_dict` | `True`/`False` |
| Payload on/off (`DBKN_KESO.py` only) | `"use_payload"` in `config_dict` | `0`/`1`/`2` |

### Getting accurate tracking on X-Y (table-plane) trajectories, to reproduce the Sim2Real deployment videos

The `results/SOARM101/12_11` and `12_22` checkpoints were trained on data collected with the default `perturbation_limits = ±0.3 rad` in `SOARM101_Env.py` (small random perturbations around the zero pose). If you just flip `idx` to `0` to draw a table-parallel `Fig8`/`Heart`, **tracking degrades noticeably** — the traced path won't match the reference. This isn't a bug: those table-plane trajectories require joint angles far outside what the training data covered, e.g.:

| Joint | Training coverage (default) | Fig8 (X-Y) actually needs | Heart (X-Y) actually needs |
|---|---|---|---|
| shoulder_lift | ±0.3 rad | -1.26 ~ -0.11 rad | -0.36 ~ 0.57 rad |
| elbow_flex | ±0.3 rad | 0.22 ~ 0.75 rad | -0.03 ~ 0.70 rad |

To get accurate table-plane tracking, widen the perturbation range, recollect data, and retrain:

```python
# In SOARM101/SOARM101_Env.py — widen perturbation_limits to cover all 5 planar shapes
self.perturbation_limits = np.array([
    [-0.45, -1.35, -1.25, -0.75, -0.3],   # lower bound
    [ 0.45,  1.45,  0.85,  0.85,  0.3],   # upper bound
])
```

```bash
# Use a new suffix to force fresh data collection + training (the second run reuses the first's data)
python train.py --model IBKN --mode train --suffix wide_range
python train.py --model DBKN --mode train --suffix wide_range

# Then verify: set idx=0 and traj_name=Fig8/Heart in the script, and run with the new suffix
python IBKN_UKF.py --model IBKN --suffix wide_range
python DBKN_KESO.py --model DBKN --suffix wide_range
```

Before widening the range, check the physical joint limits (the `range` attribute on each `<joint>` in `SOARM101/SO101/so101_new_calib_v.xml`) and keep the new perturbation bounds within them.

## ✍️ Handwriting (multi-stroke: pen-up -> transit -> pen-down)

### What it does

`CartesianTrajectoryGenerator.generate_multi_stroke()` in `control/TrajectoryGenerator.py` supports "multi-stroke" trajectories — unlike `generate()`, which draws one continuous shape (Fig8/Heart/…), this writes each stroke separately at table height and automatically inserts a "lift (raise) → transit (move at lift height) → lower" segment between strokes. The output is still a plain `(N,3)` Cartesian point sequence, fed into the same existing IK + MPC pipeline — no changes needed to the control stack.

`control/hanzi_loader.py` converts stroke data from [hanzi-writer-data](https://github.com/chanind/hanzi-writer-data) (makemeahanzi stroke data, local path `/home/cym/ROS/Arms/isaac_so_arm101/hanzi-writer-data/data/`, 9574 characters) into the format `generate_multi_stroke()` expects, using each character's `medians` (stroke centerlines):

```python
from control.hanzi_loader import hanzi_to_strokes, text_to_strokes

strokes = hanzi_to_strokes("永", size_m=0.12)       # single character
strokes = text_to_strokes("你好", size_m=0.10)       # a string of characters laid out left-to-right (range grows fast — validate one character first)
```

`write_demo.py` is a ready-to-run demo that reuses the `Test` class (MuJoCo sim + MPC player) from `DBKN_KESO.py`:

```bash
conda activate koopman-sim2real
cd Lerobot-mujoco-sim2real
python write_demo.py --model DBKN --suffix wide_range
```

Change `CHAR = "二"` at the top of `write_demo.py` to write a different character. It also has a `check_reachability()` helper that warns (without stopping) if the character's required joint angles fall outside what `wide_range` training covered.

`write_height` (pen-down height) / `lift_height` (pen-up transit height) are set where `generate_multi_stroke()` is called, in meters, in world coordinates — the table surface in this model sits around `z≈0`, so `write_height=0.055` means 5.5cm above the table; if a real pen is mounted on the gripper, subtract however far the pen tip extends below the gripper's reference point. Both call sites (the preview pass and the real pass — see `fixed_wrist_roll_rad` below) must use the same value.

### Pinning the gripper's twist (wrist_roll)

By default `target_orientation=None` (IK leaves orientation unconstrained), so `wrist_roll` varies freely at every point. To hold it fixed at a specific value for the whole trajectory (e.g. the trajectory's natural starting orientation rotated by another 90°), use `fixed_wrist_roll_rad` — this isn't a post-hoc overwrite of the result array (which would leave the recorded XYZ inconsistent with the actual joint configuration); it excludes `wrist_roll` from the set of joints IK is allowed to solve for, so only the other 4 joints reach the target position — kinematically consistent:

```python
# Solve freely once first, to get the trajectory's natural starting wrist_roll
_, preview_joint_traj, _ = traj_generator.generate_multi_stroke(strokes, center=(0.35, 0.0))
fixed_wrist_roll_rad = preview_joint_traj[0, 4] - np.radians(90)   # -90° from the natural start

# Generate again, this time with wrist_roll pinned
cartesian_points, joint_angle_traj, pen_down_mask = traj_generator.generate_multi_stroke(
    strokes, center=(0.35, 0.0), fixed_wrist_roll_rad=fixed_wrist_roll_rad,
)
```

> Note: if the pinned angle is far from what this joint saw during training (`wrist_roll` only covered ±0.3rad≈±17° during `wide_range` training), `check_reachability()` will warn — the written shape shouldn't be affected much (XYZ position is mainly driven by the other 4 joints), but tracking accuracy for the gripper's own orientation isn't guaranteed; expect possible wobble/drift on that joint. The "return to Home" phase is unaffected by this parameter — it still uses the Home keyframe's `wrist_roll=0` (that segment is a plain linear interpolation from the trajectory's end pose to Home, unrelated to IK).

## 🤖 Deploying to the real arm

This step depends on a sibling project in the same workspace, [`so-arm101-ros2-bridge`](../so-arm101-ros2-bridge/), which has already turned `Lerobot-mujoco-sim2real/lerobot_sim2real/so101_real.py` into a bidirectional ZMQ bridge (see that project's README for details):

```
koopman-sim2real (python3.10)          lerobot-env (python3.12, has lerobot==0.6.2 + feetech-servo-sdk)
──────────────────────────             ──────────────────────────────────────────────────
write_demo.py / DBKN_KESO.py etc  ──ZMQ PUB :5555 (target joint degrees)──►  so101_real.py ──► real arm
(MuJoCo sim + Koopman + MPC)      ◄──ZMQ PUB :5556 (current joint degrees)──  (SUB :5555, PUB :5556)
```

The two environments are split because ROS2 Humble's bundled `rclpy` compiled extensions require Python 3.10, while the LeRobot version that supports `so101_follower` needs Python 3.12 — they can't coexist in one process. That's why the real-hardware driver (`so101_real.py`) must run in its own `lerobot-env`, separate from the `koopman-sim2real` env used for training/simulation.

### Running it

```bash
# Terminal 1 — real-hardware driver (holds the serial port exclusively)
conda activate lerobot-env
cd Lerobot-mujoco-sim2real/lerobot_sim2real
python so101_real.py

# Terminal 2 — sim + MPC + handwriting, automatically streams commands to the real arm over ZMQ
conda activate koopman-sim2real
cd Lerobot-mujoco-sim2real
python write_demo.py --model DBKN --suffix wide_range
```

Before actually starting to write/draw, `write_demo.py`/`DBKN_KESO.py`/`IBKN_UKF.py` call `sync_real_to_sim_start()`: it subscribes to the real robot's current pose (broadcast by `so101_real.py` on `:5556`) and smoothly interpolates over 5 seconds to the trajectory's start pose, instead of letting the real arm "blindly chase" once the command stream begins — so you can visually confirm sim and real are aligned before the real run starts. **This step requires `so101_real.py` to already be running**, otherwise it errors out after a 2-second timeout (it won't run blind).

### Calibrating `joint_offsets` (required first, otherwise angles will be wrong)

Real-robot target degrees = sim degrees − `joint_offsets` (the `sim_to_real()` function at the top of `DBKN_KESO.py`/`IBKN_UKF.py`). This offset corrects for the difference between the simulation model's joint zero and the real robot's calibrated zero. **It must be re-derived every time you re-run `lerobot-calibrate`**, otherwise the real arm will move to the wrong angles.

1. Calibrate the real robot with `lerobot-calibrate` first (a standard step from the `so-arm101-ros2-bridge` project, not something this project provides itself), producing `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`.
   - **To recalibrate**: rerun the same command; when prompted, type `c` and press Enter (just pressing Enter reuses the old file without actually recalibrating).
   - **To just read the current pose without recalibrating**: use `read_current_pose.py` (below) — don't run `lerobot-calibrate`.
2. Open a live MuJoCo window with the sim held at the `home` keyframe pose as a reference:
   ```bash
   conda activate koopman-sim2real
   cd Lerobot-mujoco-sim2real
   python show_home_pose.py
   ```
   You can rotate/zoom the view yourself; the terminal prints the target angles.
3. Manually pose the real arm to match the window (you may need to release/power off servo torque to move it freely by hand), then read the real robot's current angles (read-only, doesn't move anything or touch the calibration file):
   ```bash
   conda activate lerobot-env
   cd Lerobot-mujoco-sim2real/lerobot_sim2real
   python read_current_pose.py
   ```
4. For each joint compute `offset = sim target degrees - real measured degrees`, and fill in the 6 values of `joint_offsets` at the top of `DBKN_KESO.py`/`IBKN_UKF.py` (order: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`; the `gripper` entry doesn't matter since MPC doesn't control it).

**Known issues:**

| Symptom | Cause / fix |
|---|---|
| Computed offsets differ noticeably between attempts | Manual pose-matching has inherent error, especially for `wrist_roll` (a twist joint that's hard to judge by eye); repeat a few times and check for convergence — the big arm joints (`shoulder_pan`/`shoulder_lift`/`elbow_flex`) are easy to match accurately, the wrist joints need more care |
| Not sure the calibration file is trustworthy, want a "cleaner zero point" | Feasible — redo the `lerobot-calibrate` flow. But **this invalidates any previously-computed `joint_offsets`**, which then need to be re-derived via the Home-pose matching steps above |
| Residuals stay small (a few degrees) across repeated alignment attempts | Consider just setting `joint_offsets` to all zeros, trusting `lerobot-calibrate`'s own zero point — `real_robot.py`'s `max_relative_target=10.0` (max 10° per command relative to current position) safety net can absorb a few degrees of residual error |
| `python read_current_pose.py` raises `could not open port /dev/ttyACM0` | USB got disconnected or the arm lost power; reconnect/re-power it, confirm with `ls /dev/ttyACM0` that the device is back, then retry |
| Wanted to use `lerobot-calibrate` just to "read the position" | That's not what it does — `lerobot-calibrate` re-calibrates (and will ask you to sweep each joint through its full range of motion), it doesn't just read. For read-only, use `read_current_pose.py` |

### Safety notes

- `real_robot.py` (part of the `so-arm101-ros2-bridge` chain, at `lerobot_sim2real/real_robot.py`) already sets `max_relative_target=10.0` (degrees) — every command is clamped to move the real arm at most 10° from its **current** position. This is the last hardware-level safety net, but it doesn't mean you can stop watching it move.
- This control scheme is **open-loop, one-way**: MPC/Koopman computes everything from the simulation's own state and never reads back real sensor feedback. The real arm just passively "chases" whatever target angles the sim sends.
- Before the first real-hardware test of any new trajectory (new character / new height / new `fixed_wrist_roll_rad` setting): make sure there's nothing in the way around the arm and on the table, keep a hand nearby, watch it the whole time — especially any joints `check_reachability()` flagged — and be ready to Ctrl+C.

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
