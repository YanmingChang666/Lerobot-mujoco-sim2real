"""Plan and simulate Piper Chinese writing; never connects to a real robot."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import threading

import mujoco
import numpy as np

from .env import PiperEnv, HERE, MODEL, REFERENCE, DT, MAX_SPEED
from .trajectory import plan, PEN_ROTATION
from .visualization import draw_trajectories
from .startup import HOME_DEG, approach_trajectory, approach_action, verify_arrival


class LearnedMPC:
    """DBKN MPC with frozen bilinear B(z0) and explicit velocity bounds."""
    def __init__(self, checkpoint, horizon=10, xdim=9, udim=6):
        self.udim = udim
        import casadi as ca
        import torch
        from models.KoopmanBase import KoopmanBlinear
        self.torch = torch
        self.horizon = horizon
        self.net = KoopmanBlinear(xdim, udim, [xdim, 64, 64, 64, 24], False, False).double().eval()
        self.net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        A = self.net.lA.weight.detach().numpy()
        B = self.net.lB.weight.detach().numpy()
        H = self.net.get_Hi_numpy()
        nz = len(A)
        z0 = ca.SX.sym("z0", nz)
        ref = ca.SX.sym("ref", xdim*horizon)
        prev = ca.SX.sym("prev", udim)
        controls = ca.SX.sym("u", udim*horizon)
        Beff = ca.DM(B)
        for j in range(nz):
            Beff += z0[j]*ca.DM(H[j])
        z, cost, last = z0, 0, prev
        weights = ca.DM([3000]*3 + [3]*udim)
        for k in range(horizon):
            u = controls[k*udim:(k+1)*udim]
            z = ca.DM(A)@z + Beff@u
            err = z[:xdim] - ref[k*xdim:(k+1)*xdim]
            cost += ca.dot(weights*err, err) + .02*ca.sumsqr(u) + .02*ca.sumsqr(u-last)
            last = u
        self.solver = ca.nlpsol("piper_mpc", "ipopt", {
            "x": controls, "p": ca.vertcat(z0, ref, prev), "f": cost,
        }, {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes", "ipopt.max_iter": 50})
        self.previous = np.zeros(udim)
        self.warmstart = np.zeros(horizon*udim)

    def action(self, state, future):
        with self.torch.no_grad():
            z = self.net.x_encoder(self.torch.tensor(state).double()[None]).numpy().ravel()
        solution = self.solver(x0=self.warmstart, p=np.r_[z, future.ravel(), self.previous],
                               lbx=-MAX_SPEED, ubx=MAX_SPEED)
        if not self.solver.stats()["success"]:
            raise RuntimeError("MPC solver failed: " + self.solver.stats()["return_status"])
        u = np.asarray(solution["x"]).ravel()
        self.previous = u[:self.udim].copy()
        self.warmstart = np.r_[u[self.udim:], u[-self.udim:]]
        return self.previous


def plot_result(path, reference, actual, down):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Break lines between strokes: no spurious ink during lifts/transfers.
    for points, label, color in [(reference, "reference", "red"), (actual, "simulated pen", "green")]:
        ink = points.copy()
        ink[~down] = np.nan
        axes[0].plot(ink[:, 0], ink[:, 1], label=label, color=color,
                     linestyle="--" if label == "reference" else "-",
                     linewidth=2.5 if label == "reference" else 1)
    axes[0].axis("equal")
    axes[0].set(xlabel="world X (m)", ylabel="world Y (m)")
    axes[0].legend()
    axes[1].plot(np.arange(len(actual))*DT, np.linalg.norm(actual-reference, axis=1)*1000)
    axes[1].set(xlabel="time (s)", ylabel="pen error (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(*, env_factory=PiperEnv, planner=plan, here=HERE, reference=REFERENCE,
         model_file=None, center=(.30, 0), height=0., home_deg=HOME_DEG, metadata_name="piper_data.json", camera_distance=1.1, camera_height=None,
         output_dir=None, hanzi_data=None, argv=None, prog=None, robot_name=None,
         rotation_deg=0., pen_yaw_deg=0., approach_seconds=5.):
    model_file = Path(model_file or MODEL)
    udim, xdim = len(home_deg), len(home_deg)+3
    parser = argparse.ArgumentParser(prog=prog, description=f"{robot_name or here.name} Chinese/English writing simulation")
    parser.add_argument("--mode", choices=["plan", "baseline", "mpc"], default="baseline")
    parser.add_argument("--text", default="二", help="Chinese text or English A-Z/a-z/spaces (rendered as capitals)")
    parser.add_argument("--size", type=float, default=.08, help="Meters: Chinese character size; English whole-text bounding box")
    parser.add_argument("--center", type=float, nargs=2, default=center)
    parser.add_argument("--pen-yaw-deg", type=float, default=pen_yaw_deg,
                        help="Pen orientation about world Z; does not rotate the text")
    parser.add_argument("--rotation-deg", type=float, default=rotation_deg,
                        help="Rotate text about its center around world +Z; clockwise = negative")
    parser.add_argument("--height", type=float, default=height)
    parser.add_argument("--lift", type=float, default=.03, help="Lift above writing plane (m)")
    parser.add_argument("--speed", type=float, default=.025, help="Peak Cartesian segment speed (m/s)")
    parser.add_argument("--hanzi-data", type=Path, default=Path(hanzi_data) if hanzi_data else here.parent.parent / "hanzi-writer-data/data")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--snapshot", type=Path, help="Save final 3D scene with robot and both trails")
    parser.add_argument("--hold", action="store_true", help="Keep viewer open after writing until closed")
    parser.add_argument("--home-deg", type=float, nargs=udim, default=home_deg,
                        metavar="DEG",
                        help="Initial joint positions in degrees (default: all zero)")
    parser.add_argument("--approach-seconds", type=float, default=approach_seconds, help="Home-to-first-stroke-above duration")
    parser.add_argument("--start-hold", type=float, default=1., help="Display initial pose for this many seconds (GUI only)")
    parser.add_argument("--approach-only", action="store_true", help="Stop above the first stroke after startup; do not write")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path(output_dir) if output_dir else here / "outputs")
    args = parser.parse_args(argv)
    if not np.isfinite(args.start_hold) or args.start_hold < 0:
        parser.error("--start-hold must be finite and nonnegative")
    view_height = args.height+.1 if camera_height is None else camera_height
    from scipy.spatial.transform import Rotation
    pen_rotation = Rotation.from_euler("z", args.pen_yaw_deg, degrees=True).as_matrix() @ PEN_ROTATION
    env = env_factory()
    if args.text.isascii():
        print("English block capitals:", args.text.strip().upper(), "(whole text fits --size)", flush=True)
    orientation_constraint = getattr(env, "orientation_constraint", "full_pose_tip_down")
    print(f"Planning {args.text!r}: {udim} joints, {orientation_constraint}, dt={DT}s", flush=True)
    xyz, q, down = planner(env, args.text, args.size, args.center, args.height, args.lift,
                        args.speed, args.hanzi_data, rotation_deg=args.rotation_deg, pen_yaw_deg=args.pen_yaw_deg)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "plan":
        reference.parent.mkdir(exist_ok=True)
        np.savez(reference, xyz=xyz, q=q, pen_down=down, dt=DT, text=args.text,
                 orientation_constraint=orientation_constraint, rotation=pen_rotation, pen_yaw_deg=args.pen_yaw_deg, rotation_deg=args.rotation_deg, model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest())
        plot_result(args.output / "reference.png", xyz, xyz, down)
        print(f"Saved {reference}; {len(q)} poses, {len(q)*DT:.2f}s; peak joint speed "
              f"{np.max(np.abs(np.diff(q, axis=0)/DT)):.4f}rad/s")
        return
    home = np.radians(args.home_deg)
    approach_t, approach_q, approach_v = approach_trajectory(home, q[0], env.limits, args.approach_seconds)
    # FK preview uses a separate model; runtime data is reset only once to home.
    preview = env_factory()
    approach_xyz = np.array([preview.reset(pose)[:3] for pose in approach_q])
    controller = None
    if args.mode == "mpc" and not args.approach_only:
        if args.checkpoint is None:
            parser.error("--mode mpc requires a matching DBKN --checkpoint")
        metadata = json.loads(args.checkpoint.with_name(metadata_name).read_text())
        if metadata["x_dim"] != xdim or metadata["u_dim"] != udim:
            raise ValueError("Checkpoint state/action dimensions do not match robot")
        model_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
        if metadata["model_sha256"] != model_hash or metadata["dt"] != DT:
            raise ValueError("Model/timestep differs from training; regenerate data and retrain")
        if np.any(q < np.array(metadata["q_min"])) or np.any(q > np.array(metadata["q_max"])):
            raise ValueError("Writing trajectory exceeds training joint coverage; re-plan and retrain")
        controller = LearnedMPC(args.checkpoint, xdim=xdim, udim=udim)
    # Set the initial simulation pose once; subsequent motion uses mj_step,
    # not per-frame qpos teleportation. There is no hardware publisher.
    state = env.reset(home)
    print("Initial joint pose (deg):", np.degrees(home).round(4))
    print("Initial pen XYZ in base/world frame (m):", state[:3].round(6))
    print("Approach target joints (deg):", np.degrees(q[0]).round(4))
    print("Approach target XYZ (m):", xyz[0].round(6), "(above first stroke)")
    print(f"Approach: {args.approach_seconds:.2f}s, {len(approach_t)-1} steps, "
          f"peak reference velocity {np.max(np.abs(approach_v)):.4f}rad/s")
    paper = env.model.geom("paper")
    half_angle = np.radians(args.rotation_deg)/2
    paper.quat[:] = [np.cos(half_angle), 0, 0, np.sin(half_angle)]
    paper.pos[:] = [*args.center, args.height-.002]
    paper.size[:2] = [max(.13, args.size*(1 if args.text.isascii() else len(args.text))*.7), max(.13, args.size*.7)]
    full_reference = np.vstack([approach_xyz, xyz[1:]])
    full_down = np.r_[np.zeros(len(approach_xyz), dtype=bool), down[1:]]
    display_actual = [state[:3]]
    states = np.c_[xyz, q]
    viewer = None
    viewer_threads = []
    if not args.headless:
        from mujoco import viewer as mj_viewer
        existing_threads = set(threading.enumerate())
        viewer = mj_viewer.launch_passive(env.model, env.data)
        # MuJoCo 3.2.3 close() only requests exit. Its daemon render thread
        # must finish destroying the window before Python runs glfw.terminate
        # at exit; otherwise a short approach-only run can segfault on exit.
        viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads]
        viewer.cam.lookat[:] = [args.center[0], args.center[1], view_height]
        viewer.cam.distance = camera_distance
        viewer.cam.azimuth, viewer.cam.elevation = 90, -45
        with viewer.lock():
            draw_trajectories(viewer.user_scn, full_reference, display_actual, full_down)
        viewer.sync()
        print("Red: reference trajectory; green: actual pen trajectory (lighter = pen up)")

    def finish_view():
        if args.snapshot:
            from PIL import Image
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            with mujoco.Renderer(env.model, height=480, width=640) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [args.center[0]-.08, args.center[1], view_height+.04]
                camera.distance, camera.azimuth, camera.elevation = camera_distance*.85/1.1, 130, -35
                renderer.update_scene(env.data, camera=camera)
                draw_trajectories(renderer.scene, full_reference, display_actual, full_down, append=True)
                Image.fromarray(renderer.render()).save(args.snapshot)
        if viewer is not None and args.hold:
            print("Motion complete. Close the MuJoCo window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(.02)

    try:
        if viewer is not None:
            until = time.monotonic()+args.start_hold
            while time.monotonic() < until and viewer.is_running():
                viewer.sync()
                time.sleep(min(.02, max(0., until-time.monotonic())))
        approach_states, approach_actions = [state.copy()], []
        start_sim, start_wall = env.data.time, time.monotonic()
        for i in range(len(approach_t)-1):
            if viewer is not None and not viewer.is_running():
                break
            command = approach_action(approach_q[i], approach_q[i+1], state[3:])
            state = env.step(command)
            approach_states.append(state.copy())
            approach_actions.append(command)
            display_actual.append(state[:3])
            if viewer is not None:
                with viewer.lock():
                    draw_trajectories(viewer.user_scn, full_reference, display_actual, full_down)
                viewer.sync()
                # Absolute deadlines avoid accumulating rendering overhead.
                time.sleep(max(0., start_wall+approach_t[i+1]-time.monotonic()))
        approach_done = len(approach_states) == len(approach_t)
        elapsed_wall = time.monotonic()-start_wall
        m = len(approach_states)
        approach_stem = args.output / f"{args.text}_approach"
        np.savez(str(approach_stem)+".npz", time_s=approach_t[:m],
                 q_reference_rad=approach_q[:m], q_reference_deg=np.degrees(approach_q[:m]),
                 qvel_reference_rad_s=approach_v[:m], xyz_reference_m=approach_xyz[:m],
                 actual_states=np.array(approach_states), actions_rad_s=np.array(approach_actions),
                 joint_names=np.array(env.joint_names), dt=DT)
        arrival_error = float(np.max(np.abs(np.degrees(state[3:]-q[0]))))
        approach_metrics = {"completed": approach_done, "duration_s": args.approach_seconds,
                            "simulated_duration_s": float(env.data.time-start_sim),
                            "wall_duration_s": elapsed_wall, "headless": args.headless,
                            "home_deg": list(args.home_deg), "target_deg": np.degrees(q[0]).tolist(),
                            "initial_tip_xyz_m": approach_xyz[0].tolist(),
                            "target_tip_xyz_m": xyz[0].tolist(), "max_arrival_joint_error_deg": arrival_error,
                            "arrival_max_joint_speed_rad_s": float(np.max(np.abs(env.data.qvel[env.vadr]))),
                            "hardware_connected": False}
        Path(str(approach_stem)+".json").write_text(json.dumps(approach_metrics, indent=2))
        print("Approach:", json.dumps(approach_metrics, indent=2))
        if not approach_done:
            print("Approach interrupted; writing not started.")
            return
        verify_arrival(state, env.data.qvel[env.vadr], q[0])
        if args.approach_only:
            finish_view()
            return
        print("Approach complete; lowering pen and starting writing.")
        # Carry actual state and velocity forward: no reset/teleport at handoff.
        actual, actions, orientations = [state[:3]], [], []
        if controller:
            controller.previous = approach_actions[-1].copy()
        for i in range(len(q)-1):
            start = time.monotonic()
            if viewer is not None and not viewer.is_running():
                break
            if controller:
                indices = np.minimum(np.arange(i+1, i+1+controller.horizon), len(q)-1)
                action = controller.action(state, states[indices])
            else:
                # Explicit, separately labelled non-learning baseline.
                action = (q[i+1]-q[i])/DT + 8*(q[i]-state[3:])
            action = np.clip(action, -MAX_SPEED, MAX_SPEED)
            state = env.step(action)
            if np.any(state[3:] < env.limits[:, 0]-.01) or np.any(state[3:] > env.limits[:, 1]+.01):
                raise RuntimeError("Simulated joint limit violation")
            actual.append(state[:3])
            display_actual.append(state[:3])
            actions.append(action)
            axis = env.data.site_xmat[env.site].reshape(3, 3)[:, 2]
            orientations.append(np.arccos(np.clip(-axis[2], -1, 1)))
            if viewer is not None:
                with viewer.lock():
                    draw_trajectories(viewer.user_scn, full_reference, display_actual, full_down)
                viewer.sync()
                time.sleep(max(0, DT-(time.monotonic()-start)))
        actual = np.array(actual)
        n = len(actual)
        error = np.linalg.norm(actual-xyz[:n], axis=1)
        metrics = {"orientation_constraint": orientation_constraint, "mode": args.mode, "text": args.text, "rotation_deg": args.rotation_deg, "pen_yaw_deg": args.pen_yaw_deg, "completed": n == len(q),
                   "steps": n, "rmse_mm": float(np.sqrt(np.mean(error**2))*1000),
                   "max_error_mm": float(error.max()*1000),
                   "max_pen_tilt_deg": float(np.degrees(max(orientations, default=0))),
                   "contact_simulated": False, "hardware_connected": False}
        metrics["approach_duration_s"] = args.approach_seconds
        metrics["tracking_pass_2mm"] = bool(n == len(q) and error.max() < .002)
        stem = args.output / f"{args.text}_{args.mode}"
        np.savez(str(stem)+".npz", reference=xyz[:n], actual=actual, q_reference=q[:n],
                 pen_down=down[:n], actions=np.array(actions), dt=DT)
        Path(str(stem)+".json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        plot_result(str(stem)+".png", xyz[:n], actual, down[:n])
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        finish_view()
    finally:
        if viewer is not None:
            viewer.close()
            for thread in viewer_threads:
                thread.join(timeout=5.)


if __name__ == "__main__":
    main()
