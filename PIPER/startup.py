"""Timed, zero-end-velocity startup from a configurable joint home pose."""
import numpy as np

from .env import DT, MAX_SPEED

# Standard Piper MoveIt fake-system initial_positions.yaml, joint1..joint6.
# This is a simulation reference, not a measurement of the physical robot.
HOME_DEG = (0., 0., 0., 0., 0., 0.)


def approach_trajectory(home, target, limits, duration=5., dt=DT):
    """Cosine-ramped velocity, common time law for all six joints.

    Unlike a quintic move, the long constant-speed section can cover the
    default home-to-writing displacement within 5s and the 0.5rad/s limit.
    Position, velocity and acceleration are continuous, with zero endpoint
    velocity/acceleration. This is joint interpolation, not collision planning.
    """
    home, target = np.asarray(home, dtype=float), np.asarray(target, dtype=float)
    if home.shape != (6,) or target.shape != (6,) or not np.isfinite([home, target]).all():
        raise ValueError("Home and target must each contain six finite joint angles")
    if np.any(np.minimum(home, target) < limits[:, 0]) or np.any(np.maximum(home, target) > limits[:, 1]):
        raise ValueError("Home/approach target is outside Piper joint limits")
    if not np.isfinite(duration) or duration < 2*dt:
        raise ValueError("Approach duration must be finite and at least two control steps")
    count = round(duration/dt)
    if not np.isclose(count*dt, duration, atol=1e-9):
        raise ValueError(f"Approach duration must be a multiple of {dt}s")
    ramp = min(.5, duration*.1)
    delta = target-home
    peak = np.max(np.abs(delta))/(duration-ramp)
    # Leave headroom for feedback rather than silently clipping a fast plan.
    if peak > .9*MAX_SPEED:
        minimum = np.max(np.abs(delta))/(.9*MAX_SPEED) + ramp
        raise ValueError(f"Approach needs {peak:.3f}rad/s; increase --approach-seconds to at least {minimum:.2f}s")
    times = np.linspace(0., duration, count+1)
    position = np.empty_like(times)
    velocity = np.empty_like(times)
    for i, t in enumerate(times):
        if t < ramp:
            position[i] = (t/2-ramp*np.sin(np.pi*t/ramp)/(2*np.pi))/(duration-ramp)
            velocity[i] = (1-np.cos(np.pi*t/ramp))/(2*(duration-ramp))
        elif t > duration-ramp:
            remaining = duration-t
            position[i] = 1-(remaining/2-ramp*np.sin(np.pi*remaining/ramp)/(2*np.pi))/(duration-ramp)
            velocity[i] = (1-np.cos(np.pi*remaining/ramp))/(2*(duration-ramp))
        else:
            position[i] = (t-ramp/2)/(duration-ramp)
            velocity[i] = 1/(duration-ramp)
    return times, home+position[:, None]*delta, velocity[:, None]*delta


def approach_action(q_reference, q_next, measured_q, dt=DT):
    return np.clip((q_next-q_reference)/dt + 8*(q_reference-measured_q), -MAX_SPEED, MAX_SPEED)


def verify_arrival(state, measured_velocity, target):
    if not np.isfinite(state).all() or not np.isfinite(measured_velocity).all():
        raise RuntimeError("Non-finite startup measurement; writing not started")
    error_deg = float(np.max(np.abs(np.degrees(state[3:]-target))))
    speed = float(np.max(np.abs(measured_velocity)))
    if error_deg > .5 or speed > .02:
        raise RuntimeError(f"Approach did not settle: error={error_deg:.3f}deg, speed={speed:.4f}rad/s; writing not started")
    return error_deg, speed
