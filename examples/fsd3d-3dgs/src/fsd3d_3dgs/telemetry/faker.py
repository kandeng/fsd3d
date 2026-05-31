"""Telemetry faker — generate realistic telemetry from ground-truth A* path.

Simulates sensor readings a real drone would produce when flying a planned
path.  Adds noise profiles matching typical hardware:

  - GPS: Gaussian noise on position (~0.3 m CEP)
  - IMU velocity: Gaussian noise + random-walk bias
  - Barometer: altitude noise (~0.1 m)
  - Orientation: slow drift + fast jitter

The output is a (N, 9) array: [x, y, z, vx, vy, vz, roll, pitch, yaw]
ready for the §2 TelemetryEncoder.
"""

import numpy as np
from typing import Optional


class TelemetryFaker:
    """Generate realistic telemetry from ground-truth waypoint positions.

    Each call to `fake()` produces a noisy telemetry trace from a given
    ground-truth path.  Cumulative state (velocity bias, orientation drift)
    persists within a call but can be reset between calls.
    """

    def __init__(
        self,
        gps_sigma: float = 0.3,
        imu_vel_sigma: float = 0.05,
        imu_vel_walk_sigma: float = 0.01,
        baro_alt_sigma: float = 0.1,
        orient_drift_sigma: float = 0.02,
        orient_jitter_sigma: float = 0.005,
        dt: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.gps_sigma = gps_sigma
        self.imu_vel_sigma = imu_vel_sigma
        self.imu_vel_walk_sigma = imu_vel_walk_sigma
        self.baro_alt_sigma = baro_alt_sigma
        self.orient_drift_sigma = orient_drift_sigma
        self.orient_jitter_sigma = orient_jitter_sigma
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Cumulative state — reset per fake() call
        self._vel_bias = np.zeros(3)
        self._orient_drift = np.zeros(3)  # roll, pitch, yaw

    def reset(self) -> "TelemetryFaker":
        """Reset cumulative state for a fresh telemetry sequence."""
        self._vel_bias = np.zeros(3)
        self._orient_drift = np.zeros(3)
        return self

    def fake(self, waypoints: np.ndarray) -> np.ndarray:
        """Generate noisy telemetry from ground-truth waypoints.

        Args:
            waypoints: (N, 3) ground-truth positions from A* planner.

        Returns:
            (N, 9) telemetry array: [x, y, z, vx, vy, vz, roll, pitch, yaw].
        """
        N = waypoints.shape[0]
        assert waypoints.shape[1] == 3, f"Expected (N, 3) waypoints, got {waypoints.shape}"

        self.reset()
        telemetry = np.zeros((N, 9), dtype=np.float32)

        # --- Position noise ---
        gps_noise = self.rng.normal(0, self.gps_sigma, size=(N, 3))
        baro_noise = np.zeros((N, 3))
        baro_noise[:, 2] = self.rng.normal(0, self.baro_alt_sigma, size=N)

        noisy_pos = waypoints + gps_noise + baro_noise

        # --- Velocity (finite-diff of noisy positions + IMU noise) ---
        # Ground-truth velocity from finite differences
        gt_vel = np.zeros_like(waypoints)
        gt_vel[0] = (waypoints[1] - waypoints[0]) / self.dt if N > 1 else np.zeros(3)
        gt_vel[-1] = (waypoints[-1] - waypoints[-2]) / self.dt if N > 1 else np.zeros(3)
        if N > 2:
            gt_vel[1:-1] = (waypoints[2:] - waypoints[:-2]) / (2 * self.dt)

        noisy_vel = np.zeros_like(gt_vel)
        for i in range(N):
            # Random walk on velocity bias
            self._vel_bias += self.rng.normal(0, self.imu_vel_walk_sigma, size=3)
            # IMU Gaussian noise + cumulative bias
            imu_noise = self.rng.normal(0, self.imu_vel_sigma, size=3)
            noisy_vel[i] = gt_vel[i] + imu_noise + self._vel_bias

        # --- Orientation ---
        noisy_orient = np.zeros((N, 3), dtype=np.float32)  # roll, pitch, yaw
        for i in range(N):
            # Slow drift (cumulative)
            self._orient_drift += self.rng.normal(0, self.orient_drift_sigma, size=3)
            # Fast jitter
            jitter = self.rng.normal(0, self.orient_jitter_sigma, size=3)

            # Compute yaw from velocity direction
            if i < N - 1 and np.linalg.norm(gt_vel[i, :2]) > 0.01:
                yaw = np.arctan2(gt_vel[i, 1], gt_vel[i, 0])
            elif i > 0 and np.linalg.norm(gt_vel[i - 1, :2]) > 0.01:
                yaw = np.arctan2(gt_vel[i - 1, 1], gt_vel[i - 1, 0])
            else:
                yaw = 0.0

            # Roll and pitch: small angles proportional to lateral/longitudinal accel
            if i > 0:
                accel = (gt_vel[i] - gt_vel[i - 1]) / self.dt
                roll = np.clip(accel[1] * 0.02, -0.3, 0.3)   # lateral accel → roll
                pitch = np.clip(-accel[0] * 0.02, -0.3, 0.3)  # longitudinal accel → pitch
            else:
                roll, pitch = 0.0, 0.0

            noisy_orient[i] = [roll, pitch, yaw] + self._orient_drift + jitter

        # Assemble: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        telemetry[:, :3] = noisy_pos
        telemetry[:, 3:6] = noisy_vel
        telemetry[:, 6:9] = noisy_orient

        return telemetry
