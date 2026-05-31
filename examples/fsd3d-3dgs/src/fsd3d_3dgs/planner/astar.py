"""A* 3D path planner on a voxel occupancy grid.

Plans drone flight paths from takeoff to landing, respecting:
  - 3D obstacle avoidance (voxel occupancy)
  - Maximum altitude constraint
  - Takeoff (ground → cruise altitude) and landing (cruise → ground)

Output is a sequence of (x, y, z) waypoints in scene-native coordinates,
downsampled for the §2 conditioner.
"""

import numpy as np
import heapq
from typing import List, Optional, Tuple

from fsd3d_3dgs.planner.voxel_map import VoxelMap


class AStarPlanner:
    """3D A* pathfinder on a voxel occupancy grid."""

    def __init__(self, voxel_map: VoxelMap, max_altitude: float = 30.0):
        self.voxel_map = voxel_map
        self.max_altitude = max_altitude

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        cruise_altitude: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Plan a path from start to goal.

        The path includes: takeoff → cruise → landing.

        Args:
            start:          (3,) start position (world coordinates).
            goal:           (3,) goal position (world coordinates).
            cruise_altitude: altitude for cruise phase (default: max_altitude * 0.6).

        Returns:
            List of (3,) waypoints in world coordinates.
        """
        if cruise_altitude is None:
            cruise_altitude = self.max_altitude * 0.6

        cruise_altitude = min(cruise_altitude, self.max_altitude)

        # Phase 1: Takeoff — start → cruise altitude above start
        takeoff_point = start.copy()
        takeoff_point[2] = cruise_altitude

        # Phase 2: Cruise — cruise altitude above start → above goal
        cruise_start = takeoff_point.copy()
        cruise_goal = goal.copy()
        cruise_goal[2] = cruise_altitude

        # Phase 3: Landing — cruise altitude above goal → goal
        landing_point = goal.copy()

        # Run A* for the cruise phase (the longest and most complex)
        cruise_path = self._astar(cruise_start, cruise_goal)

        # Assemble full path
        waypoints = [start.copy()]

        # Takeoff: interpolate from ground to cruise altitude
        n_takeoff = max(3, int(abs(cruise_altitude - start[2]) / self.voxel_map.resolution))
        for i in range(1, n_takeoff + 1):
            t = i / n_takeoff
            pos = start.copy()
            pos[2] = start[2] + t * (cruise_altitude - start[2])
            waypoints.append(pos)

        # Cruise: A* path
        if cruise_path:
            waypoints.extend(cruise_path)

        # Landing: interpolate from cruise altitude to ground
        n_landing = max(3, int(abs(cruise_altitude - goal[2]) / self.voxel_map.resolution))
        for i in range(1, n_landing + 1):
            t = i / n_landing
            pos = goal.copy()
            pos[2] = cruise_goal[2] + t * (goal[2] - cruise_goal[2])
            waypoints.append(pos)

        return waypoints

    def _astar(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Run A* on the voxel grid from start to goal.

        Both start and goal are in world coordinates at the same altitude.
        """
        vm = self.voxel_map
        start_idx = vm.world_to_grid(start)
        goal_idx = vm.world_to_grid(goal)

        # Ensure start/goal are within bounds and not occupied
        grid_shape = np.array(vm.grid_shape)
        start_idx = np.clip(start_idx, 0, grid_shape - 1)
        goal_idx = np.clip(goal_idx, 0, grid_shape - 1)

        # A* state: (f_score, counter, current_idx)
        counter = 0
        open_set = [(0, counter, tuple(start_idx))]
        came_from = {}
        g_score = {tuple(start_idx): 0}

        def heuristic(a, b):
            return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))

        # 26-connectivity (all neighbors in 3D)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((dx, dy, dz))

        max_iterations = 500000
        iteration = 0

        while open_set and iteration < max_iterations:
            _, _, current = heapq.heappop(open_set)
            current_arr = np.array(current)

            if np.array_equal(current_arr, goal_idx):
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(vm.grid_to_world(np.array(current)))
                    current = came_from[current]
                path.reverse()
                return path

            for dx, dy, dz in neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                neighbor_arr = np.array(neighbor)

                # Bounds check
                if np.any(neighbor_arr < 0) or np.any(neighbor_arr >= grid_shape):
                    continue

                # Altitude constraint (convert to world Z)
                world_z = vm.grid_to_world(neighbor_arr)[2]
                if world_z > self.max_altitude:
                    continue

                # Occupancy check
                if vm.grid[neighbor[0], neighbor[1], neighbor[2]]:
                    continue

                # Cost: Euclidean distance in grid space * resolution
                move_cost = np.sqrt(dx*dx + dy*dy + dz*dz) * vm.resolution
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal_idx) * vm.resolution
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

            iteration += 1

        # No path found — return direct line
        print(f"Warning: A* found no path from {start} to {goal}, using direct line")
        n_steps = max(3, int(np.linalg.norm(goal - start) / vm.resolution))
        path = []
        for i in range(1, n_steps + 1):
            t = i / n_steps
            path.append(start + t * (goal - start))
        return path

    @staticmethod
    def downsample_waypoints(
        waypoints: List[np.ndarray],
        target_count: int = 16,
    ) -> np.ndarray:
        """Downsample waypoints to a fixed count for the conditioner.

        Uses uniform spacing along the path arc length.

        Args:
            waypoints:    List of (3,) waypoints.
            target_count: Desired number of output waypoints.

        Returns:
            (target_count, 3) numpy array.
        """
        if len(waypoints) <= target_count:
            result = np.array(waypoints)
            # Pad by repeating last point
            while len(result) < target_count:
                result = np.vstack([result, result[-1:]])
            return result

        # Compute cumulative arc length
        pts = np.array(waypoints)
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
        total_length = cum_length[-1]

        # Sample at uniform arc length intervals
        target_lengths = np.linspace(0, total_length, target_count)
        result = np.zeros((target_count, 3))
        for i, t_len in enumerate(target_lengths):
            idx = np.searchsorted(cum_length, t_len, side='right') - 1
            idx = max(0, min(idx, len(pts) - 2))
            frac = (t_len - cum_length[idx]) / (seg_lengths[idx] + 1e-8)
            result[i] = pts[idx] + frac * diffs[idx]

        return result
