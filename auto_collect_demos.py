"""Automated demo collection with IK and motion planning for bimanual SO-101.

Generates demonstrations of the donut packing task using Jacobian-based inverse
kinematics and waypoint interpolation. Saves to LeRobot v3.0 dataset format.

Task sequence:
  1. Left arm picks up box, holds it in the air
  2. Right arm picks up donut, places it in the held box
  3. Left arm puts box back on table

Usage:
    python auto_collect_demos.py --n-episodes 50 --force
    python auto_collect_demos.py --n-episodes 10 --noise-scale 0.01 --force
"""

import argparse
import os
import shutil
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from bi_so101_env import JOINT_NAMES, BiSO101Env

JOINT_NAMES_DOT = [f"{n}.pos" for n in JOINT_NAMES]

LEFT_IK_JOINTS = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
]
RIGHT_IK_JOINTS = [
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
]

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.5


class JacobianIKSolver:
    """Damped least-squares IK for a single SO-101 arm (5 DOF, no gripper)."""

    def __init__(self, model, data, site_name, joint_names, damping=0.05):
        self.model = model
        self.data = data
        self.damping = damping

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names
        ]
        self.dof_ids = [model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.qpos_addrs = [model.jnt_qposadr[jid] for jid in self.joint_ids]

        self.q_low = np.array([model.jnt_range[jid, 0] for jid in self.joint_ids])
        self.q_high = np.array([model.jnt_range[jid, 1] for jid in self.joint_ids])

        self.n_dof = len(joint_names)
        self.jacp = np.zeros((3, model.nv))

    def _get_q(self):
        return np.array([self.data.qpos[a] for a in self.qpos_addrs])

    def _set_q(self, q):
        for a, val in zip(self.qpos_addrs, q):
            self.data.qpos[a] = val

    def _get_site_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    def solve(self, target_pos, q_init=None, max_iter=100, tol=0.001):
        original_qpos = self.data.qpos.copy()
        original_qvel = self.data.qvel.copy()

        if q_init is not None:
            self._set_q(q_init)
        mujoco.mj_forward(self.model, self.data)

        for _ in range(max_iter):
            current_pos = self._get_site_pos()
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                result = self._get_q()
                self.data.qpos[:] = original_qpos
                self.data.qvel[:] = original_qvel
                mujoco.mj_forward(self.model, self.data)
                return result

            self.jacp[:] = 0
            mujoco.mj_jacSite(self.model, self.data, self.jacp, None, self.site_id)

            J = self.jacp[:, self.dof_ids]

            JJT = J @ J.T + (self.damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)

            q = self._get_q() + dq
            q = np.clip(q, self.q_low, self.q_high)
            self._set_q(q)
            mujoco.mj_forward(self.model, self.data)

        result = self._get_q()
        self.data.qpos[:] = original_qpos
        self.data.qvel[:] = original_qvel
        mujoco.mj_forward(self.model, self.data)
        return result


class AutoCollector:
    def __init__(self, env, dataset_name, dataset_root, task_description,
                 noise_scale=0.005, n_episodes=50, seed=0):
        self.env = env
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.task_description = task_description
        self.noise_scale = noise_scale
        self.n_episodes = n_episodes
        self.seed = seed

        self.left_ik = JacobianIKSolver(
            env.model, env.data, "left_gripperframe", LEFT_IK_JOINTS
        )
        self.right_ik = JacobianIKSolver(
            env.model, env.data, "right_gripperframe", RIGHT_IK_JOINTS
        )

        self._left_joint_indices = list(range(0, 5))
        self._left_gripper_index = 5
        self._right_joint_indices = list(range(6, 11))
        self._right_gripper_index = 11

    def _get_current_action(self):
        return self.env._get_joint_positions().copy()

    def _step_physics(self, action):
        clipped = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        self.env.data.ctrl[self.env._actuator_ids] = clipped
        for _ in range(self.env._n_substeps):
            mujoco.mj_step(self.env.model, self.env.data)
        self.env._step_count += 1

    def _interpolate_and_step(self, start_action, end_action, n_steps, states, actions):
        for i in range(n_steps):
            t = (i + 1) / n_steps
            action = start_action + t * (end_action - start_action)
            state = self.env._get_joint_positions()
            states.append(state.copy())
            actions.append(action.copy())
            self._step_physics(action)

    def _solve_left(self, target_pos, current_action):
        q_init = current_action[self._left_joint_indices]
        q_solved = self.left_ik.solve(target_pos, q_init=q_init)
        new_action = current_action.copy()
        for i, idx in enumerate(self._left_joint_indices):
            new_action[idx] = q_solved[i]
        return new_action

    def _solve_right(self, target_pos, current_action):
        q_init = current_action[self._right_joint_indices]
        q_solved = self.right_ik.solve(target_pos, q_init=q_init)
        new_action = current_action.copy()
        for i, idx in enumerate(self._right_joint_indices):
            new_action[idx] = q_solved[i]
        return new_action

    def _set_gripper(self, action, arm, value):
        new_action = action.copy()
        if arm == "left":
            new_action[self._left_gripper_index] = value
        else:
            new_action[self._right_gripper_index] = value
        return new_action

    def _add_noise(self, pos, rng):
        if self.noise_scale > 0:
            return pos + rng.normal(0, self.noise_scale, size=3)
        return pos

    def _noisy_steps(self, base_steps, rng):
        if self.noise_scale > 0:
            scale = 1.0 + rng.uniform(-0.1, 0.1)
            return max(5, int(base_steps * scale))
        return base_steps

    def _generate_episode(self, ep_seed):
        rng = np.random.RandomState(ep_seed)
        self.env.reset(seed=ep_seed)
        mujoco.mj_forward(self.env.model, self.env.data)

        initial_qpos = self.env.data.qpos.copy()
        initial_qvel = self.env.data.qvel.copy()

        donut_pos = self.env.data.site_xpos[self.env._donut_site_id].copy()
        box_pos = self.env.data.site_xpos[self.env._box_grasp_site_id].copy()

        states = []
        actions_list = []

        current_action = self._get_current_action()

        # Phase 1: Left arm → pre-grasp above box
        pre_grasp_box = self._add_noise(box_pos + np.array([0, 0, 0.06]), rng)
        target_action = self._solve_left(pre_grasp_box, current_action)
        n = self._noisy_steps(60, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 2: Left arm → grasp box (move down)
        grasp_box = self._add_noise(box_pos, rng)
        target_action = self._solve_left(grasp_box, current_action)
        n = self._noisy_steps(30, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 3: Left arm close gripper
        target_action = self._set_gripper(current_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(15, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 4: Left arm lifts box
        hold_pos = np.array([box_pos[0], box_pos[1], 0.55])
        hold_pos = self._add_noise(hold_pos, rng)
        target_action = self._solve_left(hold_pos, current_action)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 5: Right arm → pre-grasp above donut
        pre_grasp_donut = self._add_noise(donut_pos + np.array([0, 0, 0.06]), rng)
        target_action = self._solve_right(pre_grasp_donut, current_action)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(60, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 6: Right arm → grasp donut
        grasp_donut = self._add_noise(donut_pos, rng)
        target_action = self._solve_right(grasp_donut, current_action)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(30, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 7: Right arm close gripper
        target_action = self._set_gripper(current_action, "right", GRIPPER_CLOSED)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(15, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 8: Right arm lifts donut
        lift_donut = np.array([donut_pos[0], donut_pos[1], 0.55])
        lift_donut = self._add_noise(lift_donut, rng)
        target_action = self._solve_right(lift_donut, current_action)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        target_action = self._set_gripper(target_action, "right", GRIPPER_CLOSED)
        n = self._noisy_steps(30, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 9: Right arm → above held box
        # The box is now held by left arm at hold_pos; target above it
        above_box_held = hold_pos + np.array([0, 0, 0.06])
        target_action = self._solve_right(above_box_held, current_action)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        target_action = self._set_gripper(target_action, "right", GRIPPER_CLOSED)
        n = self._noisy_steps(50, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 10: Right arm → place donut in box (move down into box)
        into_box = hold_pos + np.array([0, 0, 0.02])
        target_action = self._solve_right(into_box, current_action)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        target_action = self._set_gripper(target_action, "right", GRIPPER_CLOSED)
        n = self._noisy_steps(30, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 11: Right arm open gripper (release donut)
        target_action = self._set_gripper(current_action, "right", GRIPPER_OPEN)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(15, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 12: Right arm retreat up
        retreat = hold_pos + np.array([0, 0, 0.08])
        target_action = self._solve_right(retreat, current_action)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(30, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 13: Right arm → home (all zeros for right joints)
        target_action = current_action.copy()
        for idx in self._right_joint_indices:
            target_action[idx] = 0.0
        target_action[self._right_gripper_index] = GRIPPER_OPEN
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 14: Left arm → table position (put box back)
        table_pos = np.array([box_pos[0], box_pos[1], 0.44])
        table_pos = self._add_noise(table_pos, rng)
        target_action = self._solve_left(table_pos, current_action)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 15: Left arm open gripper (release box)
        target_action = self._set_gripper(current_action, "left", GRIPPER_OPEN)
        n = self._noisy_steps(15, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 16: Left arm → home
        target_action = current_action.copy()
        for idx in self._left_joint_indices:
            target_action[idx] = 0.0
        target_action[self._left_gripper_index] = GRIPPER_OPEN
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)

        return initial_qpos, initial_qvel, states, actions_list

    def _create_dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        video_info = {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30,
            "video.channels": 3,
            "has_audio": False,
        }
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (12,),
                "names": JOINT_NAMES_DOT,
            },
            "observation.images.top_camera": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": video_info,
            },
            "observation.images.front_camera": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": video_info,
            },
            "observation.images.left_wrist_camera": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": video_info,
            },
            "observation.images.right_wrist_camera": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": video_info,
            },
            "action": {
                "dtype": "float32",
                "shape": (12,),
                "names": JOINT_NAMES_DOT,
            },
        }

        dataset = LeRobotDataset.create(
            repo_id=self.dataset_name,
            fps=30,
            features=features,
            root=self.dataset_root,
            robot_type="bi_so_follower",
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=1,
        )
        return dataset

    def _save_episode(self, dataset, initial_qpos, initial_qvel, states, actions_list):
        n_frames = len(actions_list)

        self.env.data.qpos[:] = initial_qpos
        self.env.data.qvel[:] = initial_qvel
        mujoco.mj_forward(self.env.model, self.env.data)

        for i in range(n_frames):
            obs = self.env._get_obs()
            frame = {
                "observation.state": states[i],
                "observation.images.top_camera": obs["pixels/top_camera"],
                "observation.images.front_camera": obs["pixels/front_camera"],
                "observation.images.left_wrist_camera": obs["pixels/left_wrist_camera"],
                "observation.images.right_wrist_camera": obs["pixels/right_wrist_camera"],
                "action": actions_list[i],
                "task": self.task_description,
            }
            dataset.add_frame(frame)

            clipped = np.clip(actions_list[i], self.env.action_space.low, self.env.action_space.high)
            self.env.data.ctrl[self.env._actuator_ids] = clipped
            for _ in range(self.env._n_substeps):
                mujoco.mj_step(self.env.model, self.env.data)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_frames} frames rendered")

        dataset.save_episode()

    def run(self):
        print("=" * 60)
        print("Automated Demo Collection - Bimanual SO-101 Donut Packing")
        print("=" * 60)
        print(f"Episodes: {self.n_episodes}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Noise scale: {self.noise_scale}")
        print(f"Task: {self.task_description}")
        print("=" * 60)

        dataset = self._create_dataset()

        for ep in range(self.n_episodes):
            ep_seed = self.seed + ep
            t_start = time.time()

            print(f"\nEpisode {ep + 1}/{self.n_episodes} (seed={ep_seed})")
            print("  Generating trajectory...")

            initial_qpos, initial_qvel, states, actions_list = self._generate_episode(ep_seed)

            print(f"  Generated {len(actions_list)} frames in {time.time() - t_start:.1f}s")
            print("  Rendering and saving...")

            t_render = time.time()
            self._save_episode(dataset, initial_qpos, initial_qvel, states, actions_list)

            print(f"  Saved in {time.time() - t_render:.1f}s "
                  f"(total: {time.time() - t_start:.1f}s)")

        dataset.finalize()
        print(f"\nDone! Saved {self.n_episodes} episodes to {self.dataset_root}")
        self.env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Automated demo collection for bimanual SO-101 donut packing"
    )
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--dataset-name", default="local/bi_so101_donut_auto")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--task", default="Pick up the donut and place it in the box.")
    parser.add_argument("--noise-scale", type=float, default=0.005)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing dataset directory")
    args = parser.parse_args()

    if args.dataset_root is None:
        args.dataset_root = os.path.join(
            os.path.dirname(__file__), "data", args.dataset_name
        )

    if os.path.exists(args.dataset_root):
        if args.force:
            print(f"Removing existing dataset: {args.dataset_root}")
            shutil.rmtree(args.dataset_root)
        else:
            print(f"Error: Dataset already exists at {args.dataset_root}")
            print("Use --force to overwrite.")
            sys.exit(1)

    env = BiSO101Env(fps=args.fps, max_episode_steps=10**9)
    collector = AutoCollector(
        env, args.dataset_name, args.dataset_root, args.task,
        noise_scale=args.noise_scale, n_episodes=args.n_episodes, seed=args.seed,
    )
    collector.run()


if __name__ == "__main__":
    main()
