"""Keyboard teleoperation demo collector for bimanual SO-101 simulation.

Records demonstrations into LeRobot v3.0 dataset format.

Controls:
  Left arm:   Q/E=shoulder_pan  W/S=shoulder_lift  A/D=elbow_flex
              R/F=wrist_flex    T/G=wrist_roll     Space=gripper toggle
  Right arm:  U/O=shoulder_pan  I/K=shoulder_lift   J/L=elbow_flex
              Y/H=wrist_flex    P/;=wrist_roll      Enter=gripper toggle

  Z = discard & reset episode
  X = save episode & start new one
  Esc = finish and finalize dataset
"""

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bi_so101_env import JOINT_NAMES, BiSO101Env

JOINT_NAMES_DOT = [f"{n}.pos" for n in JOINT_NAMES]

LEFT_KEYS = {
    "q": (0, -1),  # left_shoulder_pan -
    "e": (0, +1),  # left_shoulder_pan +
    "w": (1, +1),  # left_shoulder_lift +
    "s": (1, -1),  # left_shoulder_lift -
    "a": (2, -1),  # left_elbow_flex -
    "d": (2, +1),  # left_elbow_flex +
    "r": (3, +1),  # left_wrist_flex +
    "f": (3, -1),  # left_wrist_flex -
    "t": (4, +1),  # left_wrist_roll +
    "g": (4, -1),  # left_wrist_roll -
}

RIGHT_KEYS = {
    "u": (6, -1),  # right_shoulder_pan -
    "o": (6, +1),  # right_shoulder_pan +
    "i": (7, +1),  # right_shoulder_lift +
    "k": (7, -1),  # right_shoulder_lift -
    "j": (8, -1),  # right_elbow_flex -
    "l": (8, +1),  # right_elbow_flex +
    "y": (9, +1),  # right_wrist_flex +
    "h": (9, -1),  # right_wrist_flex -
    "p": (10, +1), # right_wrist_roll +
    ";": (10, -1), # right_wrist_roll -
}

STEP_SIZE = 0.1


class TeleopCollector:
    def __init__(self, env: BiSO101Env, dataset_name: str, dataset_root: str, task_description: str):
        self.env = env
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.task_description = task_description

        self._action = np.zeros(12, dtype=np.float32)
        self._left_gripper_open = True
        self._right_gripper_open = True
        self._pressed_keys: set[str] = set()
        self._episode_frames: list[dict] = []
        self._episodes_saved = 0
        self._should_save = False
        self._should_discard = False
        self._should_quit = False
        self._recording = True
        self._episode_ended = False

    def _key_callback(self, keycode, action_type):
        key_char = chr(keycode).lower() if 32 <= keycode < 127 else ""

        if action_type == 1:  # press
            self._pressed_keys.add(key_char)

            if keycode == 32:  # Space - toggle left gripper
                self._left_gripper_open = not self._left_gripper_open
            elif keycode == 257:  # Enter - toggle right gripper
                self._right_gripper_open = not self._right_gripper_open
            elif key_char == "z":
                self._should_discard = True
            elif key_char == "x":
                self._should_save = True
            elif keycode == 256:  # Esc
                self._should_quit = True

        elif action_type == 0:  # release
            self._pressed_keys.discard(key_char)

    def _compute_action(self, current_pos: np.ndarray) -> np.ndarray:
        delta = np.zeros(12, dtype=np.float32)

        for key in self._pressed_keys:
            if key in LEFT_KEYS:
                idx, sign = LEFT_KEYS[key]
                delta[idx] += sign * STEP_SIZE
            elif key in RIGHT_KEYS:
                idx, sign = RIGHT_KEYS[key]
                delta[idx] += sign * STEP_SIZE

        self._pressed_keys.clear()

        target = current_pos + delta

        # Gripper targets
        target[5] = 0.0 if self._left_gripper_open else 1.5
        target[11] = 0.0 if self._right_gripper_open else 1.5

        target = np.clip(target, self.env.action_space.low, self.env.action_space.high)
        return target

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
                "shape": [12],
                "names": JOINT_NAMES_DOT,
            },
            "observation.images.top_camera": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": video_info,
            },
            "observation.images.front_camera": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": video_info,
            },
            "observation.images.left_wrist_camera": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": video_info,
            },
            "observation.images.right_wrist_camera": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": video_info,
            },
            "action": {
                "dtype": "float32",
                "shape": [12],
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

    def _save_episode_to_dataset(self, dataset):
        if len(self._episode_frames) == 0:
            print("No frames to save, skipping.")
            return

        for frame in self._episode_frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        self._episodes_saved += 1
        print(f"Episode {self._episodes_saved} saved ({len(self._episode_frames)} frames)")
        self._episode_frames = []

    def run(self):
        print("=" * 60)
        print("Bimanual SO-101 Keyboard Teleoperation")
        print("=" * 60)
        print(__doc__)
        print(f"Dataset: {self.dataset_name}")
        print(f"Task: {self.task_description}")
        print("=" * 60)

        dataset = self._create_dataset()

        obs, _ = self.env.reset(seed=42)
        current_pos = obs["agent_pos"].copy()

        def key_callback(keycode):
            self._key_callback(keycode, 1)

        with mujoco.viewer.launch_passive(
            self.env.model, self.env.data, key_callback=key_callback
        ) as viewer:
            frame_count = 0
            episode_start_time = time.time()

            while viewer.is_running() and not self._should_quit:
                step_start = time.time()

                if self._should_discard:
                    print(f"Episode discarded ({len(self._episode_frames)} frames)")
                    self._episode_frames = []
                    obs, _ = self.env.reset()
                    current_pos = obs["agent_pos"].copy()
                    self._left_gripper_open = True
                    self._right_gripper_open = True
                    self._should_discard = False
                    self._episode_ended = False
                    episode_start_time = time.time()
                    frame_count = 0

                if self._should_save:
                    self._save_episode_to_dataset(dataset)
                    obs, _ = self.env.reset()
                    current_pos = obs["agent_pos"].copy()
                    self._left_gripper_open = True
                    self._right_gripper_open = True
                    self._should_save = False
                    self._episode_ended = False
                    episode_start_time = time.time()
                    frame_count = 0

                if not self._episode_ended:
                    action = self._compute_action(current_pos)

                    frame = {
                        "observation.state": current_pos.copy(),
                        "observation.images.top_camera": obs["pixels/top_camera"],
                        "observation.images.front_camera": obs["pixels/front_camera"],
                        "observation.images.left_wrist_camera": obs["pixels/left_wrist_camera"],
                        "observation.images.right_wrist_camera": obs["pixels/right_wrist_camera"],
                        "action": action.copy(),
                        "task": self.task_description,
                    }
                    self._episode_frames.append(frame)
                    frame_count += 1

                    obs, reward, terminated, truncated, info = self.env.step(action)
                    current_pos = obs["agent_pos"].copy()

                    if terminated or truncated:
                        elapsed = time.time() - episode_start_time
                        status = "SUCCESS" if info["is_success"] else "timeout"
                        print(f"Episode ended ({status}, {frame_count} frames, {elapsed:.1f}s)")
                        print("Press X to save, Z to discard")
                        self._episode_ended = True

                viewer.sync()

                # Maintain target FPS
                elapsed = time.time() - step_start
                sleep_time = (1.0 / self.env.fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Save any remaining episode
        if len(self._episode_frames) > 0:
            print("Saving final episode...")
            self._save_episode_to_dataset(dataset)

        dataset.finalize()
        print(f"\nDone! Saved {self._episodes_saved} episodes to {self.dataset_root}")
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect teleoperation demos for bimanual SO-101")
    parser.add_argument("--dataset-name", default="local/bi_so101_donut_packing",
                        help="Dataset repo_id (default: local/bi_so101_donut_packing)")
    parser.add_argument("--dataset-root", default=None,
                        help="Local dataset root directory")
    parser.add_argument("--task", default="Pick up the donut and place it in the box.",
                        help="Task description string")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max steps per episode (0 = no timeout, use X/Z to end)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing dataset directory if it exists")
    args = parser.parse_args()

    if args.dataset_root is None:
        args.dataset_root = os.path.join(os.path.dirname(__file__), "data", args.dataset_name)

    if os.path.exists(args.dataset_root):
        if args.force:
            import shutil
            print(f"Removing existing dataset: {args.dataset_root}")
            shutil.rmtree(args.dataset_root)
        else:
            print(f"Error: Dataset already exists at {args.dataset_root}")
            print("Use --force to overwrite, or --dataset-name to use a different name.")
            sys.exit(1)

    max_steps = args.max_steps if args.max_steps > 0 else 10**9
    env = BiSO101Env(fps=args.fps, max_episode_steps=max_steps)
    collector = TeleopCollector(env, args.dataset_name, args.dataset_root, args.task)
    collector.run()


if __name__ == "__main__":
    main()
