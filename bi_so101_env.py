import os
import pathlib

import gymnasium as gym
import mujoco
import numpy as np

SCENE_XML = str(pathlib.Path(__file__).parent / "assets" / "scene_bi_so101_donut.xml")

JOINT_NAMES = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]

N_JOINTS = len(JOINT_NAMES)
IMG_HEIGHT = 480
IMG_WIDTH = 640
DEFAULT_FPS = 30
MAX_EPISODE_STEPS = 300


class BiSO101Env(gym.Env):
    """Bimanual SO-101 donut packing environment.

    Two SO-101 arms on a table with a donut and an open-top box.
    Goal: place the donut inside the box.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": DEFAULT_FPS}

    def __init__(
        self,
        scene_xml: str = SCENE_XML,
        fps: int = DEFAULT_FPS,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        img_height: int = IMG_HEIGHT,
        img_width: int = IMG_WIDTH,
        render_mode: str = "rgb_array",
        success_threshold: float = 0.03,
        donut_x_range: tuple[float, float] = (0.0, 0.08),
        donut_y_range: tuple[float, float] = (-0.05, 0.05),
    ):
        super().__init__()
        self.fps = fps
        self.max_episode_steps = max_episode_steps
        self.img_height = img_height
        self.img_width = img_width
        self.render_mode = render_mode
        self.success_threshold = success_threshold
        self.donut_x_range = donut_x_range
        self.donut_y_range = donut_y_range

        os.environ.setdefault("MUJOCO_GL", "egl")

        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        n_physics_steps = int(1.0 / (self.fps * self.model.opt.timestep))
        self._n_substeps = max(1, n_physics_steps)

        self._joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        )
        self._actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES]
        )
        self._donut_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "donut")
        self._donut_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "donut_joint")
        self._box_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        self._box_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "box_center")
        self._box_grasp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "box_grasp_site")
        self._donut_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "donut_site")

        ctrl_low = np.array([self.model.actuator_ctrlrange[i, 0] for i in self._actuator_ids], dtype=np.float32)
        ctrl_high = np.array([self.model.actuator_ctrlrange[i, 1] for i in self._actuator_ids], dtype=np.float32)

        self.action_space = gym.spaces.Box(low=ctrl_low, high=ctrl_high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "pixels/top_camera": gym.spaces.Box(
                    0, 255, shape=(img_height, img_width, 3), dtype=np.uint8
                ),
                "pixels/front_camera": gym.spaces.Box(
                    0, 255, shape=(img_height, img_width, 3), dtype=np.uint8
                ),
                "pixels/left_wrist_camera": gym.spaces.Box(
                    0, 255, shape=(img_height, img_width, 3), dtype=np.uint8
                ),
                "pixels/right_wrist_camera": gym.spaces.Box(
                    0, 255, shape=(img_height, img_width, 3), dtype=np.uint8
                ),
                "agent_pos": gym.spaces.Box(-np.inf, np.inf, shape=(N_JOINTS,), dtype=np.float32),
            }
        )

        self._renderer = None
        self._step_count = 0
        self._donut_qpos_addr = self.model.jnt_qposadr[self._donut_joint_id]
        self._box_qpos_addr = self.model.jnt_qposadr[self._box_joint_id]

    def _get_joint_positions(self) -> np.ndarray:
        return np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids],
            dtype=np.float32,
        )

    def _render_camera(self, camera_name: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.img_height, self.img_width)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render().copy()

    def _get_obs(self) -> dict:
        return {
            "pixels/top_camera": self._render_camera("top_camera"),
            "pixels/front_camera": self._render_camera("front_camera"),
            "pixels/left_wrist_camera": self._render_camera("left_wrist_camera"),
            "pixels/right_wrist_camera": self._render_camera("right_wrist_camera"),
            "agent_pos": self._get_joint_positions(),
        }

    def _check_success(self) -> bool:
        donut_pos = self.data.site_xpos[self._donut_site_id]
        box_pos = self.data.site_xpos[self._box_site_id]
        dist = np.linalg.norm(donut_pos - box_pos)
        return bool(dist < self.success_threshold)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.np_random is not None:
            donut_x = self.np_random.uniform(*self.donut_x_range)
            donut_y = self.np_random.uniform(*self.donut_y_range)
        else:
            donut_x = 0.05
            donut_y = 0.0

        addr = self._donut_qpos_addr
        self.data.qpos[addr] = donut_x
        self.data.qpos[addr + 1] = donut_y
        self.data.qpos[addr + 2] = 0.44
        self.data.qpos[addr + 3] = 1.0
        self.data.qpos[addr + 4:addr + 7] = 0.0

        box_addr = self._box_qpos_addr
        self.data.qpos[box_addr] = -0.08
        self.data.qpos[box_addr + 1] = 0.0
        self.data.qpos[box_addr + 2] = 0.42
        self.data.qpos[box_addr + 3] = 1.0
        self.data.qpos[box_addr + 4:box_addr + 7] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        obs = self._get_obs()
        info = {"is_success": False}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[self._actuator_ids] = action

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        success = self._check_success()
        reward = 1.0 if success else 0.0
        terminated = success
        truncated = self._step_count >= self.max_episode_steps
        info = {"is_success": success}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_camera("top_camera")
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
