"""Automated demo collection with IK and motion planning for bimanual SO-101.

Generates demonstrations of the donut packing task using Jacobian-based inverse
kinematics and waypoint interpolation. Uses kinematic attachment (direct qpos
override) for reliable object holding during grasping/transport phases.
Saves to LeRobot v3.0 dataset format.

Task sequence (strictly sequential — one arm moves at a time):
  1. Left arm approaches box from the side, grasps wall edge
  2. Left arm lifts box to hold position
  3. Right arm approaches donut, grasps it
  4. Right arm lifts donut and places it in the held box
  5. Right arm retreats, left arm lowers box to table

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
LEFT_IK_JOINTS_NO_ROLL = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
]
RIGHT_IK_JOINTS = [
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
]
RIGHT_IK_JOINTS_NO_ROLL = [
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
]

GRIPPER_OPEN = 1.5
GRIPPER_CLOSED = -0.175


class JacobianIKSolver:
    """Damped least-squares IK for a single SO-101 arm (5 DOF, no gripper)."""

    def __init__(self, model, data, site_name, joint_names, damping=0.01):
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

    def solve(self, target_pos, q_init=None, max_iter=500, tol=0.001):
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
            env.model, env.data, "left_grasp_center", LEFT_IK_JOINTS
        )
        self.left_ik_4 = JacobianIKSolver(
            env.model, env.data, "left_grasp_center", LEFT_IK_JOINTS_NO_ROLL
        )
        self.right_ik = JacobianIKSolver(
            env.model, env.data, "right_grasp_center", RIGHT_IK_JOINTS
        )
        self.right_ik_4 = JacobianIKSolver(
            env.model, env.data, "right_grasp_center", RIGHT_IK_JOINTS_NO_ROLL
        )

        self._left_joint_indices = list(range(0, 5))
        self._left_gripper_index = 5
        self._right_joint_indices = list(range(6, 11))
        self._right_gripper_index = 11

        self._left_gripper_body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_body")
        self._right_gripper_body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_body")
        self._box_body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self._donut_body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "donut")
        self._box_joint_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        self._donut_joint_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "donut_joint")
        self._left_gc_site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_center")
        self._right_gc_site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_center")
        self._left_wrist_roll_addr = env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_roll")]
        self._right_wrist_roll_addr = env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_roll")]
        self._attachments = []
        self._saved_contacts = {}
        self._frozen_joints = {}

    def _get_current_action(self):
        return self.env._get_joint_positions().copy()

    def _step_physics(self, action):
        clipped = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        self.env.data.ctrl[self.env._actuator_ids] = clipped
        for _ in range(self.env._n_substeps):
            mujoco.mj_step(self.env.model, self.env.data)
            for i, jid in enumerate(self.env._joint_ids):
                if i == self._left_gripper_index or i == self._right_gripper_index:
                    continue
                addr = self.env.model.jnt_qposadr[jid]
                self.env.data.qpos[addr] = clipped[i]
                dof = self.env.model.jnt_dofadr[jid]
                self.env.data.qvel[dof] = 0
            if self._attachments:
                self._enforce_attachments()
            for jid, saved in self._frozen_joints.items():
                addr = self.env.model.jnt_qposadr[jid]
                self.env.data.qpos[addr:addr + 7] = saved
                dof_addr = self.env.model.jnt_dofadr[jid]
                self.env.data.qvel[dof_addr:dof_addr + 6] = 0
            mujoco.mj_forward(self.env.model, self.env.data)
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

    def _solve_left_oriented(self, target_pos, current_action, wrist_roll):
        """Solve left arm IK with wrist_roll fixed for proper finger orientation."""
        self.env.data.qpos[self._left_wrist_roll_addr] = wrist_roll
        q_init = np.array([current_action[i] for i in range(4)])
        q_solved = self.left_ik_4.solve(target_pos, q_init=q_init)
        new_action = current_action.copy()
        for i in range(4):
            new_action[i] = q_solved[i]
        new_action[4] = wrist_roll
        return new_action

    def _solve_right_oriented(self, target_pos, current_action, wrist_roll):
        """Solve right arm IK with wrist_roll fixed for proper finger orientation."""
        self.env.data.qpos[self._right_wrist_roll_addr] = wrist_roll
        q_init = np.array([current_action[i] for i in range(6, 10)])
        q_solved = self.right_ik_4.solve(target_pos, q_init=q_init)
        new_action = current_action.copy()
        for i in range(4):
            new_action[6 + i] = q_solved[i]
        new_action[10] = wrist_roll
        return new_action

    def _freeze_object(self, obj_joint_id):
        addr = self.env.model.jnt_qposadr[obj_joint_id]
        self._frozen_joints[obj_joint_id] = self.env.data.qpos[addr:addr + 7].copy()

    def _unfreeze_object(self, obj_joint_id):
        self._frozen_joints.pop(obj_joint_id, None)

    def _disable_body_contacts(self, body_id):
        """Disable collision for all geoms of a body (prevents physics fighting kinematic override)."""
        if body_id in self._saved_contacts:
            return
        saved = {}
        geom_start = self.env.model.body_geomadr[body_id]
        geom_count = self.env.model.body_geomnum[body_id]
        for i in range(geom_count):
            gid = geom_start + i
            saved[gid] = (int(self.env.model.geom_contype[gid]),
                          int(self.env.model.geom_conaffinity[gid]))
            self.env.model.geom_contype[gid] = 0
            self.env.model.geom_conaffinity[gid] = 0
        self._saved_contacts[body_id] = saved

    def _restore_body_contacts(self, body_id):
        """Restore collision for a body's geoms."""
        if body_id not in self._saved_contacts:
            return
        for gid, (ct, ca) in self._saved_contacts[body_id].items():
            self.env.model.geom_contype[gid] = ct
            self.env.model.geom_conaffinity[gid] = ca
        del self._saved_contacts[body_id]

    def _restore_all_contacts(self):
        """Restore all saved contact states."""
        for body_id in list(self._saved_contacts.keys()):
            self._restore_body_contacts(body_id)

    def _attach(self, gripper_body_id, obj_body_id, obj_joint_id,
                keep_upright=False, rel_pos_override=None):
        """Record relative pose for kinematic attachment and disable object contacts.

        rel_pos_override: when provided for keep_upright mode, uses this fixed
        world-frame offset instead of computing from simulation state. Prevents
        offset errors caused by servo tracking lag.
        """
        if keep_upright:
            if rel_pos_override is not None:
                rel_pos = np.array(rel_pos_override, dtype=np.float64)
            else:
                b2_pos = self.env.data.xpos[obj_body_id].copy()
                site_id = (self._left_gc_site_id if gripper_body_id == self._left_gripper_body_id
                           else self._right_gc_site_id)
                ref_pos = self.env.data.site_xpos[site_id].copy()
                rel_pos = b2_pos - ref_pos
            rel_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            b1_pos = self.env.data.xpos[gripper_body_id].copy()
            b2_pos = self.env.data.xpos[obj_body_id].copy()
            b1_mat = self.env.data.xmat[gripper_body_id].reshape(3, 3)
            b1_quat = np.zeros(4)
            mujoco.mju_mat2Quat(b1_quat, self.env.data.xmat[gripper_body_id])
            b2_quat = np.zeros(4)
            mujoco.mju_mat2Quat(b2_quat, self.env.data.xmat[obj_body_id])
            rel_pos = b1_mat.T @ (b2_pos - b1_pos)
            b1_quat_conj = b1_quat.copy()
            b1_quat_conj[1:] *= -1
            rel_quat = np.zeros(4)
            mujoco.mju_mulQuat(rel_quat, b1_quat_conj, b2_quat)

        self._disable_body_contacts(obj_body_id)
        self._attachments.append((gripper_body_id, obj_body_id, obj_joint_id,
                                  rel_pos, rel_quat, keep_upright))

    def _detach(self, obj_body_id):
        """Remove kinematic attachment and restore object contacts."""
        self._restore_body_contacts(obj_body_id)
        self._attachments = [a for a in self._attachments if a[1] != obj_body_id]

    def _enforce_attachments(self):
        """Set attached objects' qpos to maintain relative pose with gripper."""
        for gripper_body_id, obj_body_id, obj_joint_id, rel_pos, rel_quat, keep_upright in self._attachments:
            if keep_upright:
                site_id = (self._left_gc_site_id if gripper_body_id == self._left_gripper_body_id
                           else self._right_gc_site_id)
                ref_pos = self.env.data.site_xpos[site_id]
                world_pos = ref_pos + rel_pos
                world_quat = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                b1_pos = self.env.data.xpos[gripper_body_id]
                b1_mat = self.env.data.xmat[gripper_body_id].reshape(3, 3)
                b1_quat = np.zeros(4)
                mujoco.mju_mat2Quat(b1_quat, self.env.data.xmat[gripper_body_id])
                world_pos = b1_pos + b1_mat @ rel_pos
                world_quat = np.zeros(4)
                mujoco.mju_mulQuat(world_quat, b1_quat, rel_quat)

            addr = self.env.model.jnt_qposadr[obj_joint_id]
            self.env.data.qpos[addr:addr + 3] = world_pos
            self.env.data.qpos[addr + 3:addr + 7] = world_quat

            dof_addr = self.env.model.jnt_dofadr[obj_joint_id]
            self.env.data.qvel[dof_addr:dof_addr + 6] = 0

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
        self._attachments = []
        self._restore_all_contacts()
        self._frozen_joints = {}

        for _ in range(200):
            mujoco.mj_step(self.env.model, self.env.data)
        mujoco.mj_forward(self.env.model, self.env.data)

        initial_qpos = self.env.data.qpos.copy()
        initial_qvel = self.env.data.qvel.copy()

        donut_pos = self.env.data.site_xpos[self.env._donut_site_id].copy()
        box_grasp_pos = self.env.data.site_xpos[self.env._box_grasp_site_id].copy()

        box_rel_pos = np.array([0.0, -0.045, -0.055])
        donut_rel_pos = np.array([0.0, 0.0, 0.0])

        LEFT_ROLL = 1.57
        RIGHT_ROLL = -2.0

        states = []
        actions_list = []
        attach_events = []

        current_action = self._get_current_action()

        attach_events.append((len(actions_list), "freeze", self._box_joint_id))
        self._freeze_object(self._box_joint_id)

        # Phase 1: Left arm approach box from +y with fingers oriented to straddle wall
        pre_grasp_box = self._add_noise(box_grasp_pos + np.array([0, 0.05, 0]), rng)
        target_action = self._solve_left_oriented(pre_grasp_box, current_action, LEFT_ROLL)
        target_action = self._set_gripper(target_action, "left", GRIPPER_OPEN)
        n = self._noisy_steps(60, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 2: Left arm to box front wall rim
        grasp_box = self._add_noise(box_grasp_pos, rng)
        target_action = self._solve_left_oriented(grasp_box, current_action, LEFT_ROLL)
        target_action = self._set_gripper(target_action, "left", GRIPPER_OPEN)
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        attach_events.append((len(actions_list), "unfreeze", self._box_joint_id))
        self._unfreeze_object(self._box_joint_id)

        # Phase 3: Left arm close gripper + attach box
        target_action = self._set_gripper(current_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(15, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action
        attach_events.append((len(actions_list), "attach",
                              self._left_gripper_body_id, self._box_body_id,
                              self._box_joint_id, True, box_rel_pos))
        self._attach(self._left_gripper_body_id, self._box_body_id,
                     self._box_joint_id, keep_upright=True, rel_pos_override=box_rel_pos)

        # Phase 4: Left arm lifts box to hold position
        hold_pos = np.array([0.0, 0.0, 0.50])
        hold_pos = self._add_noise(hold_pos, rng)
        target_action = self._solve_left_oriented(hold_pos, current_action, LEFT_ROLL)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(60, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        box_held_center = self.env.data.site_xpos[self.env._box_site_id].copy()

        attach_events.append((len(actions_list), "freeze", self._donut_joint_id))
        self._freeze_object(self._donut_joint_id)

        # Phase 5a: Right arm to high intermediate (above table level)
        intermediate = self._add_noise(np.array([0.10, -0.10, 0.52]), rng)
        target_action = self._solve_right(intermediate, current_action)
        target_action = self._set_gripper(target_action, "right", GRIPPER_OPEN)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(60, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 5b: Right arm pre-grasp above donut with horizontal finger orientation
        pre_grasp_donut = self._add_noise(donut_pos + np.array([0, 0, 0.05]), rng)
        target_action = self._solve_right_oriented(pre_grasp_donut, current_action, RIGHT_ROLL)
        target_action = self._set_gripper(target_action, "right", GRIPPER_OPEN)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(60, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 6: Right arm descend to donut (raised 8mm to keep fingers above table)
        grasp_donut = self._add_noise(donut_pos, rng)
        grasp_donut[2] = donut_pos[2] + 0.008
        target_action = self._solve_right_oriented(grasp_donut, current_action, RIGHT_ROLL)
        target_action = self._set_gripper(target_action, "right", GRIPPER_OPEN)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(30, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        attach_events.append((len(actions_list), "unfreeze", self._donut_joint_id))
        self._unfreeze_object(self._donut_joint_id)

        # Phase 7: Right arm close gripper + attach donut
        target_action = self._set_gripper(current_action, "right", GRIPPER_CLOSED)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(15, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action
        attach_events.append((len(actions_list), "attach",
                              self._right_gripper_body_id, self._donut_body_id,
                              self._donut_joint_id, True, donut_rel_pos))
        self._attach(self._right_gripper_body_id, self._donut_body_id,
                     self._donut_joint_id, keep_upright=True, rel_pos_override=donut_rel_pos)

        # Phase 7.5: Lift donut straight up (clear of table before lateral move)
        lift_pos = np.array([donut_pos[0], donut_pos[1], 0.55])
        target_action = self._solve_right_oriented(lift_pos, current_action, RIGHT_ROLL)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        target_action = self._set_gripper(target_action, "right", GRIPPER_CLOSED)
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 8: Right arm moves to above box opening
        above_box = box_held_center + np.array([0, 0, 0.08])
        target_action = self._solve_right_oriented(above_box, current_action, RIGHT_ROLL)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        target_action = self._set_gripper(target_action, "right", GRIPPER_CLOSED)
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 9: Right arm lowers donut into box center
        into_box = box_held_center.copy()
        target_action = self._solve_right_oriented(into_box, current_action, RIGHT_ROLL)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        target_action = self._set_gripper(target_action, "right", GRIPPER_CLOSED)
        n = self._noisy_steps(30, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 10: Detach donut from right, re-attach to left arm
        attach_events.append((len(actions_list), "detach", self._donut_body_id))
        self._detach(self._donut_body_id)
        donut_to_left_offset = box_held_center - hold_pos
        attach_events.append((len(actions_list), "attach",
                              self._left_gripper_body_id, self._donut_body_id,
                              self._donut_joint_id, True, donut_to_left_offset))
        self._attach(self._left_gripper_body_id, self._donut_body_id,
                     self._donut_joint_id, keep_upright=True,
                     rel_pos_override=donut_to_left_offset)

        target_action = self._set_gripper(current_action, "right", GRIPPER_OPEN)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(15, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 10.5: Right arm lifts above box before retreating
        retreat_up = np.array([0.05, -0.05, 0.55])
        target_action = self._solve_right(retreat_up, current_action)
        target_action = self._set_gripper(target_action, "right", GRIPPER_OPEN)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(30, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 11: Right arm retreat to home
        target_action = current_action.copy()
        for idx in self._right_joint_indices:
            target_action[idx] = 0.0
        target_action[self._right_gripper_index] = GRIPPER_OPEN
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 12: Left arm lowers box+donut to table
        table_pos = self._add_noise(box_grasp_pos.copy(), rng)
        target_action = self._solve_left_oriented(table_pos, current_action, LEFT_ROLL)
        target_action = self._set_gripper(target_action, "left", GRIPPER_CLOSED)
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        attach_events.append((len(actions_list), "detach", self._donut_body_id))
        self._detach(self._donut_body_id)
        attach_events.append((len(actions_list), "detach", self._box_body_id))
        self._detach(self._box_body_id)
        target_action = self._set_gripper(current_action, "left", GRIPPER_OPEN)
        n = self._noisy_steps(15, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)
        current_action = target_action

        # Phase 13: Left arm return to home
        target_action = current_action.copy()
        for idx in self._left_joint_indices:
            target_action[idx] = 0.0
        target_action[self._left_gripper_index] = GRIPPER_OPEN
        n = self._noisy_steps(40, rng)
        self._interpolate_and_step(current_action, target_action, n, states, actions_list)

        return initial_qpos, initial_qvel, states, actions_list, attach_events

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

    def _save_episode(self, dataset, initial_qpos, initial_qvel, states, actions_list,
                      attach_events):
        n_frames = len(actions_list)

        self.env.data.qpos[:] = initial_qpos
        self.env.data.qvel[:] = initial_qvel
        self._attachments = []
        self._restore_all_contacts()
        mujoco.mj_forward(self.env.model, self.env.data)

        self._frozen_joints = {}
        event_idx = 0

        for i in range(n_frames):
            while event_idx < len(attach_events) and attach_events[event_idx][0] == i:
                ev = attach_events[event_idx]
                if ev[1] == "attach":
                    keep_upright = ev[5] if len(ev) > 5 else False
                    rel_override = ev[6] if len(ev) > 6 else None
                    self._attach(ev[2], ev[3], ev[4],
                                 keep_upright=keep_upright,
                                 rel_pos_override=rel_override)
                elif ev[1] == "detach":
                    self._detach(ev[2])
                elif ev[1] == "freeze":
                    self._freeze_object(ev[2])
                elif ev[1] == "unfreeze":
                    self._unfreeze_object(ev[2])
                event_idx += 1

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

            self._step_physics(actions_list[i])

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

            initial_qpos, initial_qvel, states, actions_list, attach_events = self._generate_episode(ep_seed)

            print(f"  Generated {len(actions_list)} frames in {time.time() - t_start:.1f}s")
            print("  Rendering and saving...")

            t_render = time.time()
            self._save_episode(dataset, initial_qpos, initial_qvel, states, actions_list,
                               attach_events)

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
