"""Quick visual check: generate trajectory and render key frames only."""
import os
os.environ['MUJOCO_GL'] = 'osmesa'

import sys
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from bi_so101_env import BiSO101Env
from auto_collect_demos import AutoCollector

env = BiSO101Env(fps=30, max_episode_steps=10**9)
collector = AutoCollector(
    env, "local/test", "/tmp/test_dataset", "test",
    noise_scale=0.0, n_episodes=1, seed=0,
)

print("Generating trajectory...")
initial_qpos, initial_qvel, states, actions_list, attach_events = collector._generate_episode(0)
n_frames = len(actions_list)
print(f"Generated {n_frames} frames")

# Key frame indices to render (approximate phase boundaries)
key_frames = {
    0: "initial",
    60: "phase1_approach",
    100: "phase2_at_wall",
    115: "phase3_gripper_closed",
    175: "phase4_box_lifted",
    235: "phase5a_right_intermediate",
    295: "phase5b_right_pregrasp",
    325: "phase6_right_at_donut",
    340: "phase7_donut_grasped",
    380: "phase7.5_donut_lifted",
    420: "phase8_above_box",
    450: "phase9_donut_in_box",
    465: "phase10_donut_released",
    495: "phase10.5_right_retreating",
    535: "phase12_lowering",
    570: "phase12_at_table",
}

# Replay and render key frames
env.data.qpos[:] = initial_qpos
env.data.qvel[:] = initial_qvel
collector._attachments = []
collector._restore_all_contacts()
collector._frozen_joints = {}
mujoco.mj_forward(env.model, env.data)

event_idx = 0
output_dir = "/tmp/visual_check"
os.makedirs(output_dir, exist_ok=True)

renderer = mujoco.Renderer(env.model, 480, 640)

for i in range(n_frames):
    while event_idx < len(attach_events) and attach_events[event_idx][0] == i:
        ev = attach_events[event_idx]
        if ev[1] == "attach":
            keep_upright = ev[5] if len(ev) > 5 else False
            rel_override = ev[6] if len(ev) > 6 else None
            collector._attach(ev[2], ev[3], ev[4],
                             keep_upright=keep_upright,
                             rel_pos_override=rel_override)
        elif ev[1] == "detach":
            collector._detach(ev[2])
        elif ev[1] == "freeze":
            collector._freeze_object(ev[2])
        elif ev[1] == "unfreeze":
            collector._unfreeze_object(ev[2])
        event_idx += 1

    if i in key_frames:
        label = key_frames[i]
        for cam_name in ["front_camera", "top_camera"]:
            renderer.update_scene(env.data, camera=cam_name)
            img = renderer.render().copy()
            path = os.path.join(output_dir, f"{i:04d}_{label}_{cam_name}.png")
            Image.fromarray(img).save(path)
        print(f"  Frame {i}: {label}")

        # Print diagnostic info
        gc_left = env.data.site_xpos[collector._left_gc_site_id]
        gc_right = env.data.site_xpos[collector._right_gc_site_id]
        box_pos = env.data.xpos[collector._box_body_id]
        donut_pos = env.data.xpos[collector._donut_body_id]
        print(f"    left_gc={gc_left}, right_gc={gc_right}")
        print(f"    box_body={box_pos}, donut_body={donut_pos}")

    collector._step_physics(actions_list[i])

renderer.close()
env.close()
print(f"\nRendered {len(key_frames)} key frames to {output_dir}")
