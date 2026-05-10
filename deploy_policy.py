"""Deploy a trained ACT policy in the bimanual SO-101 MuJoCo simulation.

Usage:
    python deploy_policy.py --checkpoint outputs/train/act_bi_so101_donut/checkpoints/last/pretrained_model
    python deploy_policy.py --checkpoint outputs/train/act_bi_so101_donut/checkpoints/last/pretrained_model --headless --n-episodes 50
"""

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(__file__))
from bi_so101_env import BiSO101Env


def load_policy(checkpoint_path: str, device: str = "cuda"):
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    return policy


def obs_to_batch(obs: dict, device: str = "cuda") -> dict:
    state = torch.from_numpy(obs["agent_pos"]).unsqueeze(0).to(device)
    top = torch.from_numpy(obs["pixels/top_camera"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    front = torch.from_numpy(obs["pixels/front_camera"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return {
        "observation.state": state.to(device),
        "observation.images.top_camera": top.to(device),
        "observation.images.front_camera": front.to(device),
    }


def run_headless(policy, env: BiSO101Env, n_episodes: int, device: str):
    successes = 0
    total_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        policy.reset()
        episode_reward = 0.0

        for step in range(env.max_episode_steps):
            batch = obs_to_batch(obs, device)
            with torch.no_grad():
                action = policy.select_action(batch)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_reward += reward

            if terminated or truncated:
                break

        success = info["is_success"]
        successes += int(success)
        total_rewards.append(episode_reward)
        status = "SUCCESS" if success else "FAIL"
        print(f"Episode {ep+1}/{n_episodes}: {status} (reward={episode_reward:.1f}, steps={step+1})")

    print(f"\nResults: {successes}/{n_episodes} successes ({100*successes/n_episodes:.1f}%)")
    print(f"Mean reward: {np.mean(total_rewards):.2f}")


def run_viewer(policy, env: BiSO101Env, device: str):
    obs, _ = env.reset(seed=0)
    policy.reset()
    episode_count = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            batch = obs_to_batch(obs, device)
            with torch.no_grad():
                action = policy.select_action(batch)
            action_np = action.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action_np)
            viewer.sync()

            if terminated or truncated:
                episode_count += 1
                status = "SUCCESS" if info["is_success"] else "timeout"
                print(f"Episode {episode_count} ended: {status}")
                obs, _ = env.reset()
                policy.reset()

            elapsed = time.time() - step_start
            sleep_time = (1.0 / env.fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Deploy ACT policy in bimanual SO-101 sim")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained model directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--headless", action="store_true", help="Run without viewer for batch eval")
    parser.add_argument("--n-episodes", type=int, default=50, help="Number of eval episodes (headless)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    env = BiSO101Env(fps=args.fps, max_episode_steps=args.max_steps)
    policy = load_policy(args.checkpoint, args.device)
    print(f"Loaded policy from {args.checkpoint}")
    print(f"Action dim: {policy.config.action_feature.shape}")

    if args.headless:
        run_headless(policy, env, args.n_episodes, args.device)
    else:
        run_viewer(policy, env, args.device)


if __name__ == "__main__":
    main()
