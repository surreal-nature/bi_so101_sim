# Bimanual SO-101 Donut Packing Simulation

A complete simulation-to-real pipeline for training a bimanual (two-arm) robotic manipulation policy using two [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot arms in MuJoCo. The task: one arm picks up a donut from the table and places it into a box held/positioned near the other arm.

This project is designed to produce a trained ACT (Action Chunking with Transformers) policy in simulation that can later be transferred to physical SO-101 hardware via fine-tuning on a small number of real demonstrations.

---

## Table of Contents

- [Motivation and Background](#motivation-and-background)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
  - [MuJoCo Scene](#mujoco-scene)
  - [Gymnasium Environment](#gymnasium-environment)
  - [Keyboard Teleoperation Collector](#keyboard-teleoperation-collector)
  - [Training Pipeline](#training-pipeline)
  - [Policy Deployment](#policy-deployment)
- [Usage Guide](#usage-guide)
  - [Step 1: Verify the Scene](#step-1-verify-the-scene)
  - [Step 2: Collect Demonstrations](#step-2-collect-demonstrations)
  - [Step 3: Train the ACT Policy](#step-3-train-the-act-policy)
  - [Step 4: Evaluate the Policy](#step-4-evaluate-the-policy)
- [Technical Details](#technical-details)
  - [SO-101 Kinematics and Joint Configuration](#so-101-kinematics-and-joint-configuration)
  - [Physics Parameters](#physics-parameters)
  - [Observation and Action Spaces](#observation-and-action-spaces)
  - [Dataset Format](#dataset-format)
  - [ACT Policy Configuration](#act-policy-configuration)
- [Sim-to-Real Transfer](#sim-to-real-transfer)
- [Troubleshooting](#troubleshooting)

---

## Motivation and Background

Long-horizon bimanual manipulation (two arms coordinating to achieve a goal) is significantly more complex than single-arm tasks. Training such policies requires:

1. **A large number of demonstrations** -- impractical to collect entirely on physical hardware before the robot arrives.
2. **Repeatable environments** -- physics simulation allows deterministic resets and randomization.
3. **Accurate kinematics** -- the simulation must match the real robot's joint structure, limits, and motor dynamics for sim-to-real transfer.

This project addresses all three by:

- Using **MuJoCo** with the official SO-101 MJCF model files (STL meshes, calibrated joint properties, STS3215 motor parameters) from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100).
- Providing a **keyboard teleoperation** interface to collect demonstrations interactively in simulation.
- Recording data directly into the **LeRobot v3.0 dataset format**, making it immediately compatible with `lerobot-train` for ACT policy training.
- Naming all joints and features to match LeRobot's built-in `bi_so_follower` robot configuration (`left_shoulder_pan.pos`, `right_gripper.pos`, etc.), so a sim-trained policy can be deployed on physical hardware with minimal adaptation.

The starting task is intentionally simple -- a single donut pick-and-place -- to validate the full pipeline before scaling to more complex long-horizon tasks (multi-donut packing, sequential assembly, etc.).

---

## Project Structure

```
bi_so101_sim/
├── assets/
│   ├── scene_bi_so101_donut.xml       # Bimanual MuJoCo scene definition
│   └── so101/                         # Official SO-101 assets (from TheRobotStudio)
│       ├── so101_new_calib.xml        #   Reference single-arm MJCF (for comparison)
│       ├── joints_properties.xml      #   STS3215 motor defaults
│       ├── scene.xml                  #   Reference single-arm scene
│       ├── *.stl                      #   13 mesh files (visual + collision geometry)
│       └── *.part                     #   CAD source files
├── bi_so101_env.py                    # Gymnasium environment wrapper
├── collect_demos.py                   # Keyboard teleoperation + LeRobot dataset recorder
├── deploy_policy.py                   # Policy deployment (viewer + headless evaluation)
├── train.sh                           # lerobot-train wrapper script (SLURM-compatible)
├── data/                              # (created at runtime) Recorded datasets
│   └── local/bi_so101_donut_packing/  #   Default dataset location
├── outputs/                           # (created at runtime) Training outputs
│   └── train/act_bi_so101_donut/      #   Default training output
└── README.md                          # This file
```

No files inside the `lerobot/` repository are modified. Everything is standalone.

---

## Prerequisites

- **Python 3.12+**
- **Conda environment** with LeRobot installed from source (see [LeRobot installation](https://github.com/huggingface/lerobot))
- **GPU** recommended for training (AMD MI300X, NVIDIA A100/H100, etc.)
- **Display server** (X11/Wayland) required for the MuJoCo viewer during teleoperation and visual evaluation. Headless evaluation works without a display via EGL.

### Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `mujoco` | >= 3.0 | Physics simulation and rendering |
| `gymnasium` | >= 0.29 | Standard RL environment interface |
| `numpy` | >= 1.24 | Numerical operations |
| `torch` | >= 2.0 | Policy inference and training |
| `lerobot` | >= 0.5.0 | Dataset format, ACT policy, training pipeline |

---

## Installation

```bash
# 1. Activate your LeRobot conda environment
conda activate act

# 2. Install MuJoCo (if not already installed)
pip install mujoco

# 3. Clone the official SO-ARM100 repo (if assets/so101/ is empty)
git clone https://github.com/TheRobotStudio/SO-ARM100.git /tmp/SO-ARM100
cp /tmp/SO-ARM100/Simulation/SO101/*.stl bi_so101_sim/assets/so101/
cp /tmp/SO-ARM100/Simulation/SO101/*.xml bi_so101_sim/assets/so101/

# 4. Verify the scene loads
python -c "
import os; os.environ['MUJOCO_GL']='egl'
import mujoco
m = mujoco.MjModel.from_xml_path('bi_so101_sim/assets/scene_bi_so101_donut.xml')
print(f'Scene loaded: {m.njnt} joints, {m.nu} actuators, {m.ngeom} geoms')
"
# Expected output: Scene loaded: 13 joints, 12 actuators, 76 geoms
```

---

## Architecture Overview

### MuJoCo Scene

**File:** `assets/scene_bi_so101_donut.xml`

The scene places two SO-101 arms on opposite sides of a table, facing each other, with a graspable donut and an open-top box on the table surface.

#### Physical Layout (top-down view)

```
         Y
         ^
         |
   +-----+-----+
   |             |        Table: 0.6m x 0.5m x 0.04m (at z=0.40m)
   |    [BOX]   |        Box:   0.1m x 0.1m x 0.06m (at x=-0.12, z=0.42)
   |             |        Donut: r=2.5cm, h=2.4cm   (at x=+0.08, z=0.44)
   |      (donut)|
   |             |
   +-----+-----+----> X
  LEFT          RIGHT
  ARM           ARM
  (blue)        (orange)
  x=-0.22       x=+0.22
```

#### Scene Components

| Component | Description | Position (m) |
|-----------|-------------|-------------|
| **Left arm** | SO-101, blue material, faces right (+X) | (-0.22, 0, 0.42) |
| **Right arm** | SO-101, orange material, faces left (-X) | (+0.22, 0, 0.42) |
| **Table** | Wooden surface with 4 legs | (0, 0, 0.40) |
| **Donut** | Cylinder (r=2.5cm, h=2.4cm), 50g, freejoint | (+0.08, 0, 0.44) |
| **Box** | Open-top container (10cm x 10cm x 6cm), 4 walls + bottom | (-0.12, 0, 0.42) |
| **Top camera** | Bird's-eye view, 480x640 (observation) | (0, -0.1, 1.2) |
| **Front camera** | Angled front view, 480x640 (observation) | (0, -0.6, 0.7) |

The donut uses a `freejoint` so it can be pushed, picked up, and dropped by the arms. Both arms have additional **fingertip collision boxes** added to the gripper for reliable grasping (the original SO-101 mesh-only collision was too coarse for small object manipulation).

#### Arm Orientations

Both arms are mounted at the table edge and rotated 90 degrees to face the table center:

- **Left arm:** quaternion `(0.707, 0, 0, 0.707)` -- rotated +90 degrees around Z, faces right (+X)
- **Right arm:** quaternion `(0.707, 0, 0, -0.707)` -- rotated -90 degrees around Z, faces left (-X)

This arrangement allows both arms to reach the center of the table where the donut and box are placed.

---

### Gymnasium Environment

**File:** `bi_so101_env.py`

The `BiSO101Env` class wraps the MuJoCo scene as a standard [Gymnasium](https://gymnasium.farama.org/) environment with the following interface:

#### Observation Space (Dict)

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `pixels/top_camera` | (480, 640, 3) | uint8 | RGB image from the bird's-eye camera |
| `pixels/front_camera` | (480, 640, 3) | uint8 | RGB image from the angled front camera |
| `agent_pos` | (12,) | float32 | Current joint positions of both arms |

The `agent_pos` vector contains 12 joint angles in the following order:

```
Index  Joint Name              Arm     Range (rad)        Range (deg)
-----  ----------------------  ------  -----------------  ----------------
  0    left_shoulder_pan       Left    [-1.920, +1.920]   [-110.0, +110.0]
  1    left_shoulder_lift      Left    [-1.745, +1.745]   [-100.0, +100.0]
  2    left_elbow_flex         Left    [-1.690, +1.690]   [ -96.8,  +96.8]
  3    left_wrist_flex         Left    [-1.658, +1.658]   [ -95.0,  +95.0]
  4    left_wrist_roll         Left    [-2.744, +2.841]   [-157.2, +162.8]
  5    left_gripper            Left    [-0.175, +1.745]   [ -10.0, +100.0]
  6    right_shoulder_pan      Right   [-1.920, +1.920]   [-110.0, +110.0]
  7    right_shoulder_lift     Right   [-1.745, +1.745]   [-100.0, +100.0]
  8    right_elbow_flex        Right   [-1.690, +1.690]   [ -96.8,  +96.8]
  9    right_wrist_flex        Right   [-1.658, +1.658]   [ -95.0,  +95.0]
 10    right_wrist_roll        Right   [-2.744, +2.841]   [-157.2, +162.8]
 11    right_gripper           Right   [-0.175, +1.745]   [ -10.0, +100.0]
```

#### Action Space

| Shape | Type | Description |
|-------|------|-------------|
| (12,) | float32 | Joint position targets (radians) for all 12 actuators |

Actions are **absolute joint position targets**, not deltas. The MuJoCo position actuators (PD controllers) drive each joint to the commanded target angle. Actions are automatically clipped to each joint's valid control range.

#### Episode Structure

| Parameter | Value | Description |
|-----------|-------|-------------|
| FPS | 30 | Control frequency |
| Max steps | 300 | Maximum steps per episode (10 seconds) |
| Physics timestep | 0.002s | MuJoCo simulation timestep |
| Substeps | 16 | Physics steps per control step (1/30 / 0.002) |
| Success threshold | 0.03m | Donut must be within 3cm of box center |

#### Reset Behavior

On each `reset()`, the arms return to their default (zero-angle) pose and the donut is placed at a random position on the table:
- X position: uniform random in `[0.0, 0.15]` meters (right half of table)
- Y position: uniform random in `[-0.08, 0.08]` meters (near center)

This randomization encourages the policy to generalize across different initial donut placements.

#### Reward and Termination

- **Reward:** Sparse binary. `1.0` if the donut center is within `success_threshold` (3cm) of the box center, `0.0` otherwise.
- **Terminated:** `True` when success is achieved.
- **Truncated:** `True` when `max_episode_steps` is reached without success.

---

### Keyboard Teleoperation Collector

**File:** `collect_demos.py`

An interactive demonstration collection tool that opens the MuJoCo viewer window and allows you to control both arms simultaneously using the keyboard. Each recorded frame is immediately written into the LeRobot v3.0 dataset format.

#### Keyboard Layout

The left hand (QWEASD region) controls the left arm, and the right hand (UIOJKL region) controls the right arm:

```
LEFT ARM (blue)                          RIGHT ARM (orange)
+-----------+----------+--------+        +-----------+----------+--------+
| Joint     | Key +    | Key -  |        | Joint     | Key +    | Key -  |
+-----------+----------+--------+        +-----------+----------+--------+
| shldr_pan | E        | Q      |        | shldr_pan | O        | U      |
| shldr_lft | W        | S      |        | shldr_lft | I        | K      |
| elbow_flx | D        | A      |        | elbow_flx | L        | J      |
| wrist_flx | R        | F      |        | wrist_flx | Y        | H      |
| wrist_rll | T        | G      |        | wrist_rll | P        | ;      |
| gripper   | Space (toggle)    |        | gripper   | Enter (toggle)    |
+-----------+----------+--------+        +-----------+----------+--------+
```

Each key press adds `+/- 0.05 radians` (~2.9 degrees) per frame to the target joint angle. Multiple keys can be held simultaneously for diagonal/compound motions.

Gripper controls are **toggle-based**: press Space once to close the left gripper, press again to open it.

#### Session Controls

| Key | Action |
|-----|--------|
| **X** | Save current episode and start a new one |
| **Z** | Discard current episode (no save) and reset |
| **Esc** | Finish recording and finalize the dataset |

#### How It Records

Each control step at 30fps records one frame containing:

```python
{
    "observation.state":              # (12,) float32 - current joint angles
    "observation.images.top_camera":  # (480, 640, 3) uint8 - bird's-eye camera
    "observation.images.front_camera":# (480, 640, 3) uint8 - angled front camera
    "action":                         # (12,) float32 - commanded joint targets
    "task":                           # str - task description
}
```

Frames accumulate in memory during an episode. When you press **X**, the episode is flushed to disk via `LeRobotDataset.add_frame()` and `save_episode()`. Video encoding (AV1 codec) happens automatically. When you press **Esc** (or close the window), the dataset is finalized.

---

### Training Pipeline

**File:** `train.sh`

A bash wrapper around `lerobot-train` configured for ACT policy training on the bimanual dataset. It is SLURM-compatible (has `#SBATCH` directives) and can also be run directly.

#### Default Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `policy.type` | `act` | Action Chunking Transformer - handles multi-modal bimanual control |
| `policy.chunk_size` | 50 | Predict 50 future actions at once (~1.67s lookahead at 30fps) |
| `policy.n_action_steps` | 50 | Execute all 50 predicted actions before re-querying |
| `policy.vision_backbone` | `resnet18` | Lightweight CNN for image encoding |
| `policy.dim_model` | 512 | Transformer hidden dimension |
| `policy.use_vae` | `true` | Variational autoencoder for action diversity |
| `steps` | 30000 | Total training steps |
| `batch_size` | 8 | Per-GPU batch size |
| `eval_freq` | 5000 | Evaluate every 5K steps |
| `save_freq` | 5000 | Checkpoint every 5K steps |
| `dataset.image_transforms.enable` | `true` | Random crops, color jitter for augmentation |

These can be overridden via environment variables or CLI arguments:

```bash
# Override via environment variables
STEPS=50000 BATCH_SIZE=16 bash train.sh

# Override via CLI arguments (passed through to lerobot-train)
bash train.sh --steps 50000 --batch_size 16
```

#### Why ACT for Bimanual Control?

ACT is particularly well-suited for this task because:

1. **Action chunking** captures temporal correlations in bimanual coordination (both arms must move in sync).
2. **VAE latent space** models the multi-modality of demonstrations (there are many valid ways to pick up a donut).
3. **Arbitrary action dimensions** -- ACT dynamically sizes its action head based on `config.action_feature.shape[0]`, so 12-DOF bimanual actions work without any code changes to the policy.
4. **Vision + proprioception** -- ACT natively fuses image features (from the ResNet backbone) with state features (joint angles), which is critical for precise manipulation.

---

### Policy Deployment

**File:** `deploy_policy.py`

Loads a trained ACT checkpoint and runs it in the simulation, either interactively in the MuJoCo viewer or headlessly for batch evaluation.

#### Observation Processing

The deployment script converts raw environment observations to the format expected by the trained policy:

```python
# Environment provides:
obs["agent_pos"]           # (12,) float32 numpy array
obs["pixels/top_camera"]   # (480, 640, 3) uint8 numpy array
obs["pixels/front_camera"] # (480, 640, 3) uint8 numpy array

# Policy expects:
batch["observation.state"]                # (1, 12) float32 tensor
batch["observation.images.top_camera"]    # (1, 3, 480, 640) float32 tensor in [0, 1]
batch["observation.images.front_camera"]  # (1, 3, 480, 640) float32 tensor in [0, 1]
```

Each image is permuted from HWC to CHW format and normalized from [0, 255] to [0.0, 1.0].

#### Viewer Mode (default)

Opens the MuJoCo viewer and runs the policy in real-time at 30fps. Useful for visual debugging and qualitative evaluation. Automatically resets on episode end.

#### Headless Mode

Runs N episodes without rendering and reports success statistics:

```
Episode 1/50: SUCCESS (reward=1.0, steps=187)
Episode 2/50: FAIL (reward=0.0, steps=300)
...
Results: 38/50 successes (76.0%)
Mean reward: 0.76
```

---

## Usage Guide

### Step 1: Verify the Scene

Confirm the MuJoCo scene loads and renders correctly:

```bash
cd bi_so101_sim

# Basic scene verification
python -c "
import os; os.environ['MUJOCO_GL']='egl'
import mujoco
m = mujoco.MjModel.from_xml_path('assets/scene_bi_so101_donut.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print(f'Joints: {m.njnt}, Actuators: {m.nu}, Geoms: {m.ngeom}')
r = mujoco.Renderer(m, 480, 640)
r.update_scene(d, camera='top_camera')
img = r.render()
print(f'Camera renders: {img.shape}, mean pixel: {img.mean():.1f}')
r.close()
"
```

Expected output:
```
Joints: 13, Actuators: 12, Geoms: 76
Camera renders: (480, 640, 3), mean pixel: 109.8
```

To view the scene interactively (requires display):

```bash
python -c "
import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path('assets/scene_bi_so101_donut.xml')
d = mujoco.MjData(m)
mujoco.viewer.launch(m, d)
"
```

### Step 2: Collect Demonstrations

Run the teleoperation collector. **Requires a display** (X11/Wayland) for the MuJoCo viewer.

```bash
python collect_demos.py
```

With custom options:

```bash
python collect_demos.py \
    --dataset-name "myuser/bi_so101_donut_v1" \
    --dataset-root ./data/my_dataset \
    --task "Pick up the donut and place it in the box." \
    --fps 30 \
    --max-steps 300
```

#### Tips for Good Demonstrations

- **Aim for 50+ episodes**. More diverse demonstrations produce more robust policies. 50 episodes at ~10 seconds each takes roughly 15-20 minutes with practice.
- **Be consistent in strategy.** Use the same general approach each time (e.g., always pick with the right arm, always approach from the same side). ACT learns from the distribution of demonstrations, so consistency reduces multi-modality.
- **Use slow, deliberate motions.** The policy must learn smooth trajectories. Jerky keyboard inputs produce noisy training data.
- **Vary the approach slightly.** While being consistent in strategy, allow natural variation in exact trajectories. This teaches the policy to generalize.
- **Discard failed episodes (Z).** Only save episodes where the task was actually completed successfully.
- **Save immediately after success (X).** Don't let the episode run to timeout after the donut is placed.

### Step 3: Train the ACT Policy

```bash
# Direct execution
bash train.sh

# SLURM submission
sbatch train.sh

# With overrides
STEPS=50000 BATCH_SIZE=16 bash train.sh
```

Training on 50 episodes (~15K frames) for 30K steps takes approximately:
- ~1 hour on an NVIDIA A100 (80GB)
- ~45 minutes on an AMD MI300X
- ~3 hours on an NVIDIA RTX 3090

Monitor training progress via the log output:
```
step: 100, loss: 0.4523, l1_loss: 0.4102, kld_loss: 0.0421, lr: 1.0e-05
step: 200, loss: 0.3891, l1_loss: 0.3512, kld_loss: 0.0379, lr: 1.0e-05
...
```

Key metrics to watch:
- **l1_loss** should steadily decrease. If it plateaus early, try increasing `dim_model` or `chunk_size`.
- **kld_loss** should stabilize around 0.01-0.05. If it collapses to 0, the VAE is not learning useful latent structure.
- **Overfitting** typically appears after 6K-10K steps on small datasets (as observed in prior gesture mimic experiments). Use checkpoints from `save_freq` intervals to find the best trade-off.

### Step 4: Evaluate the Policy

#### Interactive Evaluation (with display)

```bash
python deploy_policy.py \
    --checkpoint outputs/train/act_bi_so101_donut/checkpoints/last/pretrained_model
```

#### Headless Batch Evaluation

```bash
python deploy_policy.py \
    --checkpoint outputs/train/act_bi_so101_donut/checkpoints/last/pretrained_model \
    --headless \
    --n-episodes 50 \
    --device cuda
```

#### Evaluating Different Checkpoints

Training saves checkpoints every 5K steps. Compare them to find the best one:

```bash
for step in 005000 010000 015000 020000 025000 030000; do
    echo "=== Checkpoint $step ==="
    python deploy_policy.py \
        --checkpoint outputs/train/act_bi_so101_donut/checkpoints/$step/pretrained_model \
        --headless --n-episodes 20
done
```

---

## Technical Details

### SO-101 Kinematics and Joint Configuration

Each SO-101 arm has **6 degrees of freedom** driven by Feetech STS3215 serial bus servos:

| # | Joint | DOF Type | Function |
|---|-------|----------|----------|
| 1 | `shoulder_pan` | Revolute (Z-axis) | Base rotation (left/right swing) |
| 2 | `shoulder_lift` | Revolute (Z-axis) | Shoulder elevation (up/down) |
| 3 | `elbow_flex` | Revolute (Z-axis) | Elbow bend |
| 4 | `wrist_flex` | Revolute (Z-axis) | Wrist pitch |
| 5 | `wrist_roll` | Revolute (Z-axis) | Wrist rotation |
| 6 | `gripper` | Revolute (Z-axis) | Gripper open/close |

The kinematic chain is: `base -> shoulder -> upper_arm -> lower_arm -> wrist -> gripper_body -> moving_jaw`

All joint axes are Z-axis in their local frame, with rotations accumulated through the body hierarchy.

### Physics Parameters

The STS3215 motor model uses position-controlled actuators with the following parameters from the official MJCF:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `kp` | 998.22 | Position gain (proportional) |
| `kv` | 2.731 | Velocity gain (derivative/damping) |
| `forcerange` | [-2.94, 2.94] Nm | Maximum actuator torque |
| `damping` | 0.60 | Joint viscous friction |
| `frictionloss` | 0.052 | Joint static/dry friction |
| `armature` | 0.028 | Rotor inertia |

These values were calibrated by TheRobotStudio to match the physical STS3215 servos, ensuring realistic motor dynamics in simulation.

Additional physics settings:
| Parameter | Value |
|-----------|-------|
| Global timestep | 0.002s (500 Hz) |
| Gravity | (0, 0, -9.81) m/s^2 |
| Donut friction | 0.8 (lateral) |
| Fingertip friction | 1.0 (lateral) |
| Contact dimensionality | 4 (tangent + torsion) |

### Observation and Action Spaces

#### Observations

The environment provides three observation modalities:

1. **Top camera** (`pixels/top_camera`): A 480x640 RGB image from a bird's-eye camera positioned at (0, -0.1, 1.2) looking straight down. This provides a global view of the workspace, both arms, the donut, and the box. Encoded as AV1 video in the dataset.

2. **Front camera** (`pixels/front_camera`): A 480x640 RGB image from an angled front camera positioned at (0, -0.6, 0.7). This provides depth and side-profile information that the top-down view misses — critical for judging gripper-to-donut approach angles and detecting occlusions when one arm crosses over the other. Encoded as AV1 video in the dataset.

3. **Proprioceptive observation** (`agent_pos`): A 12-dimensional vector of current joint angles (radians) for all joints on both arms. This gives the policy direct knowledge of the arm configurations without having to infer them from the images.

#### Actions

The action space is a 12-dimensional vector of **absolute joint position targets** in radians. The MuJoCo position actuators (PD controllers with kp=998.22, kv=2.731) drive each joint to the commanded target. This is the same control mode used by the physical SO-101 via Feetech serial protocol.

The ordering is: `[left_shoulder_pan, left_shoulder_lift, left_elbow_flex, left_wrist_flex, left_wrist_roll, left_gripper, right_shoulder_pan, right_shoulder_lift, right_elbow_flex, right_wrist_flex, right_wrist_roll, right_gripper]`.

### Dataset Format

Demonstrations are stored in [LeRobot v3.0 format](https://github.com/huggingface/lerobot), which consists of:

```
data/local/bi_so101_donut_packing/
├── meta/
│   ├── info.json           # Dataset metadata (fps, features, robot_type, etc.)
│   ├── episodes.jsonl      # Per-episode metadata (index, length, task)
│   └── tasks.jsonl         # Task descriptions
├── data/
│   └── chunk-000/
│       └── episode_*.parquet   # Tabular data (states, actions, timestamps)
└── videos/
    └── chunk-000/
        ├── observation.images.top_camera/
        │   └── episode_*.mp4   # Bird's-eye view video (AV1)
        └── observation.images.front_camera/
            └── episode_*.mp4   # Front view video (AV1)
```

Feature specification:

```json
{
    "observation.state": {
        "dtype": "float32",
        "shape": [12],
        "names": [
            "left_shoulder_pan.pos", "left_shoulder_lift.pos",
            "left_elbow_flex.pos", "left_wrist_flex.pos",
            "left_wrist_roll.pos", "left_gripper.pos",
            "right_shoulder_pan.pos", "right_shoulder_lift.pos",
            "right_elbow_flex.pos", "right_wrist_flex.pos",
            "right_wrist_roll.pos", "right_gripper.pos"
        ]
    },
    "observation.images.top_camera": {
        "dtype": "video",
        "shape": [480, 640, 3]
    },
    "observation.images.front_camera": {
        "dtype": "video",
        "shape": [480, 640, 3]
    },
    "action": {
        "dtype": "float32",
        "shape": [12],
        "names": ["(same as observation.state)"]
    }
}
```

The `robot_type` field is set to `"bi_so_follower"` to match LeRobot's built-in bimanual SO-101 configuration.

### ACT Policy Configuration

The Action Chunking with Transformers (ACT) policy is configured for bimanual control:

```
Architecture:
  Vision backbone:  ResNet-18 (pretrained on ImageNet), shared across cameras
  Transformer:      dim_model=512, nhead=8, num_layers=1
  VAE:              latent_dim=32, kl_weight=10.0
  Action head:      Linear(512 -> 12)  # 12-DOF output

Input processing:
  Top image:   (3, 480, 640) -> ResNet-18 -> (512,) features
  Front image: (3, 480, 640) -> ResNet-18 -> (512,) features
  State:       (12,) -> Linear(12, 512) -> (512,) features
  Combined:    All image + state tokens processed by transformer encoder

Output:
  Action chunk: (chunk_size, 12) = (50, 12)
  At inference: all 50 actions are executed sequentially,
                then a new chunk is predicted
```

ACT handles arbitrary action dimensions because the action head is dynamically sized: `nn.Linear(config.dim_model, config.action_feature.shape[0])`. The 12-DOF bimanual configuration works without any code changes to the ACT policy implementation.

ACT also natively supports multiple image inputs. Each image feature listed in `config.image_features` is passed through the vision backbone independently, producing a set of visual tokens that are concatenated with the state token and fed to the transformer. Adding a second camera doubles the visual context without any architectural changes.

---

## Sim-to-Real Transfer

When physical SO-101 arms arrive, the sim-trained policy can be transferred with these steps:

### 1. Hardware Setup

Use LeRobot's built-in `BiSOFollower` robot configuration:

```python
from lerobot.robots.bi_so_follower import BiSOFollowerConfig, BiSOFollower

config = BiSOFollowerConfig(
    left_arm_config=SOFollowerConfig(port="/dev/ttyUSB0"),
    right_arm_config=SOFollowerConfig(port="/dev/ttyUSB1"),
)
robot = BiSOFollower(config)
```

The joint naming convention already matches: `left_shoulder_pan.pos`, `right_gripper.pos`, etc.

### 2. Camera Alignment

Mount two physical cameras to match the simulated viewpoints:

**Top camera:**
- Position: directly above the workspace, ~80cm height
- Orientation: pointing straight down
- Resolution: 480x640

**Front camera:**
- Position: in front of the workspace, ~30cm height, ~60cm away
- Orientation: angled upward toward the table
- Resolution: 480x640

### 3. Domain Adaptation Options

- **Direct transfer:** Load the sim-trained checkpoint and run it on the real robot. This may work for simple tasks but will likely have a sim-to-real gap.
- **Fine-tuning (recommended):** Collect 10-20 real demonstrations using `lerobot-record` with the physical `BiSOFollower`, then fine-tune the sim-trained policy on the combined sim+real dataset.
- **Domain randomization:** Modify `bi_so101_env.py` to randomize textures, lighting, camera position, and dynamics parameters during training. This produces more robust policies for transfer.

### 4. Recording Real Demonstrations

```bash
# Using lerobot's built-in recording pipeline
lerobot-record \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port=/dev/ttyUSB0 \
    --robot.right_arm_config.port=/dev/ttyUSB1 \
    --teleop.type=bi_so_leader \
    --dataset.repo_id=myuser/bi_so101_donut_real \
    --fps 30
```

### 5. Fine-Tuning

```bash
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=myuser/bi_so101_donut_real \
    --checkpoint_path=outputs/train/act_bi_so101_donut/checkpoints/best/pretrained_model \
    --steps=5000 \
    --batch_size=4
```

---

## Troubleshooting

### "OpenGL platform library has not been loaded"

This error occurs on headless servers (no display). Set the rendering backend to EGL:

```bash
export MUJOCO_GL=egl
```

The environment sets this automatically via `os.environ.setdefault("MUJOCO_GL", "egl")`, but the MuJoCo viewer (`mujoco.viewer.launch_passive`) still requires a display for the GUI window.

### MuJoCo viewer not opening

The teleoperation collector and viewer-mode deployment require a display server. Options:
- Run on a local machine with a monitor
- Use X11 forwarding: `ssh -X user@server`
- Use a virtual display: `Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1`

### "torchcodec ABI mismatch" warning

This is a harmless warning. LeRobot falls back to PyAV for video decoding, which works correctly. You can suppress it by uninstalling torchcodec: `pip uninstall torchcodec`.

### Dataset validation errors

Verify your dataset is valid after recording:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="local/bi_so101_donut_packing",
    root="./data/local/bi_so101_donut_packing"
)
print(f"Episodes: {ds.meta.total_episodes}")
print(f"Frames: {ds.num_frames}")
print(f"Features: {list(ds.meta.features.keys())}")
sample = ds[0]
print(f"State shape: {sample['observation.state'].shape}")
print(f"Action shape: {sample['action'].shape}")
print(f"Top camera shape: {sample['observation.images.top_camera'].shape}")
print(f"Front camera shape: {sample['observation.images.front_camera'].shape}")
```

### Training loss not decreasing

- **Check dataset size.** At least 30-50 episodes are recommended for ACT to learn meaningful bimanual coordination.
- **Check for corrupted episodes.** Discard episodes where the task was not completed or the arms behaved erratically.
- **Reduce learning rate.** The default may be too aggressive for small datasets.
- **Increase chunk_size.** Longer action chunks can help with temporal coherence in bimanual tasks.

### Donut falls through table / doesn't respond to gripper

- Verify `contype` and `conaffinity` flags. The donut and fingertips must have `contype=1 conaffinity=1` to collide with each other.
- Check that fingertip collision boxes (`left_fingertip`, `right_fingertip`, `left_jaw_tip`, `right_jaw_tip`) are present in the scene XML.
- Increase donut friction if it slides out of the gripper too easily.
