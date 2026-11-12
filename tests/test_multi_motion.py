"""Tests for multi-motion tracking functionality."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
  MotionLoader,
  _resolve_motion_file,
)
from mjlab.tasks.tracking.mdp.terminations import motion_end
from mjlab.utils.forward_kinematics import ForwardKinematics


@pytest.fixture(scope="module")
def fk() -> ForwardKinematics:
  """Forward kinematics of the G1 to its first three bodies."""
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  return ForwardKinematics(
    model,
    torch.arange(1, model.nbody),
    torch.from_numpy(model.jnt_qposadr[1:]),
    torch.tensor([0, 1, 2]),
  )


@pytest.fixture
def create_motion_file(tmp_path, fk):
  """Factory to create motion NPZ files of random root and joint states of the
  robot of ``fk``."""

  def _create(name: str, num_frames: int) -> str:
    root_pos, root_lin_vel, root_ang_vel = torch.randn(3, num_frames, 3)
    root_quat = torch.nn.functional.normalize(torch.randn(num_frames, 4), dim=-1)
    joint_pos, joint_vel = torch.randn(2, num_frames, len(fk.joint_names))
    path = tmp_path / f"{name}.npz"
    np.savez(
      path,
      joint_pos=joint_pos.numpy(),
      joint_vel=joint_vel.numpy(),
      root_pos_w=root_pos.numpy(),
      root_quat_w=root_quat.numpy(),
      root_lin_vel_w=root_lin_vel.numpy(),
      root_ang_vel_w=root_ang_vel.numpy(),
      joint_names=fk.joint_names,
    )
    return str(path)

  return _create


# _resolve_motion_file tests


def test_resolve_single_path(create_motion_file):
  path = create_motion_file("motion", num_frames=100)
  assert _resolve_motion_file(path) == [path]


def test_resolve_list_of_paths(create_motion_file):
  paths = [
    create_motion_file("a", num_frames=50),
    create_motion_file("b", num_frames=100),
  ]
  assert _resolve_motion_file(paths) == paths


def test_resolve_glob_pattern(create_motion_file, tmp_path):
  for i in range(3):
    create_motion_file(f"motion_{i}", num_frames=50)
  pattern = str(tmp_path / "motion_*.npz")
  result = _resolve_motion_file(pattern)
  assert len(result) == 3


def test_resolve_glob_no_match(tmp_path):
  pattern = str(tmp_path / "nonexistent_*.npz")
  with pytest.raises(ValueError, match="No files matched"):
    _resolve_motion_file(pattern)


# MotionLoader tests


def test_loader_single_motion(create_motion_file, fk):
  path = create_motion_file("motion", num_frames=100)
  loader = MotionLoader(path, fk)

  assert loader.time_step_total == 100
  assert loader.joint_pos.shape == (100, len(fk.joint_names))
  assert loader.root_pos_w.shape == (100, 3)


def test_loader_multiple_motions_concatenated(create_motion_file, fk):
  paths = [
    create_motion_file("a", num_frames=50),
    create_motion_file("b", num_frames=100),
    create_motion_file("c", num_frames=75),
  ]
  loader = MotionLoader(paths, fk)

  # Should be concatenated: 50 + 100 + 75 = 225 frames
  assert loader.time_step_total == 225
  assert loader.joint_pos.shape == (225, len(fk.joint_names))
  assert loader.root_pos_w.shape == (225, 3)


def test_loader_motion_ends(create_motion_file, fk):
  """Verify motion_ends tracks cumulative boundaries."""
  paths = [
    create_motion_file("a", num_frames=50),
    create_motion_file("b", num_frames=100),
    create_motion_file("c", num_frames=75),
  ]
  loader = MotionLoader(paths, fk)

  # motion_ends should be cumulative: [50, 150, 225]
  assert loader.motion_ends.tolist() == [50, 150, 225]


def test_loader_concatenation_preserves_data(create_motion_file, fk):
  """Verify that concatenated data comes from the correct motion files."""
  paths = [
    create_motion_file("a", num_frames=50),
    create_motion_file("b", num_frames=100),
  ]
  loader = MotionLoader(paths, fk)

  # First 50 frames from motion_a, frames 50-149 from motion_b.
  joint_pos = [torch.from_numpy(np.load(path)["joint_pos"]) for path in paths]
  assert torch.equal(loader.joint_pos[:50], joint_pos[0])
  assert torch.equal(loader.joint_pos[50:], joint_pos[1])


def test_loader_rejects_other_joints(create_motion_file, fk):
  path = create_motion_file("motion", num_frames=20)
  data = dict(np.load(path))
  data["joint_names"] = data["joint_names"][::-1]
  np.savez(path, **data)
  with pytest.raises(ValueError, match="robot's"):
    MotionLoader(path, fk)


@pytest.fixture
def motion_env(tmp_path, floating_base_articulated_xml):
  paths = []
  for clip in range(2):
    frames = np.arange(12, dtype=np.float32)
    root_pos = np.column_stack((clip + frames * 0.01, frames * 0, frames * 0 + 1))
    path = tmp_path / f"clip_{clip}.npz"
    np.savez(
      path,
      root_pos_w=root_pos,
      root_quat_w=np.tile([np.cos(0.3), 0, 0, np.sin(0.3)], (12, 1)),
      root_lin_vel_w=np.tile([0.1, 0.2, 0.3], (12, 1)),
      root_ang_vel_w=np.tile([0.3, -0.2, 0.1], (12, 1)),
      joint_pos=np.column_stack((frames * 0.001, -frames * 0.001)),
      joint_vel=np.tile([0.1, -0.2], (12, 1)),
      joint_names=["joint1", "joint2"],
    )
    paths.append(str(path))
  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    scene=SceneCfg(
      num_envs=2,
      env_spacing=3.0,
      entities={
        "robot": EntityCfg(
          spec_fn=lambda: mujoco.MjSpec.from_string(floating_base_articulated_xml)
        )
      },
    ),
    commands={
      "motion": MotionCommandCfg(
        motion_file=paths,
        entity_name="robot",
        anchor_body_name="base",
        body_names=("link2", "base", "link1"),
        sampling_mode="external",
        joint_position_range=(0.0, 0.0),
        resampling_time_range=(1e9, 1e9),
      )
    },
    terminations={
      "motion_end": TerminationTermCfg(
        func=motion_end, time_out=True, params={"command_name": "motion"}
      )
    },
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  yield env
  env.close()


def test_motion_reset_keeps_root_state_and_selected_reference_bodies(motion_env):
  env = motion_env
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  env_ids = torch.arange(env.num_envs)
  command.reset_to_frame(env_ids, 1)
  env.sim.forward()

  # The first tracked body is deliberately not the free root. Both reset writes
  # and FK must retain the root LINK's world pose and world-frame twist.
  root = torch.cat(
    (
      command.motion.root_pos_w[command.time_steps] + env.scene.env_origins,
      command.motion.root_quat_w[command.time_steps],
      command.motion.root_lin_vel_w[command.time_steps],
      command.motion.root_ang_vel_w[command.time_steps],
    ),
    dim=-1,
  )
  torch.testing.assert_close(
    torch.cat(
      (command.robot.data.root_link_pose_w, command.robot.data.root_link_vel_w), -1
    ),
    root,
  )
  torch.testing.assert_close(command.robot_body_pos_w, command.body_pos_w)
  torch.testing.assert_close(command.robot_body_quat_w, command.body_quat_w)
  torch.testing.assert_close(command.robot_body_lin_vel_w, command.body_lin_vel_w)
  torch.testing.assert_close(command.robot_body_ang_vel_w, command.body_ang_vel_w)

  untouched = command.body_pos_w[1].clone()
  command.reset_to_frame(torch.tensor([0]), 13)
  env.sim.forward()
  assert command.time_steps.tolist() == [13, 1]
  torch.testing.assert_close(command.body_pos_w[1], untouched)
  torch.testing.assert_close(command.robot_body_pos_w, command.body_pos_w)


def test_external_reset_distribution_and_each_clip_end_stay_connected(motion_env):
  env = motion_env
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  probabilities = torch.zeros(command.bin_count)
  probabilities[-1] = 1.0
  command.set_sampling_distribution(probabilities)
  command.reset(torch.arange(env.num_envs))
  assert torch.all(command.time_steps >= command.motion.motion_ends[0])
  assert not torch.isin(command.time_steps, command.motion.motion_ends - 1).any()

  command.set_time_steps(torch.arange(env.num_envs), command.motion.motion_ends - 1)
  assert env.termination_manager.compute().all()
  assert env.termination_manager.time_outs.all()
  assert not env.termination_manager.terminated.any()
  command.set_time_steps(torch.arange(env.num_envs), command.motion.motion_ends - 2)
  assert not env.termination_manager.compute().any()
