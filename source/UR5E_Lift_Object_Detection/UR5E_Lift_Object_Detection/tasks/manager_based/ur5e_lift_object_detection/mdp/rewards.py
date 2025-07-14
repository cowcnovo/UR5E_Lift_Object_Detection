# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("ur5e"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return distance

def object_goal_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    std_reach: float,
    std_height: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("ur5e"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    
    object_goal_dist = object_goal_distance(env, command_name, robot_cfg, object_cfg)
    
    # print("Goal:")
    # print(object_goal_dist)

    lifting_reward = object_is_lifted(env, std_reach, std_height, object_cfg, ee_frame_cfg)

    return (1 - torch.tanh(object_goal_dist / std)) * lifting_reward

def orientation_command_error(
        env: ManagerBasedRLEnv, 
        command_name: str, 
        asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def object_position_error(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(ee_w - cube_pos_w, dim=1)
    return object_ee_distance

def object_position_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object_ee_distance = object_position_error(env, object_cfg, ee_frame_cfg)
    return 1 - torch.tanh(object_ee_distance / std)

def object_is_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    std_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""

    object: RigidObject = env.scene[object_cfg.name]
    object_height_from_initial = object.data.root_pos_w[:, 2] - 1.072
    object_height_reward = torch.tanh(object_height_from_initial / std_height)

    reach_reward = object_position_error_tanh(env, std, object_cfg, ee_frame_cfg)

    reward = reach_reward * object_height_reward
    # print("Lift:")
    # print(object_height_from_initial, reward)

    return reward

def object_stability(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Object"),
) -> torch.Tensor:
 
    object: RigidObject = env.scene[object_cfg.name]
    object_vel = object.data.root_lin_vel_w[:, 0:2]
    object_xy_speed = torch.norm(object_vel, dim=1)
    #print(object_xy_speed)

    return object_xy_speed

def not_hit_floor(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("Object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_z_pos_w = object.data.root_pos_w[:, 2]
    ee_z_pos_w = ee_frame.data.target_pos_w[..., 0, 2]
    height = ee_z_pos_w - cube_z_pos_w

    #print(height)

    return 0.5*torch.tanh(1000*(height + 0.003))-0.5