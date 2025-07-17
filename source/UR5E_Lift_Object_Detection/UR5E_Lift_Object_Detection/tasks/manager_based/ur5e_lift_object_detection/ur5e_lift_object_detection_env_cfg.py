# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from dataclasses import MISSING

from isaaclab.app import AppLauncher

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg

import os
import math

from .ur5e_config import UR5E_CONFIG
from .mdp.rewards import orientation_command_error
from .mdp.rewards import object_position_error, object_position_error_tanh
from .mdp.rewards import not_hit_floor, object_is_lifted, object_stability
from .mdp.rewards import object_goal_distance_tanh, object_goal_distance
from .mdp.observations import object_position_in_robot_root_frame

from .object_detection import load_model, reset_model, inference

##
# Scene definition
##

arm_joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",]
gripper_joint_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"]

@configclass
class UR5ESceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "models/plane.usd"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.5),
            scale=(1000.0, 1000.0, 1.0),
        )
    )

    # spawn a cuboid with colliders and rigid body
    Object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.20)),
    )

    # Tray
    tray = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tray",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "models/tray.usd"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 1.10),
            rot=(0.707, 0, 0, 0.707)
        ),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.6, 0.0, 1.05),
            rot=(0.707, 0.0, 0, 0.707),  
        ),
    )

    # Camera
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.5, focus_distance=0.8, horizontal_aperture=3.896,
        ),
        width=1280,
        height=960,
        update_period=1/20,
        offset=CameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 1.85), 
            rot=(-0.24184, 0.66446, 0.66446, -0.24184), # real, x, y, z (zyx rotation with frames changing with each subrotation)
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    ur5e: ArticulationCfg = UR5E_CONFIG.replace(prim_path="{ENV_REGEX_NS}/ur5e")

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/ur5e/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/ur5e/gripper_end_effector",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    )
##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_orientation = mdp.UniformPoseCommandCfg(
        asset_name="ur5e",
        body_name="gripper_end_effector", 
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(0.0, 0.0),
        ),
    )

    goal_pose = mdp.UniformPoseCommandCfg(
        asset_name="ur5e",
        body_name="gripper_end_effector", 
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.4, 0.4),
            pos_z=(0.3, 0.3),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="ur5e", 
        joint_names=arm_joint_names,
        scale=1.0
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="ur5e", 
        joint_names=gripper_joint_names,
        open_command_expr={"left_outer_knuckle_joint": 0.0, "right_outer_knuckle_joint": 0.0},
        close_command_expr={"left_outer_knuckle_joint": 0.698, "right_outer_knuckle_joint": -0.698},
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("ur5e", joint_names=arm_joint_names+gripper_joint_names),
            }
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("ur5e", joint_names=arm_joint_names+gripper_joint_names),
            }
        )

        object_positions = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "rgb",
                "model_name": "yolo_model",
                "model_zoo_cfg": {
                    "yolo_model": {
                        "model": load_model,
                        "reset": reset_model,
                        "inference": inference,
                    }
                }
            }
        )

        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_orientation"})
        goal_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.2), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("Object", body_names="Object"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=object_position_error,
        weight=-3.0,
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=object_position_error_tanh,
        weight=10.0,
        params={"std": 0.1},
    )
    end_effector_orientation_tracking = RewTerm(
        func=orientation_command_error,
        weight=-8.0,
        params={"asset_cfg": SceneEntityCfg("ur5e", body_names=["gripper_end_effector"]), "command_name": "ee_orientation"},
    )
    lifting_reward = RewTerm(
        func=object_is_lifted,
        weight=50.0,
        params={"std": 0.1, "std_height": 0.1},
    )
    object_stability_reward = RewTerm(
        func=object_stability,
        weight=-0.1
    )
    not_hit_plane = RewTerm(
        func=not_hit_floor,
        weight=10.0
    )
    goal_tracking_fine_grained = RewTerm(
        func=object_goal_distance_tanh,
        weight=100.0,
        params={"command_name": "goal_pose", "std": 0.3, "std_reach": 0.1, "std_height": 0.1},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("ur5e")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.7, "asset_cfg": SceneEntityCfg("Object")}
    )


# @configclass
class CurriculumCfg:
#     """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4000}
    )

    # object_stability_reward = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "object_stability_reward", "weight": -3.0, "num_steps": 4000}
    # )

##
# Environment configuration
##

@configclass
class UR5ELiftObjectDetectionEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: UR5ESceneCfg = UR5ESceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        # viewer settings
        self.viewer.eye = [3, 3, 3.0]
        self.viewer.lookat = [0.2, 0.0, 1.0]