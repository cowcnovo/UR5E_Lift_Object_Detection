import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
 
import os
 
 
UR5E_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "models/ur5e.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # Default to False, adjust if needed
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 0.7853,
            "wrist_1_joint": -0.7853,
            "wrist_2_joint": -1.5708,
            "wrist_3_joint": 1.5708,
            "left_outer_knuckle_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
        },
        pos=(0.0, 0.0, 1.05),
    ),
    actuators={
        "shoulder_pan_joint_act": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=1.5,
            stiffness=260.0,
            damping=26.0,
        ),
        "shoulder_lift_joint_act": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=1.5,
            stiffness=260.0,
            damping=26.0,
        ),
        "elbow_joint_act": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=1.5,
            stiffness=260.0,
            damping=26.0,
        ),
        "wrist_1_joint_act": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=1.5,
            stiffness=260.0,
            damping=26.0,
        ),
        "wrist_2_joint_act": ImplicitActuatorCfg(
            joint_names_expr=["wrist_2_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=1.5,
            stiffness=260.0,
            damping=26.0,
        ),
        "wrist_3_joint_act": ImplicitActuatorCfg(
            joint_names_expr=["wrist_3_joint"],
            effort_limit_sim=28.0,
            velocity_limit_sim=1.5,
            stiffness=260.0,
            damping=26.0,
        ),
        "left_outer_knuckle_joint_act": ImplicitActuatorCfg(
            joint_names_expr=["left_outer_knuckle_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.5,
            stiffness=60.0,
            damping=20.0,
        ),
        "right_outer_knuckle_joint_act": ImplicitActuatorCfg(
            joint_names_expr=["right_outer_knuckle_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.5,
            stiffness=60.0,
            damping=20.0,
        ),
    },
)