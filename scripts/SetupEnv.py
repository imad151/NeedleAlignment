"""
    Contains the configuration for the dVRK scene
"""

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import matplotlib.pyplot as plt
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg



@configclass
class dVRKSceneCfg(InteractiveSceneCfg):
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    light: AssetBaseCfg = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)))

    Robot = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Robots/dVRK/PSM/psm_col.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
        ),
        prim_path="/World/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "psm_yaw_joint": 0.01,
                "psm_pitch_end_joint": 0.01,
                "psm_main_insertion_joint": 0.07,
                "psm_tool_roll_joint": 0.01,
                "psm_tool_pitch_joint": 0.01,
                "psm_tool_yaw_joint": 0.01,
                "psm_tool_gripper1_joint": -0.5,
                "psm_tool_gripper2_joint": 0.5,
            },
            pos=(0.0, 0.0, 0.15),
        ),
        actuators={
            "psm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "psm_yaw_joint",
                    "psm_pitch_end_joint",
                    "psm_main_insertion_joint",
                    "psm_tool_roll_joint",
                    "psm_tool_pitch_joint",
                    "psm_tool_yaw_joint",
                ],
                effort_limit=12.0,
                velocity_limit=1.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "psm_tool": ImplicitActuatorCfg(
                joint_names_expr=["psm_tool_gripper.*"],
                effort_limit=0.1,
                velocity_limit=0.2,
                stiffness=500,
                damping=0.1,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    camera = CameraCfg(
        prim_path="/World/camera",
        update_period=0.1,
        
        height=2160,
        width=3840,
        
        #height=1080,
        #width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(0.0018, -0.11761, 0.0519), rot = rot_utils.euler_angles_to_quats(np.array([-90.0, 0.0, 0.0]), degrees=True)),
    )


    Needle: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Needle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0123, 0., -0.06104), rot=rot_utils.euler_angles_to_quats(np.array([90.0, 0.0, -90.0]), degrees=True)),
        spawn=UsdFileCfg(
            usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_needle/needle_sdf.usd",
            scale=(0.4, 0.4, 0.4),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_angular_velocity=200,
                max_linear_velocity=200,
                max_depenetration_velocity=1.0,
                disable_gravity=True,
            ),
        ),
    )