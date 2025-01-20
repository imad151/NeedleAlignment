"""
Default imports for launching isaac sim
"""

from __future__ import annotations

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="dVRK setup environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to launch")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

"""
Rest of the code goes here
"""

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim import SimulationContext                                
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.core.utils.numpy.rotations as rot_utils

import carb
import omni.kit.commands
import omni.replicator.core as rep

settings = carb.settings.get_settings()
print(settings)
# settings.set("/rtx/pathtracing/enabled", True)  # Enable path tracing



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
                "psm_tool_gripper1_joint": -0.09,
                "psm_tool_gripper2_joint": 0.09,
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
        
        # height=2160,
        # width=3840,
        
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(0.0018, -0.11761, 0.10819), rot = rot_utils.euler_angles_to_quats(np.array([-90.0, 0.0, 0.0]), degrees=True)),
    )


import cv2
def display_image(img_list):
    for idx, img in enumerate(img_list):
        # Convert PyTorch tensor to NumPy array if needed
        if hasattr(img, 'detach'):
            img = img.detach().cpu().numpy()
        
        # Print image shape for debugging
        print(f"Image {idx+1} shape: {img.shape}")

        # Check if image is already in (H, W, C) format
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert RGB to BGR (OpenCV uses BGR format)
            img_to_show = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        elif len(img.shape) == 2:  # Grayscale image (H, W)
            img_to_show = img.astype(np.uint8)
        else:
            raise ValueError(f"Invalid image format: {img.shape}. Expected (H, W) or (H, W, 3).")

        # Display the image
        cv2.imwrite(f"image_{idx+1}.png", img_to_show)



    




def run_sim(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["Robot"]
    sim_dt = sim.get_physics_dt()
    count = 0
    done = False
    while sim_app.is_running():
        scene.update(sim_dt)
        sim.step()
        
        count += 1
        '''
        if not done:
            
            if count == 100:
                rgb_img = [scene["camera"].data.output["rgb"][0, ..., :3], scene["camera"].data.output["rgb"][0]]
                display_image(rgb_img)
                done = True'''




def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = dVRKSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    run_sim(sim, scene)


if __name__ == "__main__":
    main()
    sim_app.close()


