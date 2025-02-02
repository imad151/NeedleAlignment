import argparse
from omni.isaac.lab.app import AppLauncher
parser = argparse.ArgumentParser(description="dVRK setup environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to launch")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app


import numpy as np
import omni.isaac.lab.sim as sim_utils
import cv2
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from SetupEnv import dVRKSceneCfg

from omni.isaac.core.utils.semantics import add_update_semantics
from pxr import Usd
import omni.isaac.core.utils.semantics as semantics

# Parse command-line arguments



def save_image(data, filename):
    """Saves image data to file."""
    cv2.imwrite(filename, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))


def process_and_save_images(scene, step_count):
    """Captures and saves RGB and segmentation images."""
    cameras = {
        "camera": scene["camera"],
        "tiled_camera": scene["tiled_camera"]
    }

    for cam_name, cam in cameras.items():
        # Access the latest sensor data
        cam_data = cam.data

        # Save RGB image if available
        if "rgb" in cam_data.output:
            rgb_image = cam_data.output["rgb"][0, ..., :3]
            rgb_image = rgb_image.detach().cpu().numpy()
            rgb_filename = f"{cam_name}_rgb_{step_count:04d}.png"
            save_image(rgb_image.astype(np.uint8), rgb_filename)
            print(f"Saved {rgb_filename}")

        # Save semantic segmentation if available
        if "semantic_segmentation" in cam_data.output:
            seg_image = cam_data.output["semantic_segmentation"][0, ..., :3]
            seg_image = seg_image.detach().cpu().numpy()
            seg_filename = f"{cam_name}_semantic_{step_count:04d}.png"
            
            # Normalize and visualize segmentation
            seg_vis = (seg_image.astype(np.uint8) / seg_image.max() * 255).astype(np.uint8)
            save_image(seg_vis, seg_filename)
            print(f"Saved {seg_filename}")


def run_sim(sim, scene):
    step_count = 0
    while step_count < 3:  # Run for 100 steps, modify as needed
        sim.step()
        process_and_save_images(scene, step_count)
        step_count += 1

def apply_semantic_labels(sim_context, psm_cfg):
    """
    Apply segmentation labels to the PSM joints and other relevant Prims after the stage is loaded.

    Args:
        sim_context (SimulationContext): The simulation context.
        psm_cfg (ArticulationCfg): The configuration for the PSM robot.
    """
    stage = sim_context.stage  # Access the USD stage through the simulation context

    # Define semantic labels for components in the scene
    semantic_labels = {
        "psm_yaw_joint": "yaw",
        "psm_pitch_end_joint": "pitch_end",
        "psm_main_insertion_joint": "insertion",
        "psm_tool_roll_joint": "tool_roll",
        "psm_tool_pitch_joint": "tool_pitch",
        "psm_tool_yaw_joint": "tool_yaw",
        "psm_tool_gripper1_joint": "gripper1",
        "psm_tool_gripper2_joint": "gripper2",
    }

    # Apply labels to each joint and other relevant Prims
    for joint_name, label in semantic_labels.items():
        prim_path = f"{psm_cfg.prim_path}/{joint_name}"  # Full path to the joint Prim
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            add_update_semantics(prim, semantic_label=label, type_label="class")  # Apply semantic label
            print(f"Applied semantic label '{label}' to {prim_path}.")
        else:
            print(f"Warning: Prim '{prim_path}' does not exist in the stage.")


def main():
    # Initialize simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Load the scene
    scene_cfg = dVRKSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    apply_semantic_labels(sim, scene_cfg.Robot)


    sim.reset()
    run_sim(sim, scene)


if __name__ == "__main__":
    main()
    sim_app.close()
