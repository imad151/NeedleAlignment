"""
    Environment with Needle only.
    For trainging model for Needle Segmentation / pose estimation
"""

# App Setup
from omni.isaac.lab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Setup with Needle Only")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app


# App Settings
import omni
import omni.usd
import carb

# Simulation Setup
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.assets import AssetBaseCfg
import omni.isaac.lab.utils as lab_utils
import omni.isaac.core.utils as core_utils
from omni.isaac.lab.sensors import CameraCfg

# Replicator Setup
import omni.replicator.core as rep

# Orbit Surgical
from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

# Extra Imports
import os
import numpy as np
import torch
import json

# Parameters
NUM_IMGS: int = 5
RESOLUTION: list = [2160, 3840]
PATH_TRACING: bool = True
GROUND_VISIBLE: bool = False
GROUND_COLOR: tuple = (0, 0, 0)
LIGHT_INTENSITY: float = 3000.0
LIGHT_COLOR: tuple = (0.0, 0.0, 0.0)
NEEDLE_SIZE: tuple = (0.3, 0.3, 0.3)
DATA_OUTPUT_PATH: str = r"/home/imad/NeedleAlignment/data/needleonly"


# Scene Cfg
@lab_utils.configclass
class DefaultScene(InteractiveSceneCfg):
    Ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn = sim_utils.GroundPlaneCfg(visible=GROUND_VISIBLE, color=GROUND_COLOR),
    )

    Light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=LIGHT_INTENSITY, color=LIGHT_COLOR)
    )

    Camera: CameraCfg = CameraCfg(
        prim_path="/World/Camera",
        height=RESOLUTION[0],
        width=RESOLUTION[1],
        data_types=["rgb", "instance_id_segmentation_fast"],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
    )


# Helper Functions
def AddSegmentationLabel(prim_path: str, label: str) -> None:
    prim = core_utils.prims.get_prim_at_path(prim_path)

    if not prim:
        print(f"Error: Prim not found at: {prim_path}")
        return
    
    core_utils.semantics.add_update_semantics(prim, label)
    print(f"Added Segmentation Label: {label} to {prim_path}")


def RNGNeedleParams() -> list[np.ndarray, np.ndarray]:
    """
        Returns list[pos, orient]
        pos = [x, y, z] from -0.15 to 0.15 and z at 0.3 constant
        orient in euler
        size in xyz
    """
    pos = np.array([*np.random.uniform([-0.15, -0.15], [0.15, 0.15]), 0.3])
    orient = np.random.uniform([-np.pi, -np.pi/2, -np.pi], [np.pi, np.pi/2, np.pi])
    scale_factor = np.random.uniform(0.4, 1.2)
    scale = np.array([1., 1., 1.]) * scale_factor
    return pos, orient, scale

def SpawnNewNeedle(num_needles: int) -> list[str]:
    needles_prim_paths = []
    prim_label: str = "Surgical Needle"

    for i in range(num_needles):
        prim_path = f"/World/Needles/Needle{i}"
        pos, orient, scale = RNGNeedleParams()

        core_utils.prims.create_prim(
            prim_path,
            position=pos,
            orientation=core_utils.numpy.rotations.euler_angles_to_quats(orient),
            scale=scale,
            usd_path = f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_needle/needle_sdf.usd",
        )

        AddSegmentationLabel(prim_path=prim_path, label=prim_label)
        needles_prim_paths.append(prim_path)


    return needles_prim_paths


def SetupCamera(sim: sim_utils.SimulationContext,scene: InteractiveScene):
    output_dir = DATA_OUTPUT_PATH
    os.makedirs(output_dir, exist_ok=True)

    camera = scene["Camera"]
    camera_pos = torch.tensor([[0.0, 0.0, 1.0]], device=sim.device)
    camera_target_point = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(camera_pos, camera_target_point)


def SimulationFlow(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    camera = scene["Camera"]
    camera_prim = core_utils.prims.get_prim_at_path(r"/World/Camera")
    dt = sim.get_physics_dt()
    
    SetupCamera(sim, scene)

    AddSegmentationLabel("/World/Ground", "Ground")

    for itertaion in range(NUM_IMGS):
        print(f"Iteration {itertaion+1}/{NUM_IMGS}")

        num_needles = np.random.randint(1, 7)
        needles = SpawnNewNeedle(num_needles)

        for _ in range(50):
            sim.step()

        needle_poses = {}
        for prim_path in needles:
            needle_prim = core_utils.prims.get_prim_at_path(prim_path)
            if needle_prim:
                needle_pose = core_utils.transformations.get_relative_transform(camera_prim, needle_prim)

                needle_poses[prim_path] = {
                    "Transformation Matrix": needle_pose.tolist(),
                    "prim path": prim_path
                }
        


        camera.update(dt)

        cam_data = lab_utils.convert_dict_to_backend(
            {k: v[0] for k, v in camera.data.output.items()}, backend="numpy"
        )
        cam_info = camera.data.info[0]

        rep_output = {"annotators": {}}
        for key, data, info in zip(cam_data.keys(), cam_data.values(), cam_info.values()):
            rep_output["annotators"][key] = {"render_product": {"data": data, **(info or {})}}
        rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}

        output_path = os.path.join(DATA_OUTPUT_PATH, f"{itertaion+1}")

        rep_writer = rep.BasicWriter(
            output_dir = output_path,
            frame_padding = 0,
            colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        )
        rep_writer.write(rep_output)

        with open(os.path.join(output_path, "6D poses.json"), "w") as f:
            json.dump(needle_poses, f)


        for prim_path in needles:
            prim = core_utils.prims.get_prim_at_path(prim_path)
            if prim:
                core_utils.prims.delete_prim(prim_path)

        
        omni.usd.get_context().get_stage().Flatten()

    print(f"=========================\nALL DATA ACCQUIRED")


def main():
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = DefaultScene(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    carb.settings.get_settings().set("persistent/app/viewport/displayOptions", 0)
    carb.settings.get_settings().set("/rtx/pathtracing/enabled", True)
    carb.settings.get_settings().set("/rtx/pathtracing/adaptiveSampling/enabled", True)
    
    """
    carb_settings = carb.settings.get_settings()
    core_utils.carb.set_carb_setting(carb_settings, "/rtx/pathtracing/enabled", True)
    core_utils.carb.set_carb_setting(carb_settings, "/rtx/pathtracing/adaptiveSampling/enabled", True)
    core_utils.carb.set_carb_setting(carb_settings, "persistent/app/viewport/displayOptions", 0)
    """
    sim.reset()
    SimulationFlow(sim, scene)

    sim_app.close()


if __name__ == "__main__":
    main()
