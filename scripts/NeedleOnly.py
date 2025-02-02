from omni.isaac.lab.app import AppLauncher
import argparse


parser = argparse.ArgumentParser(description="dVRK setup environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to launch")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import os
import torch
import numpy as np

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

import omni.replicator.core as rep
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, create_prim
import omni.usd
import carb
import omni.isaac.core.utils.carb as carb_utils
from omni.isaac.core.utils.carb import set_carb_setting


# Number of images to capture
NUM_IMGS = 100


def set_segmentation_label(prim_path, label):
    """Assigns a segmentation label to a given prim."""
    prim = get_prim_at_path(prim_path)
    if not prim:
        print(f"Error: Prim not found at {prim_path}")
        return
    add_update_semantics(prim, label)
    print(f"Added segmentation label '{label}' to {prim_path}")


def spawn_needles(scene, num_needles):
    """Spawns a given number of surgical needles in the scene with random poses."""
    spawned_prim_paths = []
    
    for i in range(num_needles):
        prim_path = f"/World/Needle{i}"
        pos = np.random.uniform(-0.5, 0.5, 2).tolist() + [0.1]
        orient = np.random.uniform(1, 100, 3).tolist()
        print(orient)
        
        create_prim(
            prim_path,
            usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_needle/needle_sdf.usd",
            position=pos,
            orientation=rot_utils.euler_angles_to_quats(orient),
            scale=(0.1, 0.1, 0.1),

        )
        set_segmentation_label(prim_path, "Surgical Needle")
        spawned_prim_paths.append(prim_path)

    return spawned_prim_paths


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Defines the static environment including ground, lighting, and camera."""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
    )

    Camera = CameraCfg(
        prim_path="/World/camera",
        update_period=0,
        height=2160,
        width=3840,
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

def run_sim(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop, spawning needles, capturing images, and cleaning up objects."""
    camera = scene["Camera"]
    image_dir = "/home/imad/NeedleAlignment/data/images/needleonly"
    os.makedirs(image_dir, exist_ok=True)

    for iteration in range(NUM_IMGS):
        print(f"Iteration {iteration + 1}/{NUM_IMGS}: Spawning needles and capturing images.")

        num_needles = np.random.randint(1, 7)
        spawned_needles = spawn_needles(scene, num_needles)
        
        camera_positions = torch.tensor([[0.3, 0.3, 0.3]], device=sim.device)
        camera_targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
        camera.set_world_poses_from_view(camera_positions, camera_targets)

        for _ in range(50):
            sim.step()

        camera.update(dt=sim.get_physics_dt())

        single_cam_data = convert_dict_to_backend(
            {k: v[0] for k, v in camera.data.output.items()}, backend="numpy"
        )
        single_cam_info = camera.data.info[0]

        rep_output = {"annotators": {}}
        for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
            rep_output["annotators"][key] = {"render_product": {"data": data, **(info or {})}}
        rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}

        output_image_path = os.path.join(image_dir, f"{iteration}")
        rep_writer = rep.BasicWriter(
            output_dir=output_image_path,
            frame_padding=0,
            colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        )
        rep_writer.write(rep_output)

        # Remove spawned needles with extra checks
        for prim_path in spawned_needles:
            prim = get_prim_at_path(prim_path)
            if prim:
                delete_prim(prim_path)

        # Force USD updates to process
        omni.usd.get_context().get_stage().Flatten()
        
    while True:
        sim.step()


def main():
    """Initializes the simulation and starts the main loop."""


    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Create scene configuration
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    carb_settings = carb.settings.get_settings()
    # set values
    set_carb_setting(carb_settings, "/rtx/pathtracing/enabled", True)

    # Reset and run simulation
    sim.reset()
    run_sim(sim, scene)

    # Close the app after execution
    sim_app.close()


if __name__ == "__main__":
    main()
