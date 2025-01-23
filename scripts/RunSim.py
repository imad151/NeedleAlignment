"""
Run the simulation with the dVRK environment
Use python RunSim.py --enable_cameras [--headless]
"""

import argparse
from omni.isaac.lab.app import AppLauncher
parser = argparse.ArgumentParser(description="dVRK setup environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to launch")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app


from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import numpy as np
import cv2


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.sensors import CameraCfg
import omni.isaac.core.utils.numpy.rotations as rot_utils



from pxr import Gf

from SetupEnv import dVRKSceneCfg

import carb
settings = carb.settings.get_settings()
settings.set("/rtx/pathtracing/enabled", True)  # Enable path tracing

NUM_IMGS = 36
ROT_CAM_INIT = Gf.Rotation(Gf.Vec3d(-1.0, 0.0, 0.0), 90)
POS_CAM_INIT = np.array([0.0018, -0.11761, -0.0593])  # Camera position
POS_TIP_INIT = np.array([0.0, 0.0, -0.0593])  # Tip position  


def save_image(img, filename):
    if hasattr(img, 'detach'):
        img = img.detach().cpu().numpy()
    img_to_show = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    print(f"Saving image to {filename}")
    cv2.imwrite(filename, img_to_show)

def compute_trajectory(tip, cam, num_points):
    radius = np.linalg.norm(cam[:2] - tip[:2])
    points = []
    orientations = []
    for point in range(num_points):
        theta = 2 * np.pi * point / num_points
        new_pos = np.array([tip[0] + radius * np.cos(theta),
                            tip[1] + radius * np.sin(theta),
                            tip[2]])
        orientation = tip[:2] - new_pos[:2]
        orientation = np.append(orientation / np.linalg.norm(orientation), 0)
        points.append(new_pos)
        orientations.append(orientation)
    return np.array(points), np.array(orientations)


def move_camera(camera: CameraCfg, points, orientations, step):
    """
    Moves the camera to the calculated position and orientation at a given step.
    """
    pos = points[step]
    orient = rot_utils.euler_angles_to_quats(orientations[step], degrees=True)
    camera.set_world_poses([pos], [orient], convention="isaac")

def run_sim(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["Robot"]
    camera = scene["camera"]

    # Compute trajectory for the camera
    points, orientations = compute_trajectory(POS_TIP_INIT, POS_CAM_INIT, NUM_IMGS)
    print(f"Computed trajectory: {len(points)} points")

    sim_dt = sim.get_physics_dt()
    count = 0

    while sim_app.is_running():
        # Update simulation
        scene.update(sim_dt)
        sim.step()

        if count < NUM_IMGS:
            # Move the camera
            move_camera(camera, points, orientations, count)
            
            # Save the image
            img = scene["camera"].data.output["rgb"][0, ..., :3]
            save_image(img, f"image_{count:04d}.png")
            
            count += 1

            if count == NUM_IMGS:
                print("Finished capturing images")



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
