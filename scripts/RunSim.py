from omni.isaac.lab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="dVRK setup environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to launch")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR
import numpy as np
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext

# Import controller
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from SetupEnv import dVRKSceneCfg

import numpy as np
import cv2
import torch
import time

def capture_image(camera, filename: str):
    img = camera.data.output["rgb"][0, ..., :3]
    img = img.detach().cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f'saving image to {filename}')
    cv2.imwrite(filename, img)
    #cv2.imshow(img, "img")


def run_sim(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["Robot"]
    camera = scene["camera"]

    sim_dt = sim.get_physics_dt()
    arm_joint_ids, arm_joint_names = robot.find_joints("psm.*")
    print(f"num of joints: {len(arm_joint_ids)}")
    print(f"joint names: {arm_joint_names}")

    joint_pos_des = np.tile(
        np.array([0.01, 0.01, 0.07, 0.01, 0.01, 0.01, -0.1, 0.1], dtype=np.float32),
        (1, 1)
    )
    print(joint_pos_des.shape)
    joint_pos_des = torch.tensor(joint_pos_des, dtype=torch.float32, device="cuda")

    capture_image(camera, "logs/before.png")

    robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)

    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)

    while True: sim.step()



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
