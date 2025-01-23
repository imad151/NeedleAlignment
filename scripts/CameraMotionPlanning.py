import numpy as np
import matplotlib.pyplot as plt

def compute_trajectory(tip, cam, num_points):
    radius = np.linalg.norm(cam[:2] - tip[:2])  # Compute radius in the XY plane
    points = []
    orientations = []
    for point in range(num_points):
        theta = 2 * np.pi * point / num_points  # Angle for circular motion
        new_pos = np.array([tip[0] + radius * np.cos(theta),  # X-coordinate
                            tip[1] + radius * np.sin(theta),  # Y-coordinate
                            tip[2]])  # Z-coordinate remains constant
        orientation = tip[:2] - new_pos[:2]  # Vector from new_pos to tip
        orientation = np.append(orientation / np.linalg.norm(orientation), 0)  # Normalize and add Z-coordinate
        points.append(new_pos)
        orientations.append(orientation)
    return np.array(points), np.array(orientations)

def plot_trajectory(points, orientations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-')  
    for i in range(len(points)):
        ax.quiver(points[i, 0], points[i, 1], points[i, 2], 
                  orientations[i, 0], orientations[i, 1], orientations[i, 2], 
                  length=0.05, color='r', normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Oriented 3D Trajectory (Fixed Z)')
    plt.show()

# Parameters
NUM_POINTS = 50
POS_CAM_INIT = np.array([0.0018, -0.11761, 1])  # Camera position
POS_TIP_INIT = np.array([0.0, 0.0, 0])  # Tip position

# Compute and plot the trajectory
points, orientations = compute_trajectory(POS_TIP_INIT, POS_CAM_INIT, NUM_POINTS)
plot_trajectory(points, orientations)
