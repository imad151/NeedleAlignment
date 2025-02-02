import numpy as np

def compute_projection_on_floor(camera_pos, target_pos, focal_length, horizontal_aperture):
    """
    Compute the projected square area on an infinite floor given camera intrinsics.
    """
    fov_h = 2 * np.arctan((horizontal_aperture / 2) / focal_length)
    fov_v = 2 * np.arctan((horizontal_aperture / (2 * (16/9))) / focal_length)  # Assuming 16:9 aspect ratio

    direction = target_pos - camera_pos
    direction /= np.linalg.norm(direction)  

    lambda_intersect = -camera_pos[2] / direction[2] 
    center_on_floor = camera_pos + lambda_intersect * direction

    half_width = np.tan(fov_h / 2) * lambda_intersect
    half_height = np.tan(fov_v / 2) * lambda_intersect

    top_left = center_on_floor + np.array([-half_width, half_height, 0])
    top_right = center_on_floor + np.array([half_width, half_height, 0])
    bottom_left = center_on_floor + np.array([-half_width, -half_height, 0])
    bottom_right = center_on_floor + np.array([half_width, -half_height, 0])

    width = np.linalg.norm(top_right - top_left)
    height = np.linalg.norm(top_left - bottom_left)
    area = width * height

    return np.array([top_left, top_right, bottom_left, bottom_right]), area

camera_pos = np.array([0, 0, 1.0])  # Camera at 
target_pos = np.array([0, 0, 0])   # Looking at
focal_length = 24.0
horizontal_aperture = 20.955

corners, area = compute_projection_on_floor(camera_pos, target_pos, focal_length, horizontal_aperture)

print("Projected Area on Floor:", area)
print("Corner Coordinates:", corners)
