
import cv2
import numpy as np

# Step 1: Read the image (ignore alpha by using cv2.IMREAD_COLOR)
image = cv2.imread('/home/imad/NeedleAlignment/data/needleonly/1/instance_id_segmentation_1_0.png', cv2.IMREAD_COLOR)  # Load as RGB (no alpha)

# Step 2: Define the exact RGB color for the green background
green_color = [140, 255, 25]  # BGR format: Blue=140, Green=255, Red=25
black_color = [0, 0, 0]  # Black in RGB: Blue=0, Green=0, Red=0
lower_green = np.array([120, 240, 10])  # Lower bound (approximate range for green)
upper_green = np.array([160, 270, 40])  # Upper bound (approximate range for green)

# Step 3: Threshold the image to create a binary mask
mask = cv2.inRange(image, lower_green, upper_green)
# Step 4: Replace the green background with black
image[mask] = black_color

# Step 5: Save the result

# Optional: Show the image
cv2.imshow('Processed Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
