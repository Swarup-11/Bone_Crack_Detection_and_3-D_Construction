import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Function to display images using matplotlib
def cv2_imshow(a):
    if a is None:
        print("Error: Image is empty.")
        return
    a = a.clip(0, 255).astype('uint8')
    plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Loaded Image")
    plt.show()

# Load the image
image_path = "/content/drive/MyDrive/B.1.jpg"
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    print("Error: Unable to load the image.")
else:
    cv2_imshow(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count and classify cracks
    num_cracks = len(contours)
    print("Number of Cracks Detected:", num_cracks)

    if num_cracks < 10:
        print("Plaster is recommended.")
    else:
        print("You may need rod surgery.")

    # 3D Reconstruction
    depth_map = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    height, width = gray.shape
    X, Y = np.meshgrid(range(width), range(height))

    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=depth_map)])
    fig.update_layout(
        title="3D Reconstruction of Bone",
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Intensity'
        )
    )
    fig.show()
