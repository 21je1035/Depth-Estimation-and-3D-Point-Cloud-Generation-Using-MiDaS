#This part of point cloud conversion was done in Google colab
import cv2
import torch
import numpy as np
from google.colab import files
# required for the model
!pip install timm
# Load the MiDaS model
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Function to load an image
#You can load image directly in jupyter notebook
def load_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        img = cv2.imread(filename)
        return img

# Load an image
img = load_image()

# Q matrix for your camera parameters
Q = np.array(([1, 0, 0, -160],
              [0, 1, 0, -120],
              [0, 0, 0, 350],
              [0, 0, 1/90, 0]), dtype=np.float32)



# Convert the image to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply MiDaS transforms and process the image
input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Convert the prediction to a numpy array
depth_map = prediction.cpu().numpy()


# Normalize the depth map
depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

# Reproject the depth map to 3D points
points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=True)

# Create a mask to filter out invalid points
mask_map = depth_map > 0.4
output_points = points_3D[mask_map]
output_colors = img[mask_map]

# Convert the depth map to a visual format
depth_map_visual = (depth_map * 255).astype(np.uint8)
depth_map_visual = cv2.applyColorMap(depth_map_visual, cv2.COLORMAP_MAGMA)

#Display the image and depth map
#cv2.imshow('Image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#cv2.imshow('Depth Map', depth_map_visual)
#cv2.waitKey(0)
cv2.destroyAllWindows()

# Function to create a PLY file from the point cloud
def create_output(vertices, colors, file_name):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(file_name, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

# Save the point cloud to a file
output_file = 'output_path'  #use your own output path
create_output(output_points, output_colors, output_file)
print(f"Point cloud saved to {output_file}")
