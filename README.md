# Description
This project leverages the MiDaS model to perform depth estimation from images and generate corresponding 3D point clouds. The depth information is visualized using Open3D and can be post-processed with various features available in Open3D.

# Key Features
 Depth Estimation: Utilizes the MiDaS model (specifically, the DPT_Large variant) to estimate depth from both uploaded images and live webcam feeds.
 3D Point Cloud Generation: Converts the estimated depth maps into 3D point clouds, which are then saved in .ply format.
 Visualization and Post-Processing: Visualizes the generated point clouds using Open3D and includes post-processing options such as filtering and transformation.
 Camera Calibration: Different Q matrix parameters or camera calibration parameters are used for different images to ensure accurate depth estimation.
