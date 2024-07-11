import cv2
import torch
import numpy as np
import open3d as o3d
#if required
!pip install timm
# Define the Q matrix based on provided camera parameters
def create_q_matrix(f, cx, cy, baseline, doffs):
    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, f],
        [0, 0, -1/baseline, (doffs - cx)/baseline]
    ])
    return Q

# Load MiDaS model
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Camera parameters
f = 3997.684
cx = 1176.728
cy = 1011.728
baseline = 193.001
doffs = 131.111
Q = create_q_matrix(f, cx, cy, baseline, doffs)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    start = time.time()
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_batch=transform(img).to(device)
    with torch.no_grad():
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic"
            align_corners=False,
        ).squeeze()
        
    depth_map = prediction.cpu().numpy()
    
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    points_3D=cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=True)
    
    mask_map = depth_map>0.4
    output_points=points_3D[mask_map]
    output_colors=img[mask_map]
    
    end=time.time()
    totalTime= end-start
    
    fps = 1/totalTime
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    depth_mapp=(depth_map*255).astype(np.uint8)
    depth_map=cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    
    cv2.putText(depth_map, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)
    
    if cv2.waitKey(5) & 0xFF ==27:
        break

#creating point clouds
def create_output(vertices,colors,file_name):
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])
    
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blur
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')

output_file='pointcloud.ply'
create_output(output_points, output_colors, output_file)

cap.release()
cv2.destroyAllWindows()
