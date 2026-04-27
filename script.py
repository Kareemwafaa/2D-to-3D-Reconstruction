from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d
import numpy as np

# Paths

PROJECT_DIR = Path(__file__).resolve().parent
torch.hub.set_dir(str(PROJECT_DIR / ".torch-cache"))

IMAGE_PATH = r"C:\Users\Karee\Desktop\3D Vision\Proj2\data\images (2).jpg"
OUTPUT_PCD = r"C:\Users\Karee\Desktop\3D Vision\Proj2\output\point_cloud.ply"
OUTPUT_MESH = r"C:\Users\Karee\Desktop\3D Vision\Proj2\output\mesh.ply"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Depth Estimation (MiDaS)
def load_depth_map(img_rgb: np.ndarray) -> np.ndarray:

    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

    model.to(DEVICE)
    model.eval()

    input_tensor = transforms.small_transform(img_rgb).to(DEVICE)

    with torch.no_grad():
        depth = model(input_tensor)
        depth = F.interpolate(depth.unsqueeze(1),size=img_rgb.shape[:2],mode="bicubic",align_corners=False,).squeeze()

    depth = depth.cpu().numpy()

    # normalize depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    print("Depth estimated")
    return depth

# 3D Reconstruction (Correct Projection)
def build_point_cloud(img_rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
    h, w = depth.shape

    fx = fy = 0.9 * w
    cx, cy = w / 2, h / 2

    step = 1  # decrase = denser cloud

    points = []
    colors = []

    for v in range(0, h, step):
        for u in range(0, w, step):
            #z = depth[v, u] * 5.0  
            z = 1.0 + depth[v, u] * 4.0
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, -y, z])  
            colors.append(img_rgb[v, u] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    return pcd

# Mesh Reconstruction 
def reconstruct_mesh(pcd: o3d.geometry.PointCloud):
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    print("Poisson surface reconstruction done")


    mesh.compute_vertex_normals()
    return mesh

# Main Pipeline
def main():
    print(f"Using device: {DEVICE}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    img = cv2.imread(str(IMAGE_PATH))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Depth
    depth = load_depth_map(img_rgb)

    # Point Cloud
    pcd = build_point_cloud(img_rgb, depth)

    # Save point cloud
    o3d.io.write_point_cloud(str(OUTPUT_PCD), pcd)
    print(f"Saved point cloud → {OUTPUT_PCD}")

    # Mesh
    mesh = reconstruct_mesh(pcd)
    o3d.io.write_triangle_mesh(str(OUTPUT_MESH), mesh)
    print(f"Saved mesh → {OUTPUT_MESH}")
    
    #depth map
    depth_vis = (depth * 255).astype(np.uint8)

    #  Visualization
    o3d.visualization.draw_geometries([pcd],window_name="3D Reconstruction (Point Cloud)")
    o3d.visualization.draw_geometries([mesh],window_name="3D Reconstruction (Mesh)")
    o3d.visualization.draw_geometries([o3d.geometry.Image(depth_vis)],window_name="Estimated Depth Map")
if __name__ == "__main__":
    main()
