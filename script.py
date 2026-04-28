from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d
import numpy as np

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
torch.hub.set_dir(str(PROJECT_DIR / ".torch-cache"))

IMAGE_PATH = r"C:\Users\Karee\Desktop\2D-to-3D-Reconstruction\data\spongebob.jpg"
OUTPUT_PCD = r"C:\Users\Karee\Desktop\2D-to-3D-Reconstruction\output\point_cloud.ply"
OUTPUT_MESH = r"C:\Users\Karee\Desktop\2D-to-3D-Reconstruction\output\mesh.ply"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Depth Estimation (MiDaS)
def load_depth_map(img_rgb: np.ndarray) -> np.ndarray:

    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

    model.to(DEVICE)
    model.eval()

    input_tensor = transforms.dpt_transform(img_rgb).to(DEVICE)

    with torch.no_grad():
        depth = model(input_tensor)
        depth = F.interpolate(depth.unsqueeze(1),size=img_rgb.shape[:2],mode="bicubic",align_corners=False,).squeeze()

    depth = depth.cpu().numpy()

    # normalize depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    print("Depth estimated")
    return depth

# 3D Reconstruction 
def build_point_cloud(img_rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
    h, w = depth.shape

    fx = fy = 1.2 * w
    cx, cy = w / 2, h / 2

    # Smooth depth slightly
    depth = cv2.bilateralFilter(depth.astype(np.float32), d=7, sigmaColor=0.1, sigmaSpace=7)

    # Removing sharp depth edges
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

    edge_threshold = np.quantile(grad, 0.99)

    points = []
    colors = []
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    foreground_mask = gray < 245

    for v in range(h):
        for u in range(w):
            if not foreground_mask[v, u]:
                continue

            if grad[v, u] > edge_threshold:
                continue

            z = 1.0 + depth[v, u] * 4.0

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, -y, z])
            colors.append(img_rgb[v, u] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))

    return pcd

# Mesh Reconstruction 
def reconstruct_mesh(pcd: o3d.geometry.PointCloud):
    pcd = pcd.voxel_down_sample(voxel_size=0.003)

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=1.5
    )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,
            max_nn=120
        )
    )

    pcd.orient_normals_consistent_tangent_plane(100)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=12,
        width=0,
        scale=1.1,
        linear_fit=True
    )

    densities = np.asarray(densities)

    vertices_to_remove = densities < np.quantile(densities, 0.03)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    mesh.compute_vertex_normals()

    print("High-quality Poisson reconstruction done")
    return mesh

# Main Pipeline
def main():
    print("here")
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
    #print(f"Saved point cloud → {OUTPUT_PCD}")o3d.io.write_point_cloud(str(OUTPUT_PCD),pcd,write_ascii=True)

    # Mesh
    mesh = reconstruct_mesh(pcd)
    #Save mesh
    #o3d.io.write_triangle_mesh(str(OUTPUT_MESH),mesh,write_ascii=True)
    o3d.io.write_triangle_mesh(str(OUTPUT_MESH), mesh)
    print(f"Saved mesh → {OUTPUT_MESH}")
    
    #depth map
    depth_vis = (depth * 255).astype(np.uint8)
    cv2.imwrite(str(PROJECT_DIR / "output" / "depth_map.png"),(depth * 255).astype(np.uint8))
    #  Visualization
    o3d.visualization.draw_geometries([pcd],window_name="3D Reconstruction (Point Cloud)")
    o3d.visualization.draw_geometries([mesh],window_name="3D Reconstruction (Mesh)")
    o3d.visualization.draw_geometries([o3d.geometry.Image(depth_vis)],window_name="Estimated Depth Map")

if __name__ == "__main__":
    main()
