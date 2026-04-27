from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("point_cloud.ply")
mesh = o3d.io.read_triangle_mesh("mesh.ply")

o3d.visualization.draw_geometries([pcd],window_name="3D Reconstruction (Point Cloud)")

o3d.visualization.draw_geometries([mesh],window_name="3D Reconstruction (Mesh)")