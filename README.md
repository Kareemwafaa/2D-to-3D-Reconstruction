# 2D to 3D Reconstruction using Depth Estimation

## Overview

This project converts a single 2D image into a 3D representation by:

* Estimating depth from the image using a deep learning model
* Generating a point cloud
* Reconstructing a 3D mesh

It demonstrates the pipeline from **image → depth → point cloud → mesh**.

---

## Methodology

1. **Depth Estimation**

   * Model: MiDaS (DPT Large / similar)
   * Input: RGB image
   * Output: Depth map

2. **Point Cloud Generation**

   * Convert Depth into 3D coordinates

3. **Mesh Reconstruction**

   * Method: Poisson Surface Reconstruction (Open3D)


---

## Installation

```bash
pip install -r requirements.txt
```

Outputs:

* `point_cloud.ply`
* `mesh.ply`

---

## Notes

* Model weights are downloaded automatically (not included in repo)


