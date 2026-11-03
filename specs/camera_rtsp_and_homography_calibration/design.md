# Design: Ground-Level Camera Ingestion and Planar Homography Calibration

## Feature Metadata
- **ID:** SVA-PRE-002
- **Name:** camera_rtsp_and_homography_calibration
- **Status:** spec_ready

---

## 1. Architectural Overview

The SVA Sentinel system monitors residential intersections to detect wrong-way vehicular and micromobility traffic (automobiles, motorcycles, bicycles, electric scooters). In camera image pixel space $(u, v)$, severe perspective foreshortening distorts object sizes, trajectory angles, and apparent speeds. Vehicles moving toward or away from the camera produce large vertical pixel displacements with minimal lateral shifts, making isotropic directional analysis in pixel coordinates geometrically invalid.

To overcome this, Feature `SVA-PRE-002` establishes a **Planar Homography Calibration Engine** that maps the 2D pavement image plane directly into real-world metric ground coordinates $(X, Y)$ in meters. 

```
                                CAMERA PERSPECTIVE VIEW (u, v)
                                +-------------------------------+
                                | (u0, v0)             (u1, v1) |
                                |    \                    /     |
                                |     \   LANE POLYGON   /      |
                                |      \                /       |
                                |    (u3, v3)-------(u2, v2)    |
                                +-------------------------------+
                                                |
                                                |  Homography Matrix H (3x3)
                                                v
                              BIRD'S-EYE VIEW / METRIC PLANE (X, Y)
                                +-------------------------------+
                                | (X0, Y0)             (X1, Y1) |
                                |    |                    |     |
                                |    |     u_legal        |     |
                                |    |       ^            |     |
                                |    |       |            |     |
                                | (X3, Y3)---|----------(X2, Y2)|
                                +-------------------------------+
```

This subsystem provides:
1. **Mathematical Projection Engine:** Fast forward and inverse coordinate transformations via a 3x3 homography matrix $\mathbf{H}$ and its inverse $\mathbf{H}^{-1}$.
2. **Kinematic Vector Engine:** Evaluation of vehicle foot-point trajectories, smoothed ground velocity vectors $\mathbf{v}_t$, and directional cosine similarity against authorized legal unit vectors $\mathbf{u}_{legal}$.
3. **Calibration Utilities:** Programmatic and interactive calibration routines to calculate $\mathbf{H}$ from 4 surveyed Ground Control Points (GCPs) with strict $< 5.0\%$ pavement distance error verification.
4. **Declarative Persistence:** Storage of calibration matrices, lane polygons, and directional thresholds in `configs/homography_config.json` guarded by Pydantic runtime schema validation.

---

## 2. Mathematical Formulation

### 2.1 Planar Homography Projection

Given a ground-level image pixel coordinate $\mathbf{p} = [u, v, 1]^T$ and its corresponding real-world ground coordinate $\mathbf{P} = [X, Y, 1]^T$ in meters, the projective transformation is governed by:

$$\begin{bmatrix} x' \\ y' \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{00} & h_{01} & h_{02} \\ h_{10} & h_{11} & h_{12} \\ h_{20} & h_{21} & h_{22} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

De-homogenization converts the projective coordinates into physical Euclidean ground coordinates:

$$X = \frac{x'}{w} = \frac{h_{00}u + h_{01}v + h_{02}}{h_{20}u + h_{21}v + h_{22}}$$

$$Y = \frac{y'}{w} = \frac{h_{10}u + h_{11}v + h_{12}}{h_{20}u + h_{21}v + h_{22}}$$

The matrix is normalized by convention such that $h_{22} = 1.0$.

### 2.2 Inverse Coordinate Transformation

To map metric world positions (or planned BEV lane overlays) back onto the camera frame, the system utilizes the analytical matrix inverse $\mathbf{H}^{-1}$:

$$\begin{bmatrix} u' \\ v' \\ w' \end{bmatrix} = \mathbf{H}^{-1} \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}$$

$$u = \frac{u'}{w'}, \quad v = \frac{v'}{w'}$$

The inverse is computed during calibration and serialized alongside $\mathbf{H}$ to avoid redundant runtime matrix inversion during edge inference loops.

### 2.3 Direct Linear Transformation (DLT) from 4 GCPs

Each pair of corresponding points $(u_i, v_i) \leftrightarrow (X_i, Y_i)$ yields two linearly independent equations:

$$\begin{bmatrix} -u_i & -v_i & -1 & 0 & 0 & 0 & u_i X_i & v_i X_i & X_i \\ 0 & 0 & 0 & -u_i & -v_i & -1 & u_i Y_i & v_i Y_i & Y_i \end{bmatrix} \mathbf{h} = \mathbf{0}$$

where $\mathbf{h} = [h_{00}, h_{01}, h_{02}, h_{10}, h_{11}, h_{12}, h_{20}, h_{21}, h_{22}]^T$. 

Four non-collinear Ground Control Points provide an $8 \times 9$ matrix $\mathbf{A}$. The solution vector $\mathbf{h}$ is the singular vector corresponding to the smallest singular value obtained via Singular Value Decomposition (SVD): $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$, yielding $\mathbf{h} = \mathbf{V}_{:, 9}$.

### 2.4 Bounding Box Foot-Point Selection

Vehicles and micromobility devices are 3D objects. Projecting the bounding box center $(u_c, v_c)$ yields significant parallax elevation error because the centroid is elevated above the road plane. 

To ensure exact planar contact, the system projects the **bottom-center foot-point** of the detected bounding box $[x_1, y_1, x_2, y_2]$:

$$u_{foot} = \frac{x_1 + x_2}{2}, \quad v_{foot} = y_2$$

This foot-point corresponds directly to tire/wheel contact with the pavement plane.

### 2.5 Metric Velocity and Cosine Similarity Direction Engine

Let $\mathbf{P}_t = (X_t, Y_t)$ and $\mathbf{P}_{t-\Delta t} = (X_{t-\Delta t}, Y_{t-\Delta t})$ be consecutive metric ground positions over temporal delta $\Delta t$ seconds:

$$\mathbf{v}_t = \frac{\mathbf{P}_t - \mathbf{P}_{t-\Delta t}}{\Delta t} = \left( \frac{X_t - X_{t-\Delta t}}{\Delta t}, \frac{Y_t - Y_{t-\Delta t}}{\Delta t} \right)$$

Speed in meters per second is:

$$s_t = \|\mathbf{v}_t\| = \sqrt{v_{x}^2 + v_{y}^2}$$

For each designated lane, the authorized flow of traffic is governed by a normalized unit vector:

$$\mathbf{u}_{legal} = \frac{\mathbf{v}_{authorized}}{\|\mathbf{v}_{authorized}\|}, \quad \|\mathbf{u}_{legal}\| = 1.0$$

The instantaneous directional alignment is evaluated using the cosine similarity dot product:

$$\cos(\theta) = \frac{\mathbf{v}_t \cdot \mathbf{u}_{legal}}{\|\mathbf{v}_t\| \|\mathbf{u}_{legal}\|} = \frac{v_x u_x + v_y u_y}{\sqrt{v_x^2 + v_y^2}}$$

#### Threshold Criteria:
1. **Stationary/Noise Filter:** If $s_t < v_{min}$ (where $v_{min} = 0.5\text{ m/s} \approx 1.8\text{ km/h}$), direction is undefined and flagged as stationary.
2. **Authorized Traffic Flow:** $\cos(\theta) \ge 0.5$ ($\theta \le 60^\circ$).
3. **Cross-Traffic / Turning Movement:** $-0.5 \le \cos(\theta) < 0.5$ ($60^\circ < \theta \le 120^\circ$).
4. **Counter-Flow / Wrong-Way Violation:** $\cos(\theta) < -0.7$ ($\theta > 134.4^\circ$, approaching antiparallel $-1.0$).
5. **Temporal Persistence Gate:** Infraction alerts are triggered only when $\cos(\theta) < -0.7$ persists for duration $t \ge t_{min}$ (default: $1.0\text{ s}$).

### 2.6 Pavement Calibration Accuracy Metric

To validate calibration fidelity, surveyed ground truth distances $D_{ground}$ between physical pavement markers (e.g., crosswalk stripe length, lane width) are compared against the Euclidean distance computed between their projected image coordinates:

$$D_{projected} = \|\mathbf{P}_A - \mathbf{P}_B\| = \sqrt{(X_A - X_B)^2 + (Y_A - Y_B)^2}$$

$$\text{Error}_{\%} = \frac{|D_{projected} - D_{ground}|}{D_{ground}} \times 100\%$$

**Acceptance Ceiling:** Calibration is deemed valid if and only if $\text{Error}_{\%} < 5.0\%$.

---

## 3. Configuration Schema & Persistence

Configuration is stored in `configs/homography_config.json` and validated by Pydantic models.

### 3.1 Pydantic Model Hierarchy (`src/pipeline/schemas.py`)

```python
from pydantic import BaseModel, Field

class Point2D(BaseModel):
    x: float
    y: float

class LaneConfig(BaseModel):
    lane_id: str = Field(..., description="Unique lane identifier")
    lane_type: str = Field(..., description="vehicular, micromobility, or mixed")
    polygon_pixels: list[list[float]] = Field(..., description="List of [u, v] polygon vertices")
    polygon_meters: list[list[float]] = Field(..., description="List of [X, Y] metric vertices")
    legal_direction_vector: list[float] = Field(..., description="Normalized [dx, dy] unit vector")
    counterflow_cosine_threshold: float = Field(default=-0.7, description="Cosine threshold for wrong-way detection")
    min_speed_threshold_mps: float = Field(default=0.5, description="Minimum speed in m/s to evaluate direction")
    min_violation_duration_sec: float = Field(default=1.0, description="Minimum duration in seconds to trigger alert")

class CalibrationMetadata(BaseModel):
    camera_id: str
    calibration_date: str
    image_resolution: list[int] = Field(..., description="[width, height] in pixels")
    ground_control_points_pixels: list[list[float]] = Field(..., description="4 [u, v] pixel points")
    ground_control_points_meters: list[list[float]] = Field(..., description="4 [X, Y] world metric points")
    reprojection_error_px: float
    pavement_distance_error_percent: float

class HomographyConfig(BaseModel):
    version: str = "1.0.0"
    metadata: CalibrationMetadata
    homography_matrix: list[list[float]] = Field(..., description="3x3 forward projection matrix")
    inverse_homography_matrix: list[list[float]] = Field(..., description="3x3 inverse projection matrix")
    lanes: list[LaneConfig]
```

### 3.2 Canonical JSON Structure (`configs/homography_config.json`)

```json
{
  "version": "1.0.0",
  "metadata": {
    "camera_id": "cam_residential_intersection_01",
    "calibration_date": "2026-11-01T12:00:00Z",
    "image_resolution": [1920, 1080],
    "ground_control_points_pixels": [
      [580.0, 620.0],
      [1340.0, 620.0],
      [1720.0, 980.0],
      [200.0, 980.0]
    ],
    "ground_control_points_meters": [
      [0.0, 12.0],
      [7.0, 12.0],
      [7.0, 0.0],
      [0.0, 0.0]
    ],
    "reprojection_error_px": 0.82,
    "pavement_distance_error_percent": 1.45
  },
  "homography_matrix": [
    [0.00512, -0.00018, -2.85],
    [0.00004, 0.01240, -7.42],
    [0.00001, 0.00195, 1.0]
  ],
  "inverse_homography_matrix": [
    [210.45, 12.30, 680.12],
    [-2.15, 92.40, 580.45],
    [-0.001, -0.180, 1.0]
  ],
  "lanes": [
    {
      "lane_id": "vehicular_northbound",
      "lane_type": "vehicular",
      "polygon_pixels": [[580, 620], [960, 620], [960, 980], [200, 980]],
      "polygon_meters": [[0.0, 12.0], [3.5, 12.0], [3.5, 0.0], [0.0, 0.0]],
      "legal_direction_vector": [0.0, 1.0],
      "counterflow_cosine_threshold": -0.7,
      "min_speed_threshold_mps": 0.5,
      "min_violation_duration_sec": 1.0
    },
    {
      "lane_id": "micromobility_southbound",
      "lane_type": "micromobility",
      "polygon_pixels": [[960, 620], [1340, 620], [1720, 980], [960, 980]],
      "polygon_meters": [[3.5, 12.0], [7.0, 12.0], [7.0, 0.0], [3.5, 0.0]],
      "legal_direction_vector": [0.0, -1.0],
      "counterflow_cosine_threshold": -0.7,
      "min_speed_threshold_mps": 0.5,
      "min_violation_duration_sec": 1.0
    }
  ]
}
```

---

## 4. File Structure & Component Signatures

### 4.1 Files to Create

1. `src/pipeline/schemas.py`: Pydantic validation schemas for calibration data, lane geometries, and direction thresholds.
2. `src/pipeline/homography.py`: Core mathematical projection engine:
   - `HomographyProjector` class:
     - `project_pixel_to_metric(u: float, v: float) -> tuple[float, float]`
     - `project_metric_to_pixel(x: float, y: float) -> tuple[float, float]`
     - `project_batch_pixels_to_metric(pts: np.ndarray) -> np.ndarray`
     - `project_bbox_footpoint(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]`
     - `calculate_velocity(p1: tuple[float, float], p2: tuple[float, float], dt: float) -> tuple[float, float, float]`
     - `calculate_cosine_similarity(v: tuple[float, float], u_legal: tuple[float, float]) -> float`
     - `is_counterflow(v: tuple[float, float], u_legal: tuple[float, float], threshold: float = -0.7) -> bool`
3. `src/pipeline/calibration.py`: Calibration engine:
   - `compute_homography_matrix(src_pixels: list[tuple[float, float]], dst_meters: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]`
   - `validate_calibration_accuracy(H: np.ndarray, test_pairs_px: list[tuple[tuple[float, float], tuple[float, float]]], test_dists_meters: list[float], max_error_percent: float = 5.0) -> float`
   - `save_homography_config(config_path: Path | str, config: HomographyConfig) -> None`
   - `load_homography_config(config_path: Path | str) -> HomographyConfig`
   - `run_interactive_calibration(...)`: OpenCV mouse click callback utility to capture 4 GCPs.
4. `configs/homography_config.json`: Canonical reference calibration file.
5. `tests/test_homography_math.py`: Unit tests validating projection math, inverse projection consistency, foot-point projection, velocity calculation, and cosine similarity.
6. `tests/test_calibration_engine.py`: Unit tests validating 4-point GCP matrix computation, distance accuracy error verification (< 5%), degenerate point detection, and JSON schema persistence.

### 4.2 Files to Mutate

1. `src/core/exceptions.py`: Add domain-specific exceptions:
   - `CalibrationError(FrameworkError)`: Base error for calibration failures.
   - `CalibrationAccuracyError(CalibrationError)`: Raised when test distance error exceeds 5%.
   - `ConfigurationError(FrameworkError)`: Raised when config file is invalid or missing.

---

## 5. Discarded Alternatives

### Discarded Alternative 1: Full 3D Camera Pose Estimation (PnP / Extrinsics + Intrinsics)
- **Description:** Calibrating camera intrinsics via a checkerboard pattern, finding camera extrinsics (rotation $\mathbf{R}$, translation $\mathbf{T}$) relative to an intersection origin, and projecting points using the full pinhole projection equation: $\mathbf{x} = \mathbf{K} [\mathbf{R} | \mathbf{T}] \mathbf{X}$.
- **Why Discarded:** In a deployed residential environment, camera poles lack physical calibration targets for 3D extrinsics, and intrinsic focal lengths vary with lens zoom. Ground traffic (cars, bicycles, scooters) moves entirely on the 2D planar roadway surface $(Z = 0)$. Planar homography directly models the projectivity between two planes with 8 degrees of freedom, requiring only 4 surveyed pavement points and avoiding unconstrained 3D pose optimization, camera intrinsics estimation, or heavy 3D vision toolchains.

### Discarded Alternative 2: Direction Estimation in Pixel Space without Homography
- **Description:** Calculating vehicle heading vectors directly from bounding box pixel centroids $[u_t - u_{t-k}, v_t - v_{t-k}]$ and comparing against 2D pixel lane arrows.
- **Why Discarded:** Perspective distortion renders pixel-space angle metrics non-linear. A vehicle traveling diagonally toward the camera at 45 degrees in world coordinates projects to near-vertical pixel displacement ($80^\circ+$ in image space). Evaluating cosine similarity in image space produces severe false positives and false negatives across different parts of the camera FOV. Transforming coordinates into metric Euclidean space $(X, Y)$ normalizes spatial orientation, ensuring uniform angle and velocity calculations everywhere in the intersection.

### Discarded Alternative 3: Dynamic Online Auto-Calibration via Deep Learning Horizon / Vanishing Point Networks
- **Description:** Employing a convolutional neural network (e.g. NeurVPS or deep horizon estimator) to continuously predict the horizon line and vanishing points from street imagery to dynamically update homography.
- **Why Discarded:** Deep neural networks for vanishing point estimation add 300MB–500MB of memory overhead and non-deterministic jitter, violating the strict 8GB total RAM budget of the local edge container. Furthermore, residential roads frequently have occluded horizons due to trees, residential housing, and curved roadways, which destabilize neural vanishing point detection. Static 4-point GCP calibration stored in declarative JSON is deterministic, zero-memory, mathematically exact, and impervious to horizon occlusions.
