# Requirements: Ground-Level Camera Ingestion and Planar Homography Calibration

## Feature Metadata
- **ID:** SVA-PRE-002
- **Name:** camera_rtsp_and_homography_calibration
- **Title:** Ground-Level Camera Ingestion and Planar Homography Calibration
- **Target Subsystem:** `src/pipeline/` (Calibration & Homography Projection Engine)
- **Configuration Target:** `configs/homography_config.json`

## Acceptance Criteria Mapping
| Acceptance Criterion | Requirements |
|---|---|
| Interactive / programmatic calibration utility to select 4 ground control points (GCPs) on the pavement to compute the 3x3 homography matrix H | R1, R2, R3, R4, R5 |
| Generation and persistence of `configs/homography_config.json` storing matrix H, lane polygon boundaries, and directional vector thresholds | R6, R7, R8, R9, R15, R16 |
| Definition of mathematical unit vector u_legal for vehicular and micromobility lanes to detect counter-flow movement (cosine similarity formula) | R8, R10, R11, R12 |
| Validation of projected world coordinate accuracy with test pavement calibration measurements (< 5% distance error) | R13, R14 |

---

## Requirements (Strict EARS Notation)

### R1 (Ubiquitous)
The system SHALL implement a planar homography transformation mapping 2D camera image coordinates $(u, v)$ to real-world metric ground coordinates $(X, Y)$ in meters via a 3x3 projection matrix $\mathbf{H}$ and its inverse $\mathbf{H}^{-1}$.

### R2 (Ubiquitous)
The system SHALL compute the 3x3 homography matrix $\mathbf{H}$ from four non-collinear Ground Control Points (GCPs) on the pavement surface using the Direct Linear Transformation (DLT) algorithm.

### R3 (Event-driven)
WHEN programmatic calibration is invoked with four pixel coordinates $(u_i, v_i)$ and four corresponding metric ground coordinates $(X_i, Y_i)$, the system SHALL calculate the normalized homography matrix $\mathbf{H}$ such that $H_{3,3} = 1.0$.

### R4 (Optional)
WHERE interactive calibration mode is enabled, the system SHALL render a GUI window displaying the camera frame to allow manual selection of four GCP corners on the pavement and prompt for real-world metric dimensions.

### R5 (Unwanted)
IF the selected ground control points are collinear, degenerate, or result in a singular matrix where $|\det(\mathbf{H})| < 10^{-7}$, THEN the system SHALL raise a `CalibrationError` and abort the calibration update without writing to disk.

### R6 (Ubiquitous)
The system SHALL serialize calibration results to `configs/homography_config.json` formatted according to a validated Pydantic schema containing the 3x3 matrix $\mathbf{H}$, its inverse $\mathbf{H}^{-1}$, camera identifier, source resolution, and calibration timestamp.

### R7 (Ubiquitous)
The system SHALL persist lane polygon boundaries for both vehicular and micromobility lanes in `configs/homography_config.json` represented as 2D coordinate lists in both pixel and metric world domains.

### R8 (Ubiquitous)
The system SHALL define and persist a normalized 2D legal direction unit vector $\mathbf{u}_{legal} = (u_x, u_y)$ satisfying $\|\mathbf{u}_{legal}\| = 1.0 \pm 10^{-6}$ for each configured lane in `configs/homography_config.json`.

### R9 (Ubiquitous)
The system SHALL persist directional vector thresholds in `configs/homography_config.json` including a counter-flow cosine similarity threshold ($\cos \theta < -0.7$), a minimum persistence duration threshold ($t_{min} \ge 1.0\text{ s}$), and a minimum speed filtering threshold ($v_{min} \ge 0.5\text{ m/s}$).

### R10 (Event-driven)
WHEN an object bounding box foot-point $(u_{mid}, v_{max})$ is received, the system SHALL project the pixel coordinates into metric ground plane coordinates $(X, Y)$ using the forward homography matrix $\mathbf{H}$.

### R11 (Event-driven)
WHEN two successive metric ground positions $\mathbf{P}_t$ and $\mathbf{P}_{t-\Delta t}$ of a tracked object are registered, the system SHALL compute the ground velocity vector $\mathbf{v}_t$ and evaluate the cosine similarity metric $\cos(\theta) = \frac{\mathbf{v}_t \cdot \mathbf{u}_{legal}}{\|\mathbf{v}_t\| \|\mathbf{u}_{legal}\|}$.

### R12 (State-driven)
WHILE a tracked vehicle or micromobility object resides within a designated lane polygon and has speed $\|\mathbf{v}_t\| \ge v_{min}$, the system SHALL flag a candidate counter-flow infraction if $\cos(\theta) < -0.7$.

### R13 (Event-driven)
WHEN validation is executed against known test pavement reference distances, the system SHALL calculate the relative distance error percentage between projected metric points and physical ground truth distances: $\text{Error} = \frac{|\|\mathbf{P}_A - \mathbf{P}_B\| - D_{ground}|}{D_{ground}} \times 100\%$.

### R14 (Unwanted)
IF the relative distance error across test pavement calibration markers exceeds 5.0%, THEN the system SHALL fail calibration validation and raise a `CalibrationAccuracyError`.

### R15 (Event-driven)
WHEN the pipeline loads `configs/homography_config.json` during initialization, the system SHALL validate all matrix values, lane polygons, and unit vectors against the `HomographyConfig` Pydantic model before initializing the projection engine.

### R16 (Unwanted)
IF `configs/homography_config.json` is missing, malformed, or fails Pydantic schema validation, THEN the system SHALL raise a `ConfigurationError` and halt pipeline startup.
