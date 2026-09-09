# Objective

Separate camera/pose, global similarity alignment,
and local deformation ownership.

# Shared Constraints

- preserve CLI
- preserve positions.csv
- preserve final.obj/final.stl
- CAD is design prior
...

# Workstream 1 — Camera and Pose Audit

Type: review
Mode: read-only

Goal:
Determine camera ownership and unresolved pose uncertainty.

Inspect:
- src/pipeline.py
- src/camera_fitting.py
- src/openscan_pose.py

Deliverable:
Structured findings only. Do not modify production code.


# Workstream 2 — Geometry Ownership Audit

Type: review
Mode: read-only

Goal:
Determine where vertex offsets currently absorb global R/T/S.

Inspect:
- src/optimizer.py
- src/mesh_io.py
- src/pipeline.py

Deliverable:
Ownership analysis and proposed boundary.


# Workstream 3 — Validation Design

Type: review
Mode: read-only

Goal:
Define regression and synthetic validation before implementation.

Deliverable:
Required tests and measurable acceptance criteria.


# Workstream 4 — Similarity Alignment Implementation

Type: implement
Depends on:
- Workstream 1
- Workstream 2
- Workstream 3

Goal:
Introduce explicit global R/T/S alignment.

Preferred ownership:
- src/alignment.py
- tests/test_alignment.py


# Workstream 5 — Pipeline Integration

Type: implement
Depends on:
- Workstream 4

Goal:
Integrate:

Camera
→ Global Alignment
→ Local Deformation

Primary files:
- src/pipeline.py
- src/optimizer.py
- src/config.py


# Workstream 6 — Independent Verification

Type: review
Depends on:
- Workstream 5

Run:
- existing tests
- alignment tests
- CLI compatibility
- export validation

Do not modify production code.