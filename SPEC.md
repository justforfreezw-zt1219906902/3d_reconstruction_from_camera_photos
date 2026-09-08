# Reconstruction Architecture Refactor

## Objective

Refactor the reconstruction pipeline so that camera/pose parameters,
global CAD-to-real alignment, and local geometry deformation have explicit
and separate ownership.

The main architectural problem to solve is that local per-vertex deformation
must not absorb unresolved global rotation, translation, or scale errors.

## Current problem

The current pipeline fits or selects camera parameters, freezes the camera,
prepares the CAD proxy, and then optimizes mesh geometry.

The geometry optimizer currently exposes per-vertex offsets directly.

This allows local geometry deformation to compensate for errors that should
belong to a global CAD-to-real alignment stage.

## Target architecture

The intended optimization ownership is:

Camera / Pose
    ↓
Global CAD-to-real Similarity Alignment
    ↓
Coarse / Fine / Local Geometry Deformation
    ↓
Final Mesh

### Camera / Pose

Own camera projection and camera pose/calibration parameters.

Camera uncertainty must remain conceptually separate from object geometry.

### Global alignment

Introduce an explicit CAD-to-real similarity transform with ownership of:

- rotation
- translation
- uniform scale

The transform should operate between CAD proxy preparation and local
geometry deformation.

### Local deformation

Local or per-vertex deformation should represent residual physical shape
differences only after global alignment has been accounted for.

## Scope

Primary implementation areas are expected to include:

- `src/pipeline.py`
- `src/optimizer.py`
- `src/config.py`

A dedicated alignment module may be introduced, for example:

- `src/alignment.py`

Add or update tests as required.

Do not assume these files must all change. Inspect the current repository
before deciding the final implementation boundary.

## Required investigation

Before implementation, establish:

1. How camera parameters and pose conventions currently flow through the pipeline.
2. Which camera parameters are frozen before geometry optimization.
3. How the CAD proxy is normalized and transformed.
4. Which variables the geometry optimizer currently owns.
5. Where unresolved global rotation, translation, or scale can currently
   leak into vertex offsets.
6. Which existing tests constrain current behavior.

## Compatibility requirements

Preserve existing behavior and interfaces unless explicitly required by this
refactor.

In particular preserve:

- `main.py` as the CLI entry point
- `positions.csv`
- RGBA input handling
- CAD/STL input handling
- output directory conventions
- `final.obj`
- `final.stl`

Preserve CLI behavior for:

- `--validate-only`
- `--camera-fit-only`
- `--skip-camera-fit`

## Validation requirements

The completed implementation should provide evidence for at least:

1. Existing relevant tests continue to pass.
2. CLI compatibility is preserved.
3. An identity global transform does not alter geometry.
4. Synthetic known similarity transforms can be represented correctly.
5. Global R/T/S parameters are explicitly separate from local vertex offsets.
6. The final mesh export path remains operational.
7. Critical final validation is run against the final candidate code.

## Multi-agent execution strategy

Use parallel agents primarily for independent investigation and verification.

Recommended initial workstreams:

- camera / pose audit
- geometry / optimizer audit
- validation and regression-test design

Do not let multiple implementation agents concurrently edit the same core
pipeline files.

After investigation, perform implementation according to the dependency graph
derived from the findings.

Use an independent review or verification task after integration.

## Completion criteria

The task is complete when:

- the ownership boundary is explicit in code;
- global similarity alignment owns R/T/S;
- local deformation owns residual shape change;
- compatibility requirements remain satisfied;
- tests and validation evidence support the final implementation;
- the resulting implementation is committed to Git.