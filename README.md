# Particle Tracking Velocimetry (PTV) – Relaxation-Based Pair Matching
MATLAB → Python Translation

## Overview

This project is a direct Python translation of a MATLAB-based Particle Tracking Velocimetry (PTV) matching library. 

The purpose of the code is to match tracer particles between successive image frames in PTV experiments, 
especially in cases where particle density is high, motion is correlated, or noise makes simple nearest-neighbor matching unreliable.

The code implements a probabilistic, relaxation-based particle matching algorithm that enforces local geometric consistency between particles. 
This approach is widely used in advanced PTV and µPTV workflows, bead tracking, and deformation measurements.

This repository focuses on the particle matching stage of PTV:
1. Particle detection is assumed to be done upstream.
2. This code takes particle coordinates from two frames.
3. It matches particles.
4. The output can be used to compute particle displacements, velocities, and trajectories.

---

## Particle Tracking Velocimetry

Particle Tracking Velocimetry (PTV) is a Lagrangian flow measurement technique used in experimental fluid mechanics, soft matter, and microscopy. 
Tracer particles are seeded into a flow and imaged over time. Individual particle positions are detected in each frame and then matched across frames
to reconstruct particle trajectories.

From these trajectories, velocities and accelerations are computed.

Unlike traditional Particle Image Velocimetry (PIV) methods, which computes average velocities in interrogation windows, 
this version of PTV tracks individual particles, providing higher spatial resolution and access to Lagrangian statistics.

The most challenging step in PTV is particle matching between frames, especially when:
- Particle spacing is comparable to particle displacement
- Flow contains shear, rotation, or turbulence
- Noise or missed detections occur
- Particle density is high

---

## Main Entry Point (Main File)

The main entry point of this project is:

match_pair.py

This file plays the role of `main.py` in a typical Python project, even though it is named differently for MATLAB compatibility.

In MATLAB, the primary user-facing function was:
match_pair.m

In Python, you call:
from match_pair import match_pair

All other files are supporting modules and should not be called directly by the user in a normal PTV workflow.

---

## High-Level Algorithm Description

The algorithm matches particles between two frames (Frame A and Frame B) using a relaxation labeling approach:

1. For each particle in Frame A, find nearby candidate particles in Frame B.
2. Initialize match probabilities for each candidate.
3. Enforce local geometric consistency by comparing particle neighborhoods.
4. Iteratively update probabilities based on neighbor agreement.
5. Perform matching in both directions (A → B and B → A).
6. Accept only mutually consistent matches with high confidence.
7. Output matched pairs and unmatched particles.

This approach assumes that local particle configurations deform smoothly between frames, which is valid for most physical flows at sufficiently small time steps.

---

## File Descriptions

### match_pair.py (MAIN FILE)

The main user-facing PTV matching function. It:
- Validates inputs
- Loads or computes preprocessing data
- Calls the relaxation-based matching algorithm
- Handles result caching and reloading
- Returns matched indices, displacements, and confidence values

This file orchestrates the entire particle matching process.

---

### match_pairRelax4.py

This file implements the relaxation-based matching algorithm. It:
- Performs forward matching (Frame A → Frame B)
- Performs backward matching (Frame B → Frame A)
- Enforces mutual consistency
- Filters matches based on distance and confidence thresholds

This is the core PTV matching engine.

---

### match_pair_preprocess.py

This file performs preprocessing steps:
- Nearest-neighbor searches
- Estimation of inter-particle distances
- Construction of candidate match lists

In PTV terms, it defines the spatial neighborhoods used to enforce local flow consistency.

---

### LinkPairsF2L.py

This module links pairwise matches across many frames to form full particle trajectories. It converts frame-to-frame matches into long Lagrangian tracks.

---

### crack_Link2FrameId.py

This utility extracts start and end frame indices from particle trajectories. It is used for filtering, trajectory trimming, and analysis.

---

### setParameter.py

A small utility function that emulates MATLAB-style optional parameter handling. It allows default values to be overridden by user-provided options.

---

## Output Variables and Their PTV Meaning

- I1, I2: Indices of matched particles in Frame A and Frame B
- I1u, I2u: Indices of unmatched particles
- dx, dy, dr: Particle displacements
- confidence1, confidence2: Match confidence values
- meanDist: Average nearest-neighbor distance (particle spacing)
- drift: Estimated global drift (if present)

Particle velocities are obtained by dividing displacements by the frame time separation.

---

## Auxiliary and Result Folders

### aux_matchpair/

This folder contains intermediate data:
- Preprocessing results
- Neighborhood definitions
- Relaxation probabilities
- Backup files for long computations

These files exist to improve performance and allow recovery from interrupted runs, which is common for large PTV datasets.

---

### res_matchpair/

This folder stores final matching results:
- Matched particle indices
- Displacements
- Confidence metrics

These files can be reused to avoid recomputing matches.

## Summary

This project implements a robust particle matching algorithm for Particle Tracking Velocimetry (PTV). It is designed for dense, noisy experimental data where simple matching methods fail. The relaxation-based approach leverages local geometric consistency to produce reliable particle tracks suitable for velocity, deformation, and flow analysis.

Main file: match_pair.py  
Application domain: Particle Tracking Velocimetry (PTV)  
Algorithm: Probabilistic relaxation labeling with neighborhood consistency
