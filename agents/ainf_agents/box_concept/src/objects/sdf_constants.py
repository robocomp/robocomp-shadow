#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
SDF Configuration Constants

Shared constants used by all SDF functions.
"""

# =============================================================================
# SDF BEHAVIOR CONSTANTS
# =============================================================================

# Smooth minimum parameter for internal points (meters)
# Smaller = closer to hard min, larger = smoother gradients
SDF_SMOOTH_K = 0.02

# Scale factor for internal points (0-1)
# Reduces influence of internal points which are less reliable
SDF_INSIDE_SCALE = 0.3
