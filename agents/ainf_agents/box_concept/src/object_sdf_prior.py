#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Object SDF and Prior Functions - Compatibility Layer

This module re-exports all SDF functions from the src.sdf package for
backward compatibility. New code should import directly from src.sdf.

Example:
    # Old way (still works):
    from src.object_sdf_prior import compute_box_sdf, OBJECT_REGISTRY

    # New way (recommended):
    from src.sdf import compute_box_sdf, OBJECT_REGISTRY
"""

# Re-export everything from the new objects package
from src.objects import (
    # Registry and helpers
    OBJECT_REGISTRY,
    get_sdf_function,
    get_prior_function,
    get_object_info,
    list_object_types,

    # Constants
    SDF_SMOOTH_K,
    SDF_INSIDE_SCALE,
    TABLE_TOP_THICKNESS,
    TABLE_LEG_RADIUS,
    CHAIR_SEAT_THICKNESS,
    CHAIR_BACK_THICKNESS,

    # Box
    compute_box_sdf,
    compute_box_priors,

    # Cylinder
    compute_cylinder_sdf,
    compute_cylinder_priors,

    # Table
    compute_table_sdf,
    compute_table_priors,

    # Chair
    compute_chair_sdf,
    compute_chair_priors,

    # TV
    compute_tv_sdf,
    compute_tv_priors,
)

# For IDE autocompletion
__all__ = [
    'OBJECT_REGISTRY',
    'get_sdf_function',
    'get_prior_function',
    'get_object_info',
    'list_object_types',
    'SDF_SMOOTH_K',
    'SDF_INSIDE_SCALE',
    'TABLE_TOP_THICKNESS',
    'TABLE_LEG_RADIUS',
    'CHAIR_SEAT_THICKNESS',
    'CHAIR_BACK_THICKNESS',
    'compute_box_sdf',
    'compute_box_priors',
    'compute_cylinder_sdf',
    'compute_cylinder_priors',
    'compute_table_sdf',
    'compute_table_priors',
    'compute_chair_sdf',
    'compute_chair_priors',
    'compute_tv_sdf',
    'compute_tv_priors',
]

