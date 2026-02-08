#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Objects Package - Central registry for all object types.

Each object type is in its own folder with:
- sdf.py: SDF and prior functions
- belief.py: Belief class
- manager.py: Manager class

Usage:
    from src.objects import OBJECT_REGISTRY, get_sdf_function
    from src.objects.box import BoxBelief, BoxManager
    from src.objects.table import TableBelief, TableManager
"""

from typing import Dict, Callable, Any

# Import from each object folder
from src.objects.box.sdf import (
    compute_box_sdf,
    compute_box_priors,
    BOX_PARAM_COUNT,
    BOX_PARAM_NAMES,
)

from src.objects.cylinder.sdf import (
    compute_cylinder_sdf,
    compute_cylinder_priors,
    CYLINDER_PARAM_COUNT,
    CYLINDER_PARAM_NAMES,
)

from src.objects.table.sdf import (
    compute_table_sdf,
    compute_table_priors,
    TABLE_PARAM_COUNT,
    TABLE_PARAM_NAMES,
    TABLE_TOP_THICKNESS,
    TABLE_LEG_RADIUS,
)

from src.objects.chair.sdf import (
    compute_chair_sdf,
    compute_chair_priors,
    CHAIR_PARAM_COUNT,
    CHAIR_PARAM_NAMES,
    CHAIR_SEAT_THICKNESS,
    CHAIR_BACK_THICKNESS,
)

from src.objects.tv.sdf import (
    compute_tv_sdf,
    compute_tv_priors,
    TV_PARAM_COUNT,
    TV_PARAM_NAMES,
    TV_TYPICAL_ASPECT,
)

# Shared priors (inter-object and temporal smoothness)
from src.objects.shared_priors import (
    SharedPriorConfig,
    DEFAULT_SHARED_PRIOR_CONFIG,
    compute_overlap_prior,
    compute_smoothness_prior,
)

# Shared constants
SDF_SMOOTH_K = 0.02
SDF_INSIDE_SCALE = 0.3


# =============================================================================
# OBJECT REGISTRY
# =============================================================================

OBJECT_REGISTRY: Dict[str, Dict[str, Any]] = {
    'box': {
        'sdf': compute_box_sdf,
        'prior': compute_box_priors,
        'param_count': BOX_PARAM_COUNT,
        'param_names': BOX_PARAM_NAMES,
        'description': 'Oriented 3D box sitting on floor',
    },
    'cylinder': {
        'sdf': compute_cylinder_sdf,
        'prior': compute_cylinder_priors,
        'param_count': CYLINDER_PARAM_COUNT,
        'param_names': CYLINDER_PARAM_NAMES,
        'description': 'Vertical cylinder sitting on floor',
    },
    'table': {
        'sdf': compute_table_sdf,
        'prior': compute_table_priors,
        'param_count': TABLE_PARAM_COUNT,
        'param_names': TABLE_PARAM_NAMES,
        'description': 'Table with box top and 4 cylindrical legs',
    },
    'chair': {
        'sdf': compute_chair_sdf,
        'prior': compute_chair_priors,
        'param_count': CHAIR_PARAM_COUNT,
        'param_names': CHAIR_PARAM_NAMES,
        'description': 'Chair with seat and backrest',
    },
    'tv': {
        'sdf': compute_tv_sdf,
        'prior': compute_tv_priors,
        'param_count': TV_PARAM_COUNT,
        'param_names': TV_PARAM_NAMES,
        'description': 'Thin TV panel with stand',
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sdf_function(object_type: str) -> Callable:
    """Get the SDF function for a given object type."""
    if object_type not in OBJECT_REGISTRY:
        raise ValueError(f"Unknown object type: {object_type}. "
                        f"Available: {list(OBJECT_REGISTRY.keys())}")
    return OBJECT_REGISTRY[object_type]['sdf']


def get_prior_function(object_type: str) -> Callable:
    """Get the prior function for a given object type."""
    if object_type not in OBJECT_REGISTRY:
        raise ValueError(f"Unknown object type: {object_type}. "
                        f"Available: {list(OBJECT_REGISTRY.keys())}")
    return OBJECT_REGISTRY[object_type]['prior']


def get_object_info(object_type: str) -> Dict[str, Any]:
    """Get full info for an object type."""
    if object_type not in OBJECT_REGISTRY:
        raise ValueError(f"Unknown object type: {object_type}. "
                        f"Available: {list(OBJECT_REGISTRY.keys())}")
    return OBJECT_REGISTRY[object_type]


def list_object_types() -> list:
    """List all available object types."""
    return list(OBJECT_REGISTRY.keys())
