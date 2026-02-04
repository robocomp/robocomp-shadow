#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Chair Object Package

Contains all files for the Chair object type:
- sdf.py: SDF and prior functions
- belief.py: ChairBelief class
- manager.py: ChairManager class
"""

from src.objects.chair.sdf import (
    compute_chair_sdf,
    compute_chair_priors,
    CHAIR_PARAM_COUNT,
    CHAIR_PARAM_NAMES,
    CHAIR_SEAT_THICKNESS,
    CHAIR_BACK_THICKNESS,
    CHAIR_LEG_RADIUS,
    CHAIR_LEG_INSET,
)

from src.objects.chair.belief import ChairBelief, ChairBeliefConfig
from src.objects.chair.manager import ChairManager
