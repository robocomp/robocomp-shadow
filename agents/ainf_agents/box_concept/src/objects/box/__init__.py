#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Box Object Package

Contains all files for the Box object type:
- sdf.py: SDF and prior functions
- belief.py: BoxBelief class
- manager.py: BoxManager class
"""

from src.objects.box.sdf import (
    compute_box_sdf,
    compute_box_priors,
    BOX_PARAM_COUNT,
    BOX_PARAM_NAMES,
)

from src.objects.box.belief import BoxBelief, BoxBeliefConfig
from src.objects.box.manager import BoxManager
