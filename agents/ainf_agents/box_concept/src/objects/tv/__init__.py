#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
TV Object Package

Contains all files for the TV object type:
- sdf.py: SDF and prior functions for thin panel TV model
- belief.py: TVBelief class
- manager.py: TVManager class

TV Characteristics:
- Thin rectangular panel (width >> depth)
- Screen aspect ratio ~16:9 (width/height)
- Typically aligned with walls
- Mounted at various heights
"""

from src.objects.tv.sdf import (
    compute_tv_sdf,
    compute_tv_priors,
    TV_PARAM_COUNT,
    TV_PARAM_NAMES,
    TV_TYPICAL_ASPECT,
)

from src.objects.tv.belief import TVBelief, TVBeliefConfig
from src.objects.tv.manager import TVManager

