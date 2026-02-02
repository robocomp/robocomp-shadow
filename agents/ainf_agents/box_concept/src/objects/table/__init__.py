#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Table Object Package

Contains all files for the Table object type:
- sdf.py: SDF and prior functions
- belief.py: TableBelief class
- manager.py: TableManager class
"""

from src.objects.table.sdf import (
    compute_table_sdf,
    compute_table_priors,
    TABLE_PARAM_COUNT,
    TABLE_PARAM_NAMES,
    TABLE_TOP_THICKNESS,
    TABLE_LEG_RADIUS,
)

from src.objects.table.belief import TableBelief, TableBeliefConfig
from src.objects.table.manager import TableManager
