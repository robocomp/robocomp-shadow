#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2024 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import g2o

import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 2000
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        """Destructor"""

    def setParams(self, params):

        self.max_iterations = 100
        # Load src/spere2500.txt file as string
        with open('test.txt', 'r') as file:
            self.data = file.read()
        return True


    @QtCore.Slot()
    def compute(self):
        print('/////////////////////////////////////////////////////////////////////////////////////////////////////////////////')
        # Call the G2Ooptimizer_optimize method from the G2Ooptimizer interface
        result = self.G2Ooptimizer_optimize(self.data)
        print(result)

        return True

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of optimize method from G2Ooptimizer interface
    #
    def G2Ooptimizer_optimize(self, trajectory):
        ret = str()
        # Create a .g2o file with the trajectory
        with open('trajectory.g2o', 'w') as file:
            file.write(trajectory)

        # solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.BlockSolverX(g2o.LinearSolverEigenX())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)

        optimizer = g2o.SparseOptimizer()
        optimizer.set_verbose(True)
        optimizer.set_algorithm(solver)

        # Load the trajectory from the .g2o file
        optimizer.load('trajectory.g2o')

        optimizer.initialize_optimization()
        optimizer.optimize(self.max_iterations)
        # save the optimized trajectory to a file
        optimizer.save('optimized_trajectory.g2o')

        # return the optimized trajectory as a string
        with open('optimized_trajectory.g2o', 'r') as file:
            ret = file.read()

        # delete temporary input and output files
        os.remove('trajectory.g2o')
        os.remove('optimized_trajectory.g2o')

        return ret
    # ===================================================================
    # ===================================================================



