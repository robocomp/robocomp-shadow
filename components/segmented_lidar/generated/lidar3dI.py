#
#    Copyright (C) 2025 by YOUR NAME HERE
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

import sys, os, Ice

ROBOCOMP = ''
try:
    ROBOCOMP = os.environ['ROBOCOMP']
except:
    print('$ROBOCOMP environment variable not set, using the default value /opt/robocomp')
    ROBOCOMP = '/opt/robocomp'
if len(ROBOCOMP)<1:
    raise RuntimeError('ROBOCOMP environment variable not set! Exiting.')


Ice.loadSlice("-I ./src/ --all ./src/Lidar3D.ice")

from RoboCompLidar3D import *

class Lidar3DI(Lidar3D):
    def __init__(self, worker):
        self.worker = worker


    def getLidarData(self, name, start, len, decimationDegreeFactor, c):
        return self.worker.Lidar3D_getLidarData(name, start, len, decimationDegreeFactor)

    def getLidarDataArrayProyectedInImage(self, name, c):
        return self.worker.Lidar3D_getLidarDataArrayProyectedInImage(name)

    def getLidarDataByCategory(self, categories, timestamp, c):
        return self.worker.Lidar3D_getLidarDataByCategory(categories, timestamp)

    def getLidarDataProyectedInImage(self, name, c):
        return self.worker.Lidar3D_getLidarDataProyectedInImage(name)

    def getLidarDataWithThreshold2d(self, name, distance, decimationDegreeFactor, c):
        return self.worker.Lidar3D_getLidarDataWithThreshold2d(name, distance, decimationDegreeFactor)
