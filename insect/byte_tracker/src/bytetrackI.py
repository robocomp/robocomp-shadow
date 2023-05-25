#
#    Copyright (C) 2023 by YOUR NAME HERE
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


Ice.loadSlice("-I ./src/ --all ./src/ByteTrack.ice")

from RoboCompByteTrack import *

class ByteTrackI(ByteTrack):
    def __init__(self, worker):
        self.worker = worker


    def allTargets(self, c):
        return self.worker.ByteTrack_allTargets()

    def getTargets(self, objects, c):
        return self.worker.ByteTrack_getTargets(objects)

    def getTargetswithdepth(self, objects, depth, c):
        return self.worker.ByteTrack_getTargetswithdepth(objects, depth)

    def setTargets(self, objects, sender, c):
        return self.worker.ByteTrack_setTargets(objects, sender)
