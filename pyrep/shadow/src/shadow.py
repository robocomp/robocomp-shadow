#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
#    Copyright (C) 2020 by YOUR NAME HERE
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

# \mainpage RoboComp::omnirobotPyrep
#
# \section intro_sec Introduction
#
# Some information about the component...
#
# \section interface_sec Interface
#
# Descroption of the interface provided...
#
# \section install_sec Installation
#
# \subsection install1_ssec Software depencences
# Software dependences....
#
# \subsection install2_ssec Compile and install
# How to compile/install the component...
#
# \section guide_sec User guide
#
# \subsection config_ssec Configuration file
#
# <p>
# The configuration file...
# </p>
#
# \subsection execution_ssec Execution
#
# Just: "${PATH_TO_BINARY}/omnirobotPyrep --Ice.Config=${PATH_TO_CONFIG_FILE}"
#
# \subsection running_ssec Once running
#
#
#

import sys, traceback, IceStorm, time, os, copy
from termcolor import colored

# Ctrl+c handling
import signal

#from PySide2 import QtCore

from specificworker import *

class CommonBehaviorI(RoboCompCommonBehavior.CommonBehavior):
    def __init__(self, _handler):
        self.handler = _handler
    def getFreq(self, current = None):
        self.handler.getFreq()
    def setFreq(self, freq, current = None):
        self.handler.setFreq()
    def timeAwake(self, current = None):
        try:
            return self.handler.timeAwake()
        except:
            print('Problem getting timeAwake')
    def killYourSelf(self, current = None):
        self.handler.killYourSelf()
    def getAttrList(self, current = None):
        try:
            return self.handler.getAttrList()
        except:
            print('Problem getting getAttrList')
            traceback.print_exc()
            status = 1
            return

#SIGNALS handler
def sigint_handler(*args):
    #QtCore.QCoreApplication.quit()
    pass
    
if __name__ == '__main__':
    #app = QtCore.QCoreApplication(sys.argv)
    params = copy.deepcopy(sys.argv)
    if len(params) > 1:
        if not params[1].startswith('--Ice.Config='):
            params[1] = '--Ice.Config=' + params[1]
    elif len(params) == 1:
        params.append('--Ice.Config=etc/config')
    ic = Ice.initialize(params)
    status = 0
    mprx = {}
    parameters = {}
    for i in ic.getProperties():
        parameters[str(i)] = str(ic.getProperties().getProperty(i))

    # Topic Manager
    proxy = ic.getProperties().getProperty("TopicManager.Proxy")
    obj = ic.stringToProxy(proxy)
    try:
        topicManager = IceStorm.TopicManagerPrx.checkedCast(obj)
    except Ice.ConnectionRefusedException as e:
        print(colored('Cannot connect to rcnode! This must be running to use pub/sub.', 'red'))
        exit(1)

    if status == 0:
        worker = SpecificWorker(mprx)
        worker.setParams(parameters)
    else:
        print("Error getting required connections, check config file")
        sys.exit(-1)

    adapter1 = ic.createObjectAdapter('CameraRGBDSimple')
    adapter1.add(camerargbdsimpleI.CameraRGBDSimpleI(worker), ic.stringToIdentity('camerargbdsimple'))
    adapter1.activate()

    adapter2 = ic.createObjectAdapter('Camera360RGB')
    adapter2.add(camera360rgbI.Camera360RGBI(worker), ic.stringToIdentity('camera360rgb'))
    adapter2.activate()

    adapter3 = ic.createObjectAdapter('CameraSimple')
    adapter3.add(camerasimpleI.CameraSimpleI(worker), ic.stringToIdentity('camerasimple'))
    adapter3.activate()

    adapter4 = ic.createObjectAdapter('Laser')
    adapter4.add(laserI.LaserI(worker), ic.stringToIdentity('laser'))
    adapter4.activate()

    adapter5 = ic.createObjectAdapter('OmniRobot')
    adapter5.add(omnirobotI.OmniRobotI(worker), ic.stringToIdentity('omnirobot'))
    adapter5.activate()

    adapter6 = ic.createObjectAdapter('FullPoseEstimation')
    adapter6.add(fullposeestimationI.FullPoseEstimationI(worker), ic.stringToIdentity('fullposeestimation'))
    adapter6.activate()

    adapter7 = ic.createObjectAdapter('BatteryStatus')
    adapter7.add(batterystatusI.BatteryStatusI(worker), ic.stringToIdentity('batterystatus'))
    adapter7.activate()

    adapter8 = ic.createObjectAdapter('RSSIStatus')
    adapter8.add(rssistatusI.RSSIStatusI(worker), ic.stringToIdentity('rssistatus'))
    adapter8.activate()

    adapter9 = ic.createObjectAdapter('JointMotorSimple')
    adapter9.add(jointmotorsimpleI.JointMotorSimpleI(worker), ic.stringToIdentity('jointmotorsimple'))
    adapter9.activate()

    adapter10 = ic.createObjectAdapter('CoppeliaUtils')
    adapter10.add(coppeliautilsI.CoppeliaUtilsI(worker), ic.stringToIdentity('coppeliautils'))
    adapter10.activate()

    adapter11 = ic.createObjectAdapter('BillCoppelia')
    adapter11.add(billcoppeliaI.BillCoppeliaI(worker), ic.stringToIdentity('billcoppelia'))
    adapter11.activate()

    adapter12 = ic.createObjectAdapter('Lidar3D')
    adapter12.add(lidar3dI.Lidar3DI(worker), ic.stringToIdentity('lidar3d'))
    adapter12.activate()

    JoystickAdapter_adapter = ic.createObjectAdapter("JoystickAdapterTopic")
    joystickadapterI_ = joystickadapterI.JoystickAdapterI(worker)
    joystickadapter_proxy = JoystickAdapter_adapter.addWithUUID(joystickadapterI_).ice_oneway()

    subscribeDone = False
    while not subscribeDone:
        try:
            joystickadapter_topic = topicManager.retrieve("JoystickAdapter")
            subscribeDone = True
        except Ice.Exception as e:
            print("Error. Topic does not exist (creating)")
            time.sleep(1)
            try:
                joystickadapter_topic = topicManager.create("JoystickAdapter")
                subscribeDone = True
            except:
                print("Error. Topic could not be created. Exiting")
                status = 0
    qos = {}
    joystickadapter_topic.subscribeAndGetPublisher(qos, joystickadapter_proxy)
    JoystickAdapter_adapter.activate()

    signal.signal(signal.SIGINT, sigint_handler)
    #app.exec_()
    worker.compute()

    if ic:
        try:
            ic.destroy()
        except:
            traceback.print_exc()
            status = 1

