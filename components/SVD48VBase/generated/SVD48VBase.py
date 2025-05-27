#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# \mainpage RoboComp::SVD48VBase
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
# Just: "${PATH_TO_BINARY}/SVD48VBase --Ice.Config=${PATH_TO_CONFIG_FILE}"
#
# \subsection running_ssec Once running
#
#
#
import argparse
# Ctrl+c handling
import signal
import sys
import os
from pathlib import Path

try:
    ROBOCOMP = os.environ['ROBOCOMP']
except KeyError:
    print('ROBOCOMP environment variable not set, using the default value /home/robocomp/robocomp')
    ROBOCOMP = '/home/robocomp/robocomp'

sys.path.append(str(os.path.join(ROBOCOMP, "classes/ConfigLoader")))
from ConfigLoader import ConfigLoader

sys.path.append(str(Path(__file__).parent.parent))
from src.specificworker import *
import interfaces


from rich.console import Console
console = Console()

#SIGNALS handler
def sigint_handler(*args):
    QtCore.QCoreApplication.quit()
    worker.__del__()


if __name__ == '__main__':
    app = QtCore.QCoreApplication(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', nargs='?', type=str, default='etc/config')
    parser.add_argument('--startup-check', action='store_true')

    args = parser.parse_args()
    configData = ConfigLoader.load_config(args.configfile)
    interface_manager = interfaces.InterfaceManager(configData)

    if interface_manager.status == 0:
        worker = SpecificWorker(interface_manager.get_proxies_map(), configData, args.startup_check)
        worker.setParams(configData)
    else:
        print("Error getting required connections, check config file")
        sys.exit(-1)

    interface_manager.set_default_hanlder(worker, configData)
    signal.signal(signal.SIGINT, sigint_handler)
    app.exec()
    interface_manager.destroy()
