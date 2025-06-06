/*
 *    Copyright (C) 2025 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "lidar3dpubI.h"

Lidar3DPubI::Lidar3DPubI(GenericWorker *_worker, const size_t id): worker(_worker), id(id)
{
	pushLidarDataHandlers = {
		[this](auto a) { return worker->Lidar3DPub_pushLidarData(a); }
	};

}


Lidar3DPubI::~Lidar3DPubI()
{
}


void Lidar3DPubI::pushLidarData(RoboCompLidar3D::TDataCategory lidarData, const Ice::Current&)
{

    #ifdef HIBERNATION_ENABLED
		worker->hibernationTick();
	#endif
    
	if (id < pushLidarDataHandlers.size())
		 pushLidarDataHandlers[id](lidarData);
	else
		throw std::out_of_range("Invalid pushLidarData id: " + std::to_string(id));

}

