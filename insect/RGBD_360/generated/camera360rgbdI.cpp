/*
 *    Copyright (C) 2026 by YOUR NAME HERE
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
#include "camera360rgbdI.h"

Camera360RGBDI::Camera360RGBDI(GenericWorker *_worker, const size_t id): worker(_worker), id(id)
{
	getROIHandlers = {
		[this](auto a, auto b, auto c, auto d, auto e, auto f) { return worker->Camera360RGBD_getROI(a, b, c, d, e, f); }
	};

}


Camera360RGBDI::~Camera360RGBDI()
{
}


RoboCompCamera360RGBD::TRGBD Camera360RGBDI::getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight, const Ice::Current&)
{

    #ifdef HIBERNATION_ENABLED
		worker->hibernationTick();
	#endif
    
	if (id < getROIHandlers.size())
		return  getROIHandlers[id](cx, cy, sx, sy, roiwidth, roiheight);
	else
		throw std::out_of_range("Invalid getROI id: " + std::to_string(id));

}

