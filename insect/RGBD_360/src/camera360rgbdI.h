/*
 *    Copyright (C) 2023 by YOUR NAME HERE
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
#ifndef CAMERA360RGBD_H
#define CAMERA360RGBD_H

// Ice includes
#include <Ice/Ice.h>
#include <Camera360RGBD.h>

#include <config.h>
#include "genericworker.h"


class Camera360RGBDI : public virtual RoboCompCamera360RGBD::Camera360RGBD
{
public:
	Camera360RGBDI(GenericWorker *_worker);
	~Camera360RGBDI();

	RoboCompCamera360RGBD::TRGBD getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight, const Ice::Current&);

private:

	GenericWorker *worker;

};

#endif
