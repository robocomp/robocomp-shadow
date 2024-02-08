/*
 *    Copyright (C) 2024 by YOUR NAME HERE
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
#include "gridderI.h"

GridderI::GridderI(GenericWorker *_worker)
{
	worker = _worker;
}


GridderI::~GridderI()
{
}


bool GridderI::IsPathBlocked(RoboCompGridder::TPath path, const Ice::Current&)
{
	return worker->Gridder_IsPathBlocked(path);
}

bool GridderI::LineOfSightToTarget(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, float robotRadius, const Ice::Current&)
{
	return worker->Gridder_LineOfSightToTarget(source, target, robotRadius);
}

RoboCompGridder::TPoint GridderI::getClosestFreePoint(RoboCompGridder::TPoint source, const Ice::Current&)
{
	return worker->Gridder_getClosestFreePoint(source);
}

RoboCompGridder::TDimensions GridderI::getDimensions(const Ice::Current&)
{
	return worker->Gridder_getDimensions();
}

RoboCompGridder::Result GridderI::getPaths(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, int maxPaths, bool tryClosestFreePoint, bool targetIsHuman, const Ice::Current&)
{
	return worker->Gridder_getPaths(source, target, maxPaths, tryClosestFreePoint, targetIsHuman);
}

bool GridderI::setGridDimensions(RoboCompGridder::TDimensions dimensions, const Ice::Current&)
{
	return worker->Gridder_setGridDimensions(dimensions);
}

RoboCompGridder::Result GridderI::setLocationAndGetPath(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, bool setFree, RoboCompGridder::TPoint obstacle, const Ice::Current&)
{
	return worker->Gridder_setLocationAndGetPath(source, target, setFree, obstacle);
}

