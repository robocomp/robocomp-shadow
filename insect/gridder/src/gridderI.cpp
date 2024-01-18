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

bool GridderI::LineOfSightToTarget(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, float robot_radius, const Ice::Current&)
{
	return worker->Gridder_LineOfSightToTarget(source, target, robot_radius);
}

RoboCompGridder::TPoint GridderI::getClosestFreePoint(RoboCompGridder::TPoint source, const Ice::Current&)
{
	return worker->Gridder_getClosestFreePoint(source);
}

RoboCompGridder::TDimensions GridderI::getDimensions(const Ice::Current&)
{
	return worker->Gridder_getDimensions();
}

RoboCompGridder::Result GridderI::getPaths(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, int max_paths, bool try_closest_free_point, bool target_is_human, const Ice::Current&)
{
	return worker->Gridder_getPaths(source, target, max_paths, try_closest_free_point, target_is_human);
}

bool GridderI::setGridDimensions(RoboCompGridder::TDimensions dimensions, const Ice::Current&)
{
	return worker->Gridder_setGridDimensions(dimensions);
}

