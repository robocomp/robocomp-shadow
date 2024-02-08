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
#ifndef GRIDDER_H
#define GRIDDER_H

// Ice includes
#include <Ice/Ice.h>
#include <Gridder.h>

#include <config.h>
#include "genericworker.h"


class GridderI : public virtual RoboCompGridder::Gridder
{
public:
	GridderI(GenericWorker *_worker);
	~GridderI();

	bool IsPathBlocked(RoboCompGridder::TPath path, const Ice::Current&);
	bool LineOfSightToTarget(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, float robotRadius, const Ice::Current&);
	RoboCompGridder::TPoint getClosestFreePoint(RoboCompGridder::TPoint source, const Ice::Current&);
	RoboCompGridder::TDimensions getDimensions(const Ice::Current&);
	RoboCompGridder::Result getPaths(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, int maxPaths, bool tryClosestFreePoint, bool targetIsHuman, const Ice::Current&);
	bool setGridDimensions(RoboCompGridder::TDimensions dimensions, const Ice::Current&);
	RoboCompGridder::Result setLocationAndGetPath(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, bool setFree, RoboCompGridder::TPoint obstacle, const Ice::Current&);

private:

	GenericWorker *worker;

};

#endif
