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
#ifndef WEBOTS2ROBOCOMP_H
#define WEBOTS2ROBOCOMP_H

// Ice includes
#include <Ice/Ice.h>
#include <Webots2Robocomp.h>

#include <config.h>
#include "genericworker.h"


class Webots2RobocompI : public virtual RoboCompWebots2Robocomp::Webots2Robocomp
{
public:
	Webots2RobocompI(GenericWorker *_worker);
	~Webots2RobocompI();

	void resetWebots(const Ice::Current&);
	void setPathToHuman(int humanId, RoboCompGridder::TPath path, const Ice::Current&);

private:

	GenericWorker *worker;

};

#endif
