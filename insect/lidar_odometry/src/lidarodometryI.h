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
#ifndef LIDARODOMETRY_H
#define LIDARODOMETRY_H

// Ice includes
#include <Ice/Ice.h>
#include <LidarOdometry.h>

#include <config.h>
#include "genericworker.h"


class LidarOdometryI : public virtual RoboCompLidarOdometry::LidarOdometry
{
public:
	LidarOdometryI(GenericWorker *_worker);
	~LidarOdometryI();

	RoboCompFullPoseEstimation::FullPoseEuler getFullPoseEuler(const Ice::Current&);
	RoboCompFullPoseEstimation::FullPoseMatrix getFullPoseMatrix(const Ice::Current&);
	RoboCompLidarOdometry::PoseAndChange getPoseAndChange(const Ice::Current&);
	void reset(const Ice::Current&);

private:

	GenericWorker *worker;

};

#endif
