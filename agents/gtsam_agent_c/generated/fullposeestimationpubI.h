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
#ifndef FULLPOSEESTIMATIONPUB_H
#define FULLPOSEESTIMATIONPUB_H

// Ice includes
#include <Ice/Ice.h>
#include <FullPoseEstimationPub.h>

#include "../src/specificworker.h"


class FullPoseEstimationPubI : public virtual RoboCompFullPoseEstimationPub::FullPoseEstimationPub
{
public:
	FullPoseEstimationPubI(GenericWorker *_worker, const size_t id);
	~FullPoseEstimationPubI();

	void newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose, const Ice::Current&);

private:

	GenericWorker *worker;
	size_t id;

	// Array handlers for each method
	std::array<std::function<void(RoboCompFullPoseEstimation::FullPoseEuler)>, 1> newFullPoseHandlers;

};

#endif
