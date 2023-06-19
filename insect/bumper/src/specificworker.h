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

/**
	\brief
	@author authorname
*/

#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include <Eigen/Dense>
#include <abstract_graphic_viewer/abstract_graphic_viewer.h>
#include <fps/fps.h>


class SpecificWorker : public GenericWorker
{
	Q_OBJECT
	public:
		SpecificWorker(TuplePrx tprx, bool startup_check);
		~SpecificWorker();
		bool setParams(RoboCompCommonBehavior::ParameterList params);

		void OmniRobot_correctOdometer(int x, int z, float alpha);
		void OmniRobot_getBasePose(int &x, int &z, float &alpha);
		void OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state);
		void OmniRobot_resetOdometer();
		void OmniRobot_setOdometer(RoboCompGenericBase::TBaseState state);
		void OmniRobot_setOdometerPose(int x, int z, float alpha);
		void OmniRobot_setSpeedBase(float advx, float advz, float rot);
		void OmniRobot_stopBase();

		void SegmentatorTrackingPub_setTrack(RoboCompVisualElements::TObject target);

	public slots:
		void compute();
		int startup_check();
		void initialize(int period);

	private:
		bool startup_check_flag;
		int DEGREES_NUMBER = 360;		// division of the circle
		int OUTER_RIG_DISTANCE = 1000;  // external maximum reach to search (mm)

   		int z_lidar_height = 750;
        std::vector<float> create_map_of_points();
        std::vector<Eigen::Vector3f> get_lidar_data();
		std::vector<float> map_of_points;

        // Viewer
		AbstractGraphicViewer *viewer;

};

#endif
