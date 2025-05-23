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
#ifndef GENERICWORKER_H
#define GENERICWORKER_H

#include "config.h"
#include <stdint.h>
#include <qlog/qlog.h>

#if Qt5_FOUND
	#include <QtWidgets>
#else
	#include <QtGui>
#endif
#include <ui_mainUI.h>
#include <CommonBehavior.h>

#include <FullPoseEstimation.h>
#include <GridPlanner.h>
#include <Gridder.h>
#include <Gridder.h>
#include <Lidar3D.h>
#include <LidarOdometry.h>
#include <SegmentatorTrackingPub.h>
#include <VisualElementsPub.h>
#include <Webots2Robocomp.h>


#define CHECK_PERIOD 5000
#define BASIC_PERIOD 100


using TuplePrx = std::tuple<RoboCompGridPlanner::GridPlannerPrxPtr,RoboCompGridPlanner::GridPlannerPrxPtr,RoboCompGridder::GridderPrxPtr,RoboCompLidar3D::Lidar3DPrxPtr,RoboCompLidarOdometry::LidarOdometryPrxPtr,RoboCompWebots2Robocomp::Webots2RobocompPrxPtr>;


class GenericWorker : public QMainWindow, public Ui_guiDlg
{
Q_OBJECT
public:
	GenericWorker(TuplePrx tprx);
	virtual ~GenericWorker();
	virtual void killYourSelf();
	virtual void setPeriod(int p);

	virtual bool setParams(RoboCompCommonBehavior::ParameterList params) = 0;
	QMutex *mutex;


	RoboCompGridPlanner::GridPlannerPrxPtr gridplanner_proxy;
	RoboCompGridPlanner::GridPlannerPrxPtr gridplanner1_proxy;
	RoboCompGridder::GridderPrxPtr gridder_proxy;
	RoboCompLidar3D::Lidar3DPrxPtr lidar3d_proxy;
	RoboCompLidarOdometry::LidarOdometryPrxPtr lidarodometry_proxy;
	RoboCompWebots2Robocomp::Webots2RobocompPrxPtr webots2robocomp_proxy;

	virtual void SegmentatorTrackingPub_setTrack (RoboCompVisualElementsPub::TObject target) = 0;

protected:

	QTimer timer;
	int Period;

private:


public slots:
	virtual void compute() = 0;
	virtual void initialize(int period) = 0;
	
signals:
	void kill();
};

#endif
