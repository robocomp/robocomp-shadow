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

#include <Gridder.h>
#include <Lidar3D.h>


#define CHECK_PERIOD 5000
#define BASIC_PERIOD 100


using TuplePrx = std::tuple<RoboCompLidar3D::Lidar3DPrxPtr,RoboCompLidar3D::Lidar3DPrxPtr>;


class GenericWorker : public QWidget, public Ui_guiDlg
{
Q_OBJECT
public:
	GenericWorker(TuplePrx tprx);
	virtual ~GenericWorker();
	virtual void killYourSelf();
	virtual void setPeriod(int p);

	virtual bool setParams(RoboCompCommonBehavior::ParameterList params) = 0;
	QMutex *mutex;


	RoboCompLidar3D::Lidar3DPrxPtr lidar3d_proxy;
	RoboCompLidar3D::Lidar3DPrxPtr lidar3d1_proxy;

	virtual bool Gridder_IsPathBlocked(RoboCompGridder::TPath path) = 0;
	virtual bool Gridder_LineOfSightToTarget(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, float robot_radius) = 0;
	virtual RoboCompGridder::TPoint Gridder_getClosestFreePoint(RoboCompGridder::TPoint source) = 0;
	virtual RoboCompGridder::TDimensions Gridder_getDimensions() = 0;
	virtual RoboCompGridder::Result Gridder_getPaths(RoboCompGridder::TPoint source, RoboCompGridder::TPoint target, int max_paths, bool try_closest_free_point, bool target_is_human) = 0;
	virtual bool Gridder_setGridDimensions(RoboCompGridder::TDimensions dimensions) = 0;

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
