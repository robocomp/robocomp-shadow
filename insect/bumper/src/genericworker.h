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

#include <Camera360RGB.h>
#include <GenericBase.h>
#include <GridPlanner.h>
#include <Lidar3D.h>
#include <OmniRobot.h>
#include <Person.h>


#define CHECK_PERIOD 5000
#define BASIC_PERIOD 100


using TuplePrx = std::tuple<RoboCompCamera360RGB::Camera360RGBPrxPtr,RoboCompLidar3D::Lidar3DPrxPtr,RoboCompOmniRobot::OmniRobotPrxPtr,RoboCompPerson::PersonPrxPtr>;


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


	RoboCompCamera360RGB::Camera360RGBPrxPtr camera360rgb_proxy;
	RoboCompLidar3D::Lidar3DPrxPtr lidar3d_proxy;
	RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy;
	RoboCompPerson::PersonPrxPtr person_proxy;

	virtual void GridPlanner_setPlan(RoboCompGridPlanner::TPlan plan) = 0;

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
