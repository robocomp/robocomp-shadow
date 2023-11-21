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
#include <door_detector_alej.h>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/action_node.h>
//#include <behaviortree_cpp/dummy_nodes.h>

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

    void draw_floor_line(const vector<Eigen::Vector2f> &lines);

public slots:
	void compute();
	int startup_check();
	void initialize(int period);
private:
//	std::shared_ptr < InnerModel > innerModel;
	bool startup_check_flag;

    // Crear un objeto de la clase DoorDetector
//    DoorDetector detector;
    Door_detector detector;
    AbstractGraphicViewer *viewer;

    float min_distance;
    int current_phi;
    RoboCompLidar3D::TPoint aux_point;
    BT::BehaviorTreeFactory factory;
    BT::Tree tree;
};

//Blocking nodes they do not return RUNNING, only SUCCESS or FAILURE
class LookForNewDoor : public BT::SyncActionNode
{
    public:
    LookForNewDoor(const std::string &name) : BT::SyncActionNode(name, {})
    {
    }
    // You must override the virtual function tick()
    BT::NodeStatus tick() override
    {
        std::cout << "LookForNewDoor: " << this->name() << std::endl;
        return BT::NodeStatus::SUCCESS;
    }
};

//Blocking nodes they do not return RUNNING, only SUCCESS or FAILURE
class GoThroughDoor : public BT::SyncActionNode
{
public:
    GoThroughDoor(const std::string &name) : BT::SyncActionNode(name, {})
    {
    }
    // You must override the virtual function tick()
    BT::NodeStatus tick() override
    {
        std::cout << "GoThroughDoor: " << this->name() << std::endl;
        return BT::NodeStatus::SUCCESS;
    }
};

//Blocking nodes they do not return RUNNING, only SUCCESS or FAILURE
class GoMiddleOfTheRoom : public BT::SyncActionNode
{
public:
    GoMiddleOfTheRoom(const std::string &name) : BT::SyncActionNode(name, {})
    {
    }
    // You must override the virtual function tick()
    BT::NodeStatus tick() override
    {
        std::cout << "GoMiddleOfTheRoom: " << this->name() << std::endl;
        return BT::NodeStatus::SUCCESS;
    }
};
#endif
